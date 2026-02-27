"""
arm_plan_cvae.py
----------------
Conditional VAE that learns a compact latent encoding of the arm motion plan.
Used as B6 baseline: the RL actor receives the CVAE latent (30D) instead of
predicted wrenches (B5) to test whether a generic learned latent can match
the physics-grounded wrench representation.

Architecture
~~~~~~~~~~~~
**Encoder** (used at RL inference — deterministic via mu):
  Input  : cat(obs_norm, plan_norm) = 185 dims
  Hidden : Linear(185->256) + LayerNorm(256) + ELU
           Linear(256->256) + LayerNorm(256) + ELU
  Output : mu(30), logvar(30)

**Decoder** (CVAE training only — reconstructs plan from z + obs):
  Input  : cat(z, obs_norm) = 145 dims
  Hidden : Linear(145->256) + LayerNorm(256) + ELU
           Linear(256->256) + LayerNorm(256) + ELU
  Output : plan_hat(70)

Parameter count:
  Encoder: 185*256+256 = 47,872  +  256*256+256 = 66,048
           + 2*(256*30+30) = 15,480  + LayerNorm = 1,024
           Total encoder: ~130,424
  Decoder: 145*256+256 = 37,376  +  256*256+256 = 66,048
           + 256*70+70 = 17,990    + LayerNorm = 1,024
           Total decoder: ~122,438
  Grand total: ~252,862

Usage
~~~~~
    # Training
    model = ArmPlanCVAE()
    z, plan_hat, mu, logvar = model.forward_train(obs, plan)
    loss = recon_loss(plan_hat, plan) + beta * kl_loss(mu, logvar)

    # RL inference (frozen)
    model.load_frozen("path/to/cvae_checkpoint.pt")
    latent = model(obs, plan)   # returns mu (deterministic), shape (B, 30)
"""

from typing import Tuple

import torch
import torch.nn as nn
from pathlib import Path


# ---------------------------------------------------------------------------
# Architecture constants (comparable to wrench predictor for fair comparison;
# encoder has 2 hidden layers vs predictor's 3, but output dim matches)
# ---------------------------------------------------------------------------
OBS_DIM = 115       # per-step actor obs
PLAN_DIM = 70       # H=5 future arm joint positions (5 x 14 joints)
LATENT_DIM = 30     # same as B5's predicted_wrench dim for obs parity
ENCODER_INPUT_DIM = OBS_DIM + PLAN_DIM   # 185
DECODER_INPUT_DIM = LATENT_DIM + OBS_DIM  # 145
ENCODER_HIDDEN = [256, 256]
DECODER_HIDDEN = [256, 256]

# Logvar clamping bounds to prevent extreme variance
LOGVAR_MIN = -20.0
LOGVAR_MAX = 2.0


class ArmPlanCVAE(nn.Module):
    """Conditional VAE for arm plan encoding.

    At RL inference time, ``forward()`` returns the encoder mean (mu) as a
    deterministic 30-dim latent — no stochastic sampling.  This gives the
    RL policy a stable observation signal.

    At CVAE training time, ``forward_train()`` uses the reparameterization
    trick for backprop through the sampling step.

    Args:
        obs_dim    (int): Observation dimension.  Default: 115.
        plan_dim   (int): Arm plan dimension.     Default: 70.
        latent_dim (int): Latent dimension.        Default: 30.

    Buffers (non-trainable, saved in state_dict):
        obs_mean   (obs_dim,)  : z-score mean for obs
        obs_std    (obs_dim,)  : z-score std for obs
        plan_mean  (plan_dim,) : z-score mean for arm plan
        plan_std   (plan_dim,) : z-score std for arm plan
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        plan_dim: int = PLAN_DIM,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()

        self._obs_dim = obs_dim
        self._plan_dim = plan_dim
        self._latent_dim = latent_dim

        encoder_input = obs_dim + plan_dim
        decoder_input = latent_dim + obs_dim

        # ------------------------------------------------------------------
        # Encoder: (obs, plan) -> (mu, logvar)
        # ------------------------------------------------------------------
        self.encoder_net = nn.Sequential(
            nn.Linear(encoder_input, ENCODER_HIDDEN[0]),
            nn.LayerNorm(ENCODER_HIDDEN[0]),
            nn.ELU(),
            nn.Linear(ENCODER_HIDDEN[0], ENCODER_HIDDEN[1]),
            nn.LayerNorm(ENCODER_HIDDEN[1]),
            nn.ELU(),
        )
        self.mu_head = nn.Linear(ENCODER_HIDDEN[-1], latent_dim)
        self.logvar_head = nn.Linear(ENCODER_HIDDEN[-1], latent_dim)

        # ------------------------------------------------------------------
        # Decoder: (z, obs) -> plan_hat
        # ------------------------------------------------------------------
        self.decoder_net = nn.Sequential(
            nn.Linear(decoder_input, DECODER_HIDDEN[0]),
            nn.LayerNorm(DECODER_HIDDEN[0]),
            nn.ELU(),
            nn.Linear(DECODER_HIDDEN[0], DECODER_HIDDEN[1]),
            nn.LayerNorm(DECODER_HIDDEN[1]),
            nn.ELU(),
            nn.Linear(DECODER_HIDDEN[-1], plan_dim),
        )

        # ------------------------------------------------------------------
        # Normalization buffers (identity default: mean=0, std=1)
        # ------------------------------------------------------------------
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.register_buffer("plan_mean", torch.zeros(plan_dim))
        self.register_buffer("plan_std", torch.ones(plan_dim))

    # ------------------------------------------------------------------
    # Forward: RL inference (deterministic — returns mu only)
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
        """Encode (obs, plan) and return the deterministic latent (mu).

        Args:
            obs  (B, 115): Raw per-step actor obs.
            plan (B,  70): Arm plan (H=5 x 14 joints).

        Returns:
            mu (B, 30): Deterministic latent vector for RL policy.
        """
        obs_n = self._normalize(obs, self.obs_mean, self.obs_std)
        plan_n = self._normalize(plan, self.plan_mean, self.plan_std)

        h = self.encoder_net(torch.cat([obs_n, plan_n], dim=-1))
        mu = self.mu_head(h)
        return mu

    # ------------------------------------------------------------------
    # Forward: CVAE training (stochastic + reconstruction)
    # ------------------------------------------------------------------

    def forward_train(
        self, obs: torch.Tensor, plan: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full CVAE forward with reparameterization trick.

        Args:
            obs  (B, 115): Raw obs.
            plan (B,  70): Raw arm plan (also the reconstruction target).

        Returns:
            z        (B, 30): Sampled latent via reparameterization.
            plan_hat (B, 70): Reconstructed plan (in normalized space).
            mu       (B, 30): Encoder mean.
            logvar   (B, 30): Encoder log-variance (clamped).
        """
        obs_n = self._normalize(obs, self.obs_mean, self.obs_std)
        plan_n = self._normalize(plan, self.plan_mean, self.plan_std)

        # Encode
        h = self.encoder_net(torch.cat([obs_n, plan_n], dim=-1))
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(LOGVAR_MIN, LOGVAR_MAX)

        # Reparameterization trick
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Decode (condition on obs + sampled z)
        plan_hat = self.decoder_net(torch.cat([z, obs_n], dim=-1))

        return z, plan_hat, mu, logvar

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(
        x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
    ) -> torch.Tensor:
        """Z-score normalization with minimum std floor."""
        return (x - mean) / std.clamp(min=1e-6)

    def set_normalization_stats(
        self,
        obs_mean: torch.Tensor,
        obs_std: torch.Tensor,
        plan_mean: torch.Tensor,
        plan_std: torch.Tensor,
    ) -> None:
        """Set z-score statistics from training data.

        Call once after computing dataset statistics, before saving.
        """
        device = self.obs_mean.device
        self.obs_mean.copy_(obs_mean.to(device))
        self.obs_std.copy_(obs_std.to(device))
        self.plan_mean.copy_(plan_mean.to(device))
        self.plan_std.copy_(plan_std.to(device))

    # ------------------------------------------------------------------
    # Frozen-model loading (mirrors WrenchPredictor.load_frozen)
    # ------------------------------------------------------------------

    def load_frozen(self, path: str) -> None:
        """Load weights from checkpoint and freeze the model.

        After this call: eval mode, all requires_grad=False,
        normalization buffers populated from checkpoint.

        Args:
            path: Path to checkpoint (.pt). Accepts bare state_dict or
                  wrapped dict with "model_state_dict" / "state_dict" key.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        raw = torch.load(path, map_location="cpu", weights_only=True)

        if isinstance(raw, dict):
            if "model_state_dict" in raw:
                state_dict = raw["model_state_dict"]
            elif "state_dict" in raw:
                state_dict = raw["state_dict"]
            else:
                state_dict = raw
        else:
            raise KeyError(
                f"Unexpected checkpoint type: {type(raw)}. "
                "Expected a dict with 'model_state_dict', 'state_dict', "
                "or a bare state_dict."
            )

        self.load_state_dict(state_dict, strict=True)
        self._freeze()

    def _freeze(self) -> None:
        """Put model in eval mode and disable all gradient computation."""
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Return total (or trainable) parameter count."""
        params = (
            self.parameters()
            if not trainable_only
            else (p for p in self.parameters() if p.requires_grad)
        )
        return sum(p.numel() for p in params)

    def num_encoder_parameters(self) -> int:
        """Return encoder-only parameter count (active at RL inference)."""
        count = sum(p.numel() for p in self.encoder_net.parameters())
        count += sum(p.numel() for p in self.mu_head.parameters())
        count += sum(p.numel() for p in self.logvar_head.parameters())
        return count
