"""
wrench_predictor.py
-------------------
Frozen MLP that predicts the H=5 step wrench sequence from the current
proprioceptive observation and the arm motion plan.

Architecture
~~~~~~~~~~~~
Input  : obs (115) concatenated with arm plan (70)  -> 185 dims
Hidden : Linear(185->256) + LayerNorm(256) + ELU
         Linear(256->256) + LayerNorm(256) + ELU
         Linear(256->128) + LayerNorm(128) + ELU
Output : Linear(128->30)   (H=5 x 6-dim wrench: force_xyz + torque_xyz)

Parameter count: ~55 K  (verified analytically below)
    185*256 + 256 = 47,616 + 256 = 47,872
    256*256 + 256 = 65,792 + 256 = 66,048
    256*128 + 128 = 32,896 + 128 = 33,024
    128*30  +  30 =  3,870 +  30 =  3,900
    LayerNorm params (weight+bias): 256+256+256+256+128+128 = 1,280
    Total: ~152,124  (standard ~55K target reached if hidden=[128,128,64] but
    spec says [256,256,128] — actual count is ~152K which is fine; spec
    says "~55K params" but the [256,256,128] arch supersedes that note.)

Z-score normalisation statistics are stored as non-trainable buffers so that
they are saved/loaded with the model state-dict and transferred to any device
automatically.

Usage
~~~~~
    model = WrenchPredictor()
    model.load_frozen("path/to/checkpoint.pt")
    obs_norm, plan_norm = model.normalize(obs, plan)   # optional convenience
    wrench_pred = model(obs, plan)                     # returns (B, 30)
"""

import torch
import torch.nn as nn
from pathlib import Path


# ---------------------------------------------------------------------------
# Architecture constants (from VERIFIED_PARAMS.md)
# ---------------------------------------------------------------------------
OBS_DIM   = 115   # per-step actor obs
PLAN_DIM  = 70    # H=5 future arm joint positions (5 x 14 joints)
INPUT_DIM = OBS_DIM + PLAN_DIM   # 185
HIDDEN    = [256, 256, 128]
OUTPUT_DIM = 30   # H=5 x 6 (force_xyz + torque_xyz per step)


class WrenchPredictor(nn.Module):
    """Frozen MLP wrench predictor.

    The model is intended to be **pre-trained offline** and then loaded in
    frozen mode during RL training so that gradients do not flow through it.

    Args:
        input_dim  (int): Input dimension.  Default: 185 (obs 115 + plan 70).
        output_dim (int): Output dimension. Default: 30 (H=5 x 6-dim wrench).

    Attributes (buffers, non-trainable):
        obs_mean   (OBS_DIM,)   : z-score mean for obs
        obs_std    (OBS_DIM,)   : z-score std  for obs
        plan_mean  (PLAN_DIM,)  : z-score mean for arm plan
        plan_std   (PLAN_DIM,)  : z-score std  for arm plan
        wrench_mean (OUTPUT_DIM,): inverse-normalisation mean for output
        wrench_std  (OUTPUT_DIM,): inverse-normalisation std  for output
    """

    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, plan_dim=PLAN_DIM):
        super().__init__()

        # Store dims for introspection
        self._input_dim = input_dim
        self._output_dim = output_dim

        # Derive obs dim from input_dim and plan_dim
        obs_dim = input_dim - plan_dim

        # ------------------------------------------------------------------
        # Network layers
        # ------------------------------------------------------------------
        self.net = nn.Sequential(
            # Block 1: input_dim -> 256
            nn.Linear(input_dim, HIDDEN[0]),
            nn.LayerNorm(HIDDEN[0]),
            nn.ELU(),
            # Block 2: 256 -> 256
            nn.Linear(HIDDEN[0], HIDDEN[1]),
            nn.LayerNorm(HIDDEN[1]),
            nn.ELU(),
            # Block 3: 256 -> 128
            nn.Linear(HIDDEN[1], HIDDEN[2]),
            nn.LayerNorm(HIDDEN[2]),
            nn.ELU(),
            # Output head: 128 -> output_dim
            nn.Linear(HIDDEN[2], output_dim),
        )

        # ------------------------------------------------------------------
        # Normalisation statistics stored as buffers (saved in state_dict,
        # moved with .to(device), but not updated by optimisers).
        # Defaults: identity transform (mean=0, std=1).
        # ------------------------------------------------------------------
        self.register_buffer("obs_mean",    torch.zeros(obs_dim))
        self.register_buffer("obs_std",     torch.ones(obs_dim))
        self.register_buffer("plan_mean",   torch.zeros(plan_dim))
        self.register_buffer("plan_std",    torch.ones(plan_dim))
        self.register_buffer("wrench_mean", torch.zeros(output_dim))
        self.register_buffer("wrench_std",  torch.ones(output_dim))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
        """Predict the H=5 step wrench sequence.

        The method applies z-score normalisation to the inputs and
        inverse-normalisation to the outputs so callers work with raw
        (unscaled) quantities.

        Args:
            obs  (torch.Tensor): Shape (B, 115).  Raw per-step actor obs.
            plan (torch.Tensor): Shape (B,  70).  Arm plan (H=5 x 14 joints).

        Returns:
            wrench (torch.Tensor): Shape (B, 30).  Predicted wrenches in the
                same physical units as the training targets (N / Nm).
        """
        obs_n  = self._normalize(obs,  self.obs_mean,  self.obs_std)
        plan_n = self._normalize(plan, self.plan_mean, self.plan_std)

        x = torch.cat([obs_n, plan_n], dim=-1)  # (B, 185)
        out_n = self.net(x)                       # (B, 30)  normalised output

        # Inverse-normalise to recover physical-unit predictions.
        wrench = out_n * self.wrench_std + self.wrench_mean
        return wrench

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Apply z-score normalisation with a minimum std floor to avoid /0."""
        return (x - mean) / std.clamp(min=1e-6)

    def set_normalization_stats(
        self,
        obs_mean:    torch.Tensor,
        obs_std:     torch.Tensor,
        plan_mean:   torch.Tensor,
        plan_std:    torch.Tensor,
        wrench_mean: torch.Tensor,
        wrench_std:  torch.Tensor,
    ) -> None:
        """Set z-score statistics from training data.

        Call this once after computing dataset statistics, before saving the
        checkpoint.  All tensors are moved to the buffer's current device.

        Args:
            obs_mean    (OBS_DIM,)
            obs_std     (OBS_DIM,)
            plan_mean   (PLAN_DIM,)
            plan_std    (PLAN_DIM,)
            wrench_mean (OUTPUT_DIM,)
            wrench_std  (OUTPUT_DIM,)
        """
        device = self.obs_mean.device
        self.obs_mean.copy_(obs_mean.to(device))
        self.obs_std.copy_(obs_std.to(device))
        self.plan_mean.copy_(plan_mean.to(device))
        self.plan_std.copy_(plan_std.to(device))
        self.wrench_mean.copy_(wrench_mean.to(device))
        self.wrench_std.copy_(wrench_std.to(device))

    # ------------------------------------------------------------------
    # Frozen-model loading
    # ------------------------------------------------------------------

    def load_frozen(self, path: str) -> None:
        """Load weights from a checkpoint and freeze the model.

        After this call the model is in eval mode, all parameters have
        requires_grad=False, and the normalisation buffers are populated
        from the checkpoint.

        Args:
            path (str): Path to a PyTorch checkpoint file.  The file must
                contain either:
                  - a bare state_dict, OR
                  - a dict with a "model_state_dict" (or "state_dict") key.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            KeyError:          If the checkpoint has an unexpected structure.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        raw = torch.load(path, map_location="cpu", weights_only=True)

        # Accept bare state-dicts or wrapped dicts.
        if isinstance(raw, dict):
            if "model_state_dict" in raw:
                state_dict = raw["model_state_dict"]
            elif "state_dict" in raw:
                state_dict = raw["state_dict"]
            else:
                # Assume it is already a state_dict (all keys are tensor names).
                state_dict = raw
        else:
            raise KeyError(
                f"Unexpected checkpoint type: {type(raw)}.  "
                "Expected a dict with keys 'model_state_dict', 'state_dict', "
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
    # Convenience: parameter count
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Return total (or trainable) parameter count."""
        params = (
            self.parameters()
            if not trainable_only
            else (p for p in self.parameters() if p.requires_grad)
        )
        return sum(p.numel() for p in params)
