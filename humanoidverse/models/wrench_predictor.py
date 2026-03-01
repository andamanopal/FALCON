"""
wrench_predictor.py
-------------------
Frozen MLP that predicts the H=5 step wrench sequence from the current
proprioceptive observation and the arm motion plan.

Architecture
~~~~~~~~~~~~
Base input : obs (115) concatenated with arm plan (70) -> 185 dims
Enhanced   : + plan velocity (56) + plan acceleration (42) -> 283 dims
             (enabled via use_plan_derivatives=True)

Hidden : Linear(input_dim->256) + LayerNorm(256) + ELU
         Linear(256->256) + LayerNorm(256) + ELU
         Linear(256->128) + LayerNorm(128) + ELU
Output : Linear(128->30)   (H=5 x 6-dim wrench: force_xyz + torque_xyz)

Parameter count (base 185D): ~152K
Parameter count (enhanced 283D): ~177K (+25K from wider first layer)

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

# Plan derivative dimensions (finite-difference velocity & acceleration)
ARM_JOINTS   = 14
HORIZON      = PLAN_DIM // ARM_JOINTS       # 5
PLAN_VEL_DIM = (HORIZON - 1) * ARM_JOINTS   # 4 * 14 = 56
PLAN_ACC_DIM = (HORIZON - 2) * ARM_JOINTS   # 3 * 14 = 42
ENHANCED_INPUT_DIM = INPUT_DIM + PLAN_VEL_DIM + PLAN_ACC_DIM  # 283

# Enhanced obs with body-side context (gait phase, foot contacts, base_lin_vel, payload)
ENHANCED_OBS_DIM = 123       # 115 + 8 body-side signals
ENHANCED_INPUT_DIM_V2 = ENHANCED_OBS_DIM + PLAN_DIM  # 193

# Known input dimensions for auto-detection from checkpoints
_KNOWN_INPUT_DIMS = {
    185: {"obs_dim": OBS_DIM, "use_derivatives": False},        # base
    193: {"obs_dim": ENHANCED_OBS_DIM, "use_derivatives": False},  # body-side context
    283: {"obs_dim": OBS_DIM, "use_derivatives": True},          # base + derivatives
    291: {"obs_dim": ENHANCED_OBS_DIM, "use_derivatives": True},   # body-side + derivatives
}


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

    def __init__(self, input_dim=None, output_dim=OUTPUT_DIM, plan_dim=PLAN_DIM,
                 use_plan_derivatives=False, obs_dim=None):
        super().__init__()

        self._use_plan_derivatives = use_plan_derivatives

        # Resolve obs_dim: default to OBS_DIM (115) if not specified
        if obs_dim is None:
            obs_dim = OBS_DIM
        self._obs_dim = obs_dim

        # Resolve input_dim from obs_dim + plan + optional derivatives
        if input_dim is None:
            base_input = obs_dim + plan_dim
            if use_plan_derivatives:
                input_dim = base_input + PLAN_VEL_DIM + PLAN_ACC_DIM
            else:
                input_dim = base_input
        self._input_dim = input_dim
        self._output_dim = output_dim

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

        # Plan derivative normalisation buffers (only when enabled)
        if use_plan_derivatives:
            self.register_buffer("plan_vel_mean", torch.zeros(PLAN_VEL_DIM))
            self.register_buffer("plan_vel_std",  torch.ones(PLAN_VEL_DIM))
            self.register_buffer("plan_acc_mean", torch.zeros(PLAN_ACC_DIM))
            self.register_buffer("plan_acc_std",  torch.ones(PLAN_ACC_DIM))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
        """Predict the H=5 step wrench sequence.

        The method applies z-score normalisation to the inputs and
        inverse-normalisation to the outputs so callers work with raw
        (unscaled) quantities.

        When ``use_plan_derivatives=True``, finite-difference velocity and
        acceleration of the arm plan are computed on-the-fly and concatenated
        to the input, providing the network with the kinematic derivatives
        that directly relate to wrench via Newton-Euler (F=ma, tau=I*alpha).

        Args:
            obs  (torch.Tensor): Shape (B, 115).  Raw per-step actor obs.
            plan (torch.Tensor): Shape (B,  70).  Arm plan (H=5 x 14 joints).

        Returns:
            wrench (torch.Tensor): Shape (B, 30).  Predicted wrenches in the
                same physical units as the training targets (N / Nm).
        """
        obs_n  = self._normalize(obs,  self.obs_mean,  self.obs_std)
        plan_n = self._normalize(plan, self.plan_mean, self.plan_std)

        if self._use_plan_derivatives:
            plan_vel, plan_acc = self._compute_plan_derivatives(plan)
            plan_vel_n = self._normalize(plan_vel, self.plan_vel_mean, self.plan_vel_std)
            plan_acc_n = self._normalize(plan_acc, self.plan_acc_mean, self.plan_acc_std)
            x = torch.cat([obs_n, plan_n, plan_vel_n, plan_acc_n], dim=-1)  # (B, 283)
        else:
            x = torch.cat([obs_n, plan_n], dim=-1)  # (B, 185)

        out_n = self.net(x)  # (B, 30)  normalised output

        # Inverse-normalise to recover physical-unit predictions.
        wrench = out_n * self.wrench_std + self.wrench_mean
        return wrench

    @staticmethod
    def _compute_plan_derivatives(plan: torch.Tensor):
        """Compute finite-difference velocity and acceleration from arm plan.

        The plan tensor contains H=5 steps of 14 joint positions.  Velocity
        is the first difference (4 steps) and acceleration is the second
        difference (3 steps).  No dt division is needed because z-score
        normalisation absorbs the scale factor.

        Args:
            plan (torch.Tensor): Shape (B, 70).

        Returns:
            plan_vel (torch.Tensor): Shape (B, 56) — 4 steps x 14 joints.
            plan_acc (torch.Tensor): Shape (B, 42) — 3 steps x 14 joints.
        """
        B = plan.shape[0]
        steps = plan.view(B, HORIZON, ARM_JOINTS)                 # (B, 5, 14)
        vel = (steps[:, 1:] - steps[:, :-1])                      # (B, 4, 14)
        acc = (vel[:, 1:] - vel[:, :-1])                           # (B, 3, 14)
        return vel.reshape(B, -1), acc.reshape(B, -1)

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
        plan_vel_mean: torch.Tensor = None,
        plan_vel_std:  torch.Tensor = None,
        plan_acc_mean: torch.Tensor = None,
        plan_acc_std:  torch.Tensor = None,
    ) -> None:
        """Set z-score statistics from training data.

        Call this once after computing dataset statistics, before saving the
        checkpoint.  All tensors are moved to the buffer's current device.
        """
        device = self.obs_mean.device
        self.obs_mean.copy_(obs_mean.to(device))
        self.obs_std.copy_(obs_std.to(device))
        self.plan_mean.copy_(plan_mean.to(device))
        self.plan_std.copy_(plan_std.to(device))
        self.wrench_mean.copy_(wrench_mean.to(device))
        self.wrench_std.copy_(wrench_std.to(device))

        if self._use_plan_derivatives and plan_vel_mean is not None:
            self.plan_vel_mean.copy_(plan_vel_mean.to(device))
            self.plan_vel_std.copy_(plan_vel_std.to(device))
            self.plan_acc_mean.copy_(plan_acc_mean.to(device))
            self.plan_acc_std.copy_(plan_acc_std.to(device))

    # ------------------------------------------------------------------
    # Frozen-model loading
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "WrenchPredictor":
        """Auto-detect configuration from checkpoint and load frozen model.

        Inspects the state dict to determine whether the checkpoint was
        trained with plan derivatives (presence of ``plan_vel_mean`` buffer)
        and constructs the model with matching architecture.

        Args:
            path:   Path to checkpoint file.
            device: Target device.

        Returns:
            A frozen WrenchPredictor on the requested device.
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
                "Expected a dict with 'model_state_dict' or 'state_dict'."
            )

        use_derivatives = "plan_vel_mean" in state_dict

        # Auto-detect obs_dim: prefer obs_mean buffer (most reliable),
        # fall back to first-layer input size for legacy checkpoints.
        if "obs_mean" in state_dict:
            obs_dim = state_dict["obs_mean"].shape[0]
        else:
            first_layer_in = state_dict["net.0.weight"].shape[1]
            detected = _KNOWN_INPUT_DIMS.get(first_layer_in)
            if detected is not None:
                obs_dim = detected["obs_dim"]
            else:
                deriv_extra = (
                    PLAN_VEL_DIM + PLAN_ACC_DIM if use_derivatives else 0
                )
                obs_dim = first_layer_in - PLAN_DIM - deriv_extra

        model = cls(
            use_plan_derivatives=use_derivatives,
            obs_dim=obs_dim,
        ).to(device)
        model.load_state_dict(state_dict, strict=True)
        model._freeze()
        return model

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
