"""
Scripted arm trajectory generators for AnticiPose.

Each generator produces (num_envs, 14) arm joint position targets per step and
exposes get_future_plan(t, horizon) -> (num_envs, H, 14) for the wrench predictor.

All trajectories use smooth trapezoidal velocity profiles:
    - Ramp up over t_ramp   (constant-acceleration phase)
    - Hold at peak          (constant-velocity phase)
    - Ramp down over t_ramp (constant-deceleration phase)

All operations are batched on GPU; no Python loops over environments.

Arm-local index mapping (14-dim sub-array of the full 29-DoF state):
    0  left_shoulder_pitch   (global 15)
    1  left_shoulder_roll    (global 16)
    2  left_shoulder_yaw     (global 17)
    3  left_elbow            (global 18)
    4  left_wrist_roll       (global 19)
    5  left_wrist_pitch      (global 20)
    6  left_wrist_yaw        (global 21)
    7  right_shoulder_pitch  (global 22)
    8  right_shoulder_roll   (global 23)
    9  right_shoulder_yaw    (global 24)
    10 right_elbow           (global 25)
    11 right_wrist_roll      (global 26)
    12 right_wrist_pitch     (global 27)
    13 right_wrist_yaw       (global 28)

Default pose: all zeros (arms hanging straight at sides).

Joint limits from VERIFIED_PARAMS.md:
    shoulder_pitch      : [-3.089,  2.670]
    shoulder_roll (L)   : [-1.588,  2.252]
    shoulder_roll (R)   : [-2.252,  1.588]
    shoulder_yaw        : [-2.618,  2.618]
    elbow               : [-1.047,  2.094]
    wrist_roll          : [-1.972,  1.972]
    wrist_pitch         : [-1.614,  1.614]
    wrist_yaw           : [-1.614,  1.614]
"""

import torch

# ---------------------------------------------------------------------------
# Joint limit tensors (arm-local, 14 dims)
# ---------------------------------------------------------------------------

# Lower limits per arm-local index
_JOINT_LOWER = [
    -3.089,  # 0  L shoulder_pitch
    -1.588,  # 1  L shoulder_roll
    -2.618,  # 2  L shoulder_yaw
    -1.047,  # 3  L elbow
    -1.972,  # 4  L wrist_roll
    -1.614,  # 5  L wrist_pitch
    -1.614,  # 6  L wrist_yaw
    -3.089,  # 7  R shoulder_pitch
    -2.252,  # 8  R shoulder_roll  (asymmetric!)
    -2.618,  # 9  R shoulder_yaw
    -1.047,  # 10 R elbow
    -1.972,  # 11 R wrist_roll
    -1.614,  # 12 R wrist_pitch
    -1.614,  # 13 R wrist_yaw
]

# Upper limits per arm-local index
_JOINT_UPPER = [
    2.670,   # 0  L shoulder_pitch
    2.252,   # 1  L shoulder_roll
    2.618,   # 2  L shoulder_yaw
    2.094,   # 3  L elbow
    1.972,   # 4  L wrist_roll
    1.614,   # 5  L wrist_pitch
    1.614,   # 6  L wrist_yaw
    2.670,   # 7  R shoulder_pitch
    1.588,   # 8  R shoulder_roll  (asymmetric!)
    2.618,   # 9  R shoulder_yaw
    2.094,   # 10 R elbow
    1.972,   # 11 R wrist_roll
    1.614,   # 12 R wrist_pitch
    1.614,   # 13 R wrist_yaw
]

# Onset time randomization range (seconds)
_ONSET_MIN = 0.5
_ONSET_MAX = 3.0

# Default control dt used by get_future_plan when not supplied
_DEFAULT_DT = 0.02


def _trapezoid(elapsed, t_ramp, t_hold):
    """
    Normalised scalar trapezoidal profile in [0, 1].

    The profile has three phases:
        [0,        t_ramp]           -> linear ramp from 0 to 1
        [t_ramp,   t_ramp + t_hold]  -> constant 1
        [t_ramp+t_hold, 2*t_ramp+t_hold] -> linear ramp from 1 to 0

    Args:
        elapsed : (num_envs,) time since motion onset
        t_ramp  : (num_envs,) ramp duration
        t_hold  : (num_envs,) hold duration at peak

    Returns:
        profile : (num_envs,) in [0, 1]
    """
    t_total = 2.0 * t_ramp + t_hold

    # Safe division: avoid divide-by-zero when t_ramp == 0
    safe_ramp = t_ramp.clamp(min=1e-6)

    up   = (elapsed / safe_ramp).clamp(0.0, 1.0)
    flat = torch.ones_like(elapsed)
    down = ((t_total - elapsed) / safe_ramp).clamp(0.0, 1.0)

    # Select phase based on elapsed time
    in_ramp_up   = elapsed < t_ramp
    in_hold      = (elapsed >= t_ramp) & (elapsed < t_ramp + t_hold)
    in_ramp_down = elapsed >= t_ramp + t_hold

    profile = (
        in_ramp_up.float()   * up
        + in_hold.float()    * flat
        + in_ramp_down.float() * down
    )
    # Zero out before onset and after full retract
    profile = profile * (elapsed >= 0.0).float()
    profile = profile * (elapsed <= t_total).float()
    return profile


def _clamp_targets(targets, lower, upper):
    """
    Clamp (num_envs, 14) targets to verified joint limits.

    Args:
        targets : (num_envs, 14)
        lower   : (14,) lower limits tensor (same device as targets)
        upper   : (14,) upper limits tensor

    Returns:
        clamped : (num_envs, 14)
    """
    return torch.max(torch.min(targets, upper.unsqueeze(0)), lower.unsqueeze(0))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ArmTrajectoryGenerator:
    """
    Abstract base class for scripted arm trajectory generators.

    Subclasses must implement:
        _randomize(env_ids) — set per-env parameters for the given indices
        _compute_targets(t) — return (num_envs, 14) arm targets for time t

    All parameter tensors are stored at (num_envs,) shape so that
    compute_targets() is a pure batched GPU operation.
    """

    DEFAULT_ARM_POS = [0.0] * 14  # all zeros per VERIFIED_PARAMS

    def __init__(self, num_envs, device, dt=_DEFAULT_DT):
        """
        Args:
            num_envs : number of parallel environments
            device   : torch device string or object (e.g. 'cuda:0')
            dt       : control timestep in seconds (default 0.02 s = 50 Hz)
        """
        self.num_envs = num_envs
        self.device = device
        self.dt = dt

        # Default arm pose (14,) — all zeros, hanging straight
        self.default_arm_pos = torch.tensor(
            self.DEFAULT_ARM_POS, dtype=torch.float32, device=device
        )  # (14,)

        # Joint limit tensors (14,)
        self._joint_lower = torch.tensor(
            _JOINT_LOWER, dtype=torch.float32, device=device
        )
        self._joint_upper = torch.tensor(
            _JOINT_UPPER, dtype=torch.float32, device=device
        )

        # onset_time (num_envs,) — will be set by reset()
        self.onset_time = torch.zeros(num_envs, dtype=torch.float32, device=device)

        # Initialise all parameter tensors to zero-length so subclasses
        # do not need to handle the "first reset" case specially
        self._randomize(torch.arange(num_envs, device=device))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, env_ids):
        """
        Randomise task parameters for the given environment indices.

        Also randomises onset_time in [0.5, 3.0] s so that disturbance
        timing varies across episodes.

        Args:
            env_ids : 1-D LongTensor of environment indices to reset
        """
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        self.onset_time[env_ids] = (
            _ONSET_MIN
            + (_ONSET_MAX - _ONSET_MIN) * torch.rand(n, device=self.device)
        )
        self._randomize(env_ids)

    def compute_targets(self, t):
        """
        Compute arm joint position targets for each environment.

        Args:
            t : (num_envs,) current episode time per environment (seconds)

        Returns:
            targets : (num_envs, 14) arm joint targets, clamped to limits
        """
        targets = self._compute_targets(t)                         # (N, 14)
        return _clamp_targets(targets, self._joint_lower, self._joint_upper)

    def get_future_plan(self, t, horizon=5):
        """
        Compute future arm targets for the wrench predictor look-ahead.

        Args:
            t       : (num_envs,) current episode time per environment
            horizon : number of future steps H (default 5, i.e. 100 ms at 50 Hz)

        Returns:
            plan : (num_envs, H, 14) future arm joint targets
        """
        steps = torch.arange(1, horizon + 1, dtype=torch.float32, device=self.device)
        # future_t : (num_envs, H)
        future_t = t.unsqueeze(1) + steps.unsqueeze(0) * self.dt
        # Vectorise over horizon by repeating envs: (num_envs * H,)
        flat_t = future_t.reshape(-1)
        # Temporarily broadcast all (num_envs,) params to (num_envs * H,)
        flat_plan = self._compute_targets_repeated(flat_t, horizon)  # (N*H, 14)
        # Clamp
        flat_plan = _clamp_targets(flat_plan, self._joint_lower, self._joint_upper)
        # Reshape back: (N, H, 14)
        return flat_plan.reshape(self.num_envs, horizon, 14)

    # ------------------------------------------------------------------
    # Internal helpers — override _randomize and _compute_targets only
    # ------------------------------------------------------------------

    def _randomize(self, env_ids):
        """Randomise per-env parameters for the given indices. Override me."""
        raise NotImplementedError

    def _compute_targets(self, t):
        """
        Compute arm targets.

        Args:
            t : (num_envs,) time per environment
        Returns:
            targets : (num_envs, 14) — NOT yet clamped
        """
        raise NotImplementedError

    def _repeat_params(self, horizon):
        """
        Return a context manager / dict of param tensors repeated H times along
        the batch dimension so that _compute_targets can be called with a
        (num_envs * H,) time vector without changing any internal state.

        This default implementation uses torch.repeat_interleave on every
        registered 1-D parameter tensor.  Subclasses override _gather_params()
        and _scatter_params() instead of this method.
        """
        return horizon

    def _compute_targets_repeated(self, flat_t, horizon):
        """
        Evaluate _compute_targets for a (num_envs * H,) time vector by
        temporarily expanding all (num_envs,) parameter tensors.

        We save the originals, expand, call, restore.  This keeps
        _compute_targets() implementations simple (they always see num_envs
        as the batch size), while get_future_plan() needs only one call.
        """
        N = self.num_envs
        H = horizon
        # Save originals and expand every 1-D tensor attribute of shape (N,)
        saved = {}
        for name, val in self.__dict__.items():
            if isinstance(val, torch.Tensor) and val.shape == (N,):
                saved[name] = val
                # Repeat each env H times: [e0, e0, ..., e1, e1, ..., eN-1, ...]
                setattr(self, name, val.repeat_interleave(H))

        self.num_envs = N * H
        try:
            result = self._compute_targets(flat_t)
        finally:
            # Restore originals
            self.num_envs = N
            for name, val in saved.items():
                setattr(self, name, val)
        return result


# ---------------------------------------------------------------------------
# Task 1: FrontalReachLift
# ---------------------------------------------------------------------------

class FrontalReachLift(ArmTrajectoryGenerator):
    """
    Training Task 1 — Both arms extend forward and upward then hold.

    Motion: shoulder_pitch (indices 0 and 7) drives both arms forward.
    A modest elbow flex (indices 3 and 10) raises the effective CoM.
    Sustained disturbance: tests slow forward CoM shift and vertical load.

    Randomised per episode:
        amplitude   : peak shoulder pitch angle in [0.35, 0.65] rad
        elbow_amp   : peak elbow flex in [0.15, 0.35] rad
        speed       : motion speed in [1.5, 3.0] rad/s
        hold_time   : sustained hold at peak in [1.0, 2.5] s
        onset_time  : motion start time in [0.5, 3.0] s (from base class)
    """

    def _randomize(self, env_ids):
        n = len(env_ids)
        device = self.device

        # Amplitude for shoulder pitch (forward reach)
        if not hasattr(self, 'amplitude'):
            self.amplitude = torch.zeros(self.num_envs, device=device)
        self.amplitude[env_ids] = 0.35 + 0.30 * torch.rand(n, device=device)

        # Elbow flex amplitude (lifts hands slightly)
        if not hasattr(self, 'elbow_amp'):
            self.elbow_amp = torch.zeros(self.num_envs, device=device)
        self.elbow_amp[env_ids] = 0.15 + 0.20 * torch.rand(n, device=device)

        # Angular speed (rad/s)
        if not hasattr(self, 'speed'):
            self.speed = torch.zeros(self.num_envs, device=device)
        self.speed[env_ids] = 1.5 + 1.5 * torch.rand(n, device=device)

        # Hold time at peak (seconds)
        if not hasattr(self, 'hold_time'):
            self.hold_time = torch.zeros(self.num_envs, device=device)
        self.hold_time[env_ids] = 1.0 + 1.5 * torch.rand(n, device=device)

    def _compute_targets(self, t):
        targets = (
            self.default_arm_pos
            .unsqueeze(0)
            .expand(self.num_envs, -1)
            .clone()
        )  # (N, 14)

        elapsed = t - self.onset_time                           # (N,)
        t_ramp  = self.amplitude / self.speed.clamp(min=1e-6)  # (N,) time to peak
        profile = _trapezoid(elapsed, t_ramp, self.hold_time)  # (N,) in [0,1]

        shoulder_delta = profile * self.amplitude   # (N,)
        elbow_delta    = profile * self.elbow_amp   # (N,)

        # Both arms move symmetrically
        targets[:, 0] = targets[:, 0] + shoulder_delta   # L shoulder_pitch
        targets[:, 7] = targets[:, 7] + shoulder_delta   # R shoulder_pitch
        targets[:, 3] = targets[:, 3] + elbow_delta      # L elbow
        targets[:, 10] = targets[:, 10] + elbow_delta    # R elbow

        return targets


# ---------------------------------------------------------------------------
# Task 2: LateralShelfPick
# ---------------------------------------------------------------------------

class LateralShelfPick(ArmTrajectoryGenerator):
    """
    Training Task 2 — One arm extends laterally to simulate a shelf pick.

    Hardest for balance: generates strong asymmetric lateral angular momentum.
    The active arm (left or right) is chosen randomly per episode.

    Arm motion: shoulder_roll (abduction) + small shoulder_pitch (forward lift).

    Note on shoulder_roll sign convention:
        Left  arm: positive roll = abduction (arm moves outward/upward)
        Right arm: negative roll = abduction (outward), BUT the joint limit
                   for R shoulder_roll lower = -2.252.  We use a negative
                   delta so the right arm abducts correctly.

    Randomised per episode:
        roll_amp    : shoulder abduction angle in [0.70, 1.20] rad
        pitch_amp   : forward lift component in [0.10, 0.30] rad
        speed       : motion speed in [2.0, 3.5] rad/s (fastest training task)
        hold_time   : sustained hold at peak in [0.8, 2.0] s
        use_left    : bool — which arm is active
        onset_time  : from base class
    """

    def _randomize(self, env_ids):
        n = len(env_ids)
        device = self.device

        if not hasattr(self, 'roll_amp'):
            self.roll_amp = torch.zeros(self.num_envs, device=device)
        self.roll_amp[env_ids] = 0.70 + 0.50 * torch.rand(n, device=device)

        if not hasattr(self, 'pitch_amp'):
            self.pitch_amp = torch.zeros(self.num_envs, device=device)
        self.pitch_amp[env_ids] = 0.10 + 0.20 * torch.rand(n, device=device)

        if not hasattr(self, 'speed'):
            self.speed = torch.zeros(self.num_envs, device=device)
        self.speed[env_ids] = 2.0 + 1.5 * torch.rand(n, device=device)

        if not hasattr(self, 'hold_time'):
            self.hold_time = torch.zeros(self.num_envs, device=device)
        self.hold_time[env_ids] = 0.8 + 1.2 * torch.rand(n, device=device)

        if not hasattr(self, 'use_left'):
            self.use_left = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.use_left[env_ids] = torch.randint(0, 2, (n,), device=device).bool()

    def _compute_targets(self, t):
        targets = (
            self.default_arm_pos
            .unsqueeze(0)
            .expand(self.num_envs, -1)
            .clone()
        )  # (N, 14)

        elapsed  = t - self.onset_time
        t_ramp   = self.roll_amp / self.speed.clamp(min=1e-6)
        profile  = _trapezoid(elapsed, t_ramp, self.hold_time)  # (N,) in [0,1]

        roll_delta  = profile * self.roll_amp    # (N,)
        pitch_delta = profile * self.pitch_amp   # (N,)

        left_mask  = self.use_left.float()     # (N,)  1 if left, 0 if right
        right_mask = 1.0 - left_mask           # (N,)

        # Left arm abducts with positive roll; right arm with negative roll
        targets[:, 1]  = targets[:, 1]  + left_mask  * roll_delta    # L shoulder_roll +
        targets[:, 8]  = targets[:, 8]  - right_mask * roll_delta    # R shoulder_roll -
        targets[:, 0]  = targets[:, 0]  + left_mask  * pitch_delta   # L shoulder_pitch
        targets[:, 7]  = targets[:, 7]  + right_mask * pitch_delta   # R shoulder_pitch

        return targets


# ---------------------------------------------------------------------------
# Task 3: ForwardPush
# ---------------------------------------------------------------------------

class ForwardPush(ArmTrajectoryGenerator):
    """
    Training Task 3 — Both arms deliver a fast impulsive push forward.

    Directly challenges reactive latency: onset 80-120 ms after trigger.
    The impulse is followed by a short hold then rapid retract.

    Motion: shoulder_pitch (indices 0 and 7), with elbow extension assist
    (elbow delta is positive = flex, countered during retract).

    Randomised per episode:
        push_amp    : peak shoulder pitch angle in [0.35, 0.65] rad
        speed       : motion speed in [2.5, 4.5] rad/s (fastest training task)
        hold_time   : brief hold at peak in [0.08, 0.20] s (impulse window)
        onset_time  : from base class
    """

    def _randomize(self, env_ids):
        n = len(env_ids)
        device = self.device

        if not hasattr(self, 'push_amp'):
            self.push_amp = torch.zeros(self.num_envs, device=device)
        self.push_amp[env_ids] = 0.35 + 0.30 * torch.rand(n, device=device)

        if not hasattr(self, 'speed'):
            self.speed = torch.zeros(self.num_envs, device=device)
        self.speed[env_ids] = 2.5 + 2.0 * torch.rand(n, device=device)

        # Very short hold creates the 80-120ms impulse character
        if not hasattr(self, 'hold_time'):
            self.hold_time = torch.zeros(self.num_envs, device=device)
        self.hold_time[env_ids] = 0.08 + 0.12 * torch.rand(n, device=device)

    def _compute_targets(self, t):
        targets = (
            self.default_arm_pos
            .unsqueeze(0)
            .expand(self.num_envs, -1)
            .clone()
        )  # (N, 14)

        elapsed  = t - self.onset_time
        t_ramp   = self.push_amp / self.speed.clamp(min=1e-6)
        profile  = _trapezoid(elapsed, t_ramp, self.hold_time)  # (N,) in [0,1]

        shoulder_delta = profile * self.push_amp   # (N,)

        targets[:, 0] = targets[:, 0] + shoulder_delta   # L shoulder_pitch
        targets[:, 7] = targets[:, 7] + shoulder_delta   # R shoulder_pitch

        return targets


# ---------------------------------------------------------------------------
# Task 4: LateralSlamDown  (held-out)
# ---------------------------------------------------------------------------

class LateralSlamDown(ArmTrajectoryGenerator):
    """
    Held-out Task 1 — Arm swings payload down + laterally then arrests.

    Tests TEMPORAL generalisation: the wrench is biphasic (positive during
    acceleration, sign-reversed during the arrest deceleration).

    Motion (left arm only — side fixed to maximise transfer difficulty):
        Phase 1 (swing): shoulder_roll abducts + elbow flexes downward.
        Phase 2 (arrest): motion arrests sharply over t_arrest seconds.
        Phase 3 (hold):   arm held at arrested pose for t_hold seconds.

    The arrest creates an angular impulse opposite to the swing, which is
    the key generalisation challenge.

    Randomised per episode:
        roll_amp    : swing shoulder_roll angle in [0.80, 1.30] rad
        elbow_amp   : swing elbow flex in [0.50, 0.90] rad
        speed       : swing speed in [2.0, 3.5] rad/s
        t_arrest    : arrest duration in [0.10, 0.25] s
        hold_time   : final hold at arrested pose in [0.5, 1.5] s
        onset_time  : from base class
    """

    def _randomize(self, env_ids):
        n = len(env_ids)
        device = self.device

        if not hasattr(self, 'roll_amp'):
            self.roll_amp = torch.zeros(self.num_envs, device=device)
        self.roll_amp[env_ids] = 0.80 + 0.50 * torch.rand(n, device=device)

        if not hasattr(self, 'elbow_amp'):
            self.elbow_amp = torch.zeros(self.num_envs, device=device)
        self.elbow_amp[env_ids] = 0.50 + 0.40 * torch.rand(n, device=device)

        if not hasattr(self, 'speed'):
            self.speed = torch.zeros(self.num_envs, device=device)
        self.speed[env_ids] = 2.0 + 1.5 * torch.rand(n, device=device)

        if not hasattr(self, 't_arrest'):
            self.t_arrest = torch.zeros(self.num_envs, device=device)
        self.t_arrest[env_ids] = 0.10 + 0.15 * torch.rand(n, device=device)

        if not hasattr(self, 'hold_time'):
            self.hold_time = torch.zeros(self.num_envs, device=device)
        self.hold_time[env_ids] = 0.5 + 1.0 * torch.rand(n, device=device)

    def _compute_targets(self, t):
        targets = (
            self.default_arm_pos
            .unsqueeze(0)
            .expand(self.num_envs, -1)
            .clone()
        )  # (N, 14)

        elapsed  = t - self.onset_time   # (N,)
        safe_spd = self.speed.clamp(min=1e-6)

        # Phase 1: swing — trapezoidal ramp up to peak, zero hold
        t_swing_ramp = self.roll_amp / safe_spd          # (N,) time to peak
        swing_profile = _trapezoid(
            elapsed, t_swing_ramp, torch.zeros_like(t_swing_ramp)
        )  # peaks at 1 then decrements; we want only the rising edge

        # Clamp to rising edge only (0 -> 1 over t_swing_ramp)
        swing_up = (elapsed / t_swing_ramp.clamp(min=1e-6)).clamp(0.0, 1.0)
        swing_up = swing_up * (elapsed >= 0.0).float()

        # Phase 2: arrest — starts after swing peak, ramps down to arrested fraction
        # We arrest to a fraction (1 - arrest_frac) of peak
        # Arrested fraction of full swing: arm stops at 60% of peak angle
        arrest_frac = 0.40   # retract 40% of peak during arrest
        t_peak = t_swing_ramp                                     # (N,) when peak reached
        elapsed_arrest = (elapsed - t_peak).clamp(min=0.0)       # (N,)
        arrest_ramp = (elapsed_arrest / self.t_arrest.clamp(min=1e-6)).clamp(0.0, 1.0)

        # Combined profile: swing up then partially retract during arrest, then hold
        # Before arrest end: swing_up - arrest_ramp * arrest_frac
        # After arrest: 1 - arrest_frac (held)
        arrested_profile = swing_up - arrest_ramp * arrest_frac   # (N,)
        arrested_profile = arrested_profile.clamp(min=0.0)

        roll_delta  = arrested_profile * self.roll_amp    # (N,)
        elbow_delta = arrested_profile * self.elbow_amp   # (N,)

        # Left arm only (held-out scenario — fixed side)
        targets[:, 1]  = targets[:, 1]  + roll_delta    # L shoulder_roll (abduction)
        targets[:, 3]  = targets[:, 3]  + elbow_delta   # L elbow (flex)

        return targets


# ---------------------------------------------------------------------------
# Task 5: CrossBodyReach  (held-out)
# ---------------------------------------------------------------------------

class CrossBodyReach(ArmTrajectoryGenerator):
    """
    Held-out Task 2 — Right arm reaches across the body toward the left side.

    Tests DIRECTIONAL generalisation: produces strong contralateral yaw torque
    unlike the training tasks.

    Motion (right arm only):
        shoulder_pitch forward  (idx 7)  — brings arm to the front
        shoulder_yaw inward     (idx 9)  — rotates across body (positive = cross)
        shoulder_roll neutral   (idx 8)  — slight adduction to clear torso
        elbow slight flex       (idx 10) — reach extension

    Randomised per episode:
        pitch_amp   : forward pitch angle in [0.60, 1.00] rad
        yaw_amp     : cross-body yaw angle in [0.70, 1.20] rad
        elbow_amp   : elbow flex in [0.20, 0.50] rad
        speed       : motion speed in [1.5, 3.0] rad/s
        hold_time   : sustained hold in [0.8, 2.0] s
        onset_time  : from base class
    """

    def _randomize(self, env_ids):
        n = len(env_ids)
        device = self.device

        if not hasattr(self, 'pitch_amp'):
            self.pitch_amp = torch.zeros(self.num_envs, device=device)
        self.pitch_amp[env_ids] = 0.60 + 0.40 * torch.rand(n, device=device)

        if not hasattr(self, 'yaw_amp'):
            self.yaw_amp = torch.zeros(self.num_envs, device=device)
        self.yaw_amp[env_ids] = 0.70 + 0.50 * torch.rand(n, device=device)

        if not hasattr(self, 'elbow_amp'):
            self.elbow_amp = torch.zeros(self.num_envs, device=device)
        self.elbow_amp[env_ids] = 0.20 + 0.30 * torch.rand(n, device=device)

        if not hasattr(self, 'speed'):
            self.speed = torch.zeros(self.num_envs, device=device)
        self.speed[env_ids] = 1.5 + 1.5 * torch.rand(n, device=device)

        if not hasattr(self, 'hold_time'):
            self.hold_time = torch.zeros(self.num_envs, device=device)
        self.hold_time[env_ids] = 0.8 + 1.2 * torch.rand(n, device=device)

    def _compute_targets(self, t):
        targets = (
            self.default_arm_pos
            .unsqueeze(0)
            .expand(self.num_envs, -1)
            .clone()
        )  # (N, 14)

        elapsed  = t - self.onset_time
        safe_spd = self.speed.clamp(min=1e-6)

        # Use the largest amplitude to compute ramp time (all joints move together)
        dominant_amp = torch.max(self.pitch_amp, self.yaw_amp)
        t_ramp   = dominant_amp / safe_spd
        profile  = _trapezoid(elapsed, t_ramp, self.hold_time)  # (N,) in [0,1]

        # Right arm cross-body reach
        targets[:, 7]  = targets[:, 7]  + profile * self.pitch_amp   # R shoulder_pitch (+fwd)
        targets[:, 9]  = targets[:, 9]  + profile * self.yaw_amp     # R shoulder_yaw (+cross)
        targets[:, 10] = targets[:, 10] + profile * self.elbow_amp   # R elbow (+flex)

        return targets


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY = {
    "frontal_reach_lift":  FrontalReachLift,
    "lateral_shelf_pick":  LateralShelfPick,
    "forward_push":        ForwardPush,
    "lateral_slam_down":   LateralSlamDown,
    "cross_body_reach":    CrossBodyReach,
}

# Training tasks (used by RandomTaskSampler)
TRAINING_TASKS = ["frontal_reach_lift", "lateral_shelf_pick", "forward_push"]

# Held-out tasks (evaluation only)
HELD_OUT_TASKS = ["lateral_slam_down", "cross_body_reach"]


# ---------------------------------------------------------------------------
# RandomTaskSampler
# ---------------------------------------------------------------------------

class RandomTaskSampler:
    """
    Assigns one of the training tasks (1-3) to each environment at random on
    reset and proxies compute_targets / get_future_plan / reset through to the
    appropriate per-task generator.

    Each environment is always controlled by exactly one task generator.
    On reset() a new task is drawn uniformly from the training set.

    Usage:
        sampler = RandomTaskSampler(num_envs, device, dt=0.02)
        sampler.reset(env_ids)                          # assign tasks, randomise params
        targets = sampler.compute_targets(t)            # (num_envs, 14)
        plan    = sampler.get_future_plan(t, horizon=5) # (num_envs, 5, 14)

    Internal design:
        One full-size generator per training task is kept alive for the full
        training run.  On reset, only the affected env_ids update their task
        assignment and parameters.  compute_targets() calls each generator for
        ALL envs and then gathers via per-env assignment masks — fully batched.
    """

    def __init__(self, num_envs, device, dt=_DEFAULT_DT):
        self.num_envs = num_envs
        self.device = device
        self.dt = dt

        # Instantiate one full-size generator per training task
        self.generators = {
            name: TASK_REGISTRY[name](num_envs, device, dt)
            for name in TRAINING_TASKS
        }
        self._task_names = TRAINING_TASKS  # stable order for indexing

        # task_ids[i] in {0, 1, 2} — which task env i is currently running
        self.task_ids = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Initialise all envs with random task assignment
        self.reset(torch.arange(num_envs, device=device))

    def reset(self, env_ids):
        """
        Randomly assign a training task and randomise parameters for each env.

        Args:
            env_ids : 1-D LongTensor of environment indices
        """
        if len(env_ids) == 0:
            return

        n = len(env_ids)

        # Sample new task assignments uniformly
        new_task_ids = torch.randint(
            0, len(self._task_names), (n,), device=self.device
        )  # (n,) in {0, 1, 2}
        self.task_ids[env_ids] = new_task_ids

        # Call reset on each generator for the env_ids assigned to it
        for task_idx, name in enumerate(self._task_names):
            mask = new_task_ids == task_idx          # (n,) bool
            selected = env_ids[mask]                 # env indices assigned to this task
            if len(selected) > 0:
                self.generators[name].reset(selected)

    def compute_targets(self, t):
        """
        Compute arm targets for all environments.

        Each env's target comes from its currently assigned generator.

        Args:
            t : (num_envs,) current episode time per environment

        Returns:
            targets : (num_envs, 14) arm joint targets
        """
        # Compute targets from all generators — (num_envs, 14) each
        all_targets = torch.stack(
            [self.generators[name].compute_targets(t) for name in self._task_names],
            dim=0
        )  # (num_tasks, num_envs, 14)

        # Gather per-env from the assigned generator
        # task_ids : (num_envs,) -> index into dim 0 of all_targets
        idx = self.task_ids.view(1, self.num_envs, 1).expand(1, self.num_envs, 14)
        targets = all_targets.gather(0, idx).squeeze(0)  # (num_envs, 14)
        return targets

    def get_future_plan(self, t, horizon=5):
        """
        Compute future arm plan for all environments.

        Args:
            t       : (num_envs,) current episode time per environment
            horizon : H lookahead steps

        Returns:
            plan : (num_envs, H, 14)
        """
        # Plans from all generators — (num_tasks, num_envs, H, 14)
        all_plans = torch.stack(
            [self.generators[name].get_future_plan(t, horizon) for name in self._task_names],
            dim=0
        )  # (num_tasks, num_envs, H, 14)

        # Gather per-env: task_ids (num_envs,) -> index dim 0
        idx = (
            self.task_ids
            .view(1, self.num_envs, 1, 1)
            .expand(1, self.num_envs, horizon, 14)
        )
        plan = all_plans.gather(0, idx).squeeze(0)  # (num_envs, H, 14)
        return plan

    @property
    def active_task_names(self):
        """Return list of task name per environment (for logging/debugging)."""
        return [self._task_names[i.item()] for i in self.task_ids]
