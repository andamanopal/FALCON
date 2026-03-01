"""AnticiPose Environment: Anticipatory whole-body control under arm manipulation.

Uses FALCON's YAML-driven observation dispatch so that:
  - pre_process_config() auto-sums obs dims -> PPO gets correct input size
  - History shifting includes extra components in each frame
  - Adding/removing obs components is just a YAML change

Observation getters (FALCON dispatches _get_obs_<name>() automatically):
  _get_obs_predicted_wrench  -> (N, H*6=30)  Predicted wrenches (B5)
  _get_obs_arm_plan          -> (N, H*14=70) Raw arm plan (B4a/B4b)
  _get_obs_current_wrench    -> (N, 6)       Current wrench (B3/critic)
  _get_obs_cvae_latent       -> (N, 30)      CVAE latent (B6)

Baselines (ablation ladder):
  B1   Reactive          — unmodified FALCON (no extra obs)
  B2   Extended History  — 10-step obs history (vs 5)
  B3   Current Wrench    — 6-dim current wrench in actor obs
  B4a  Direct Plan (Actor)  — 70-dim arm plan in actor obs
  B4b  Direct Plan (Critic) — 70-dim arm plan in critic only
  B5   AnticiPose        — 30-dim predicted future wrench (OUR METHOD)
  B6   CVAE Latent       — 30-dim CVAE latent encoding of arm plan

Hierarchy:
  BaseTask -> LeggedRobotBase -> LeggedRobotLocomotion
    -> LeggedRobotDecoupledLocomotionStance
    -> LeggedRobotDecoupledLocomotionStanceHeightWBC
    -> LeggedRobotDecoupledLocomotionStanceHeightWBCForce
    -> AnticiPoseEnv  (this class)
"""

from __future__ import annotations

from humanoidverse.envs.decoupled_locomotion.decoupled_locomotion_stand_height_waist_wbc_ma_diff_force import (
    LeggedRobotDecoupledLocomotionStanceHeightWBCForce,
)
from humanoidverse.envs.arm_trajectory_generators import RandomTaskSampler, TASK_REGISTRY
from humanoidverse.utils.analytical_wrench import AnalyticalWrench

import torch
from typing import Optional
from loguru import logger

# ---------------------------------------------------------------------------
# Constants (verified against VERIFIED_PARAMS.md)
# ---------------------------------------------------------------------------
_WRENCH_DIM = 6           # Fx, Fy, Fz, Tx, Ty, Tz
_ARM_JOINTS = 14           # 7 left + 7 right
_ARM_DOF_START = 15        # First arm DOF index in 29-DOF config
_ARM_DOF_END = 29          # Past-the-end arm DOF index

# Rigid-body names for analytical wrench computation.
# Verified from g1_29dof_waist_fakehand.yaml (32 bodies total).
_LEFT_ARM_BODY_NAMES = [
    "left_shoulder_pitch_link", "left_shoulder_roll_link",
    "left_shoulder_yaw_link", "left_elbow_link",
    "left_wrist_roll_link", "left_wrist_pitch_link",
    "left_wrist_yaw_link", "left_rubber_hand",
]
_RIGHT_ARM_BODY_NAMES = [
    "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link",
    "right_wrist_roll_link", "right_wrist_pitch_link",
    "right_wrist_yaw_link", "right_rubber_hand",
]


class AnticiPoseEnv(LeggedRobotDecoupledLocomotionStanceHeightWBCForce):
    """AnticiPose locomotion environment.

    Adds scripted arm trajectories, payload simulation, and wrench-aware
    observations on top of FALCON's decoupled WBC force environment.
    All AnticiPose-specific logic is confined to hook overrides --- no parent
    method is copy-pasted.

    Config keys under ``config.env.config``:
        anticipose_mode (str): "reactive" | "direct_plan" | "anticipose" | "cvae"
        anticipose_horizon (int): H, default 5.
        arm_trajectory_task (str): Task type for trajectory generator.
        collect_wrench_data (bool): Whether to collect supervision data.
        collect_buffer_size (int): Ring buffer capacity for data collection.
        wrench_predictor_ckpt (str | None): Path to frozen predictor checkpoint.
        cvae_ckpt (str | None): Path to frozen CVAE encoder checkpoint.
        max_payload_mass (float): Max payload mass in kg (randomized per env).
    """

    def __init__(self, config, device):
        self.init_done = False

        # Parse AnticiPose config BEFORE super().__init__() triggers
        # _init_buffers, so buffer dimensions are known ahead of time.
        # Note: `config` IS the env config node (config.env.config in Hydra terms).
        # Hydra's instantiate(config=config.env) passes the `config` sub-key.
        self._ap_mode: str = getattr(config, "anticipose_mode", "reactive")
        self._ap_horizon: int = getattr(config, "anticipose_horizon", 5)
        self._ap_collect: bool = getattr(config, "collect_wrench_data", False)
        self._max_payload_mass: float = getattr(
            config, "max_payload_mass", 0.0,
        )

        assert self._ap_mode in {
            "reactive", "direct_plan", "anticipose", "cvae"
        }, (
            f"Unknown anticipose_mode: {self._ap_mode!r}. "
            "Choose from: reactive, direct_plan, anticipose, cvae."
        )

        logger.info(
            f"[AnticiPoseEnv] mode={self._ap_mode!r}, "
            f"horizon={self._ap_horizon}"
        )

        super().__init__(config, device)

        # ---- Arm trajectory generator (scripted, not learned) ----
        task_type = getattr(self.config, "arm_trajectory_task", "random")
        if task_type == "random":
            self._arm_traj_gen = RandomTaskSampler(
                self.num_envs, self.device, dt=self.dt,
            )
        else:
            self._arm_traj_gen = TASK_REGISTRY[task_type](
                self.num_envs, self.device, dt=self.dt,
            )

        # ---- AnalyticalWrench (Newton-Euler rigid body dynamics) ----
        body_names_list = self._resolve_body_names()
        left_ids = [body_names_list.index(n) for n in _LEFT_ARM_BODY_NAMES]
        right_ids = [body_names_list.index(n) for n in _RIGHT_ARM_BODY_NAMES]
        self._analytical_wrench = AnalyticalWrench(
            env=self,
            left_arm_body_indices=left_ids,
            right_arm_body_indices=right_ids,
            dt=self.dt,
        )

        # ---- Frozen wrench predictor (B5 anticipose only) ----
        self._wrench_predictor: Optional[torch.nn.Module] = None
        predictor_ckpt = getattr(self.config, "wrench_predictor_ckpt", None)
        if self._ap_mode == "anticipose" and predictor_ckpt is not None:
            self._wrench_predictor = self._load_wrench_predictor(
                predictor_ckpt,
            )
            logger.info(
                f"[AnticiPoseEnv] Loaded wrench predictor "
                f"from {predictor_ckpt!r}"
            )
        elif self._ap_mode == "anticipose":
            logger.warning(
                "[AnticiPoseEnv] anticipose mode but no "
                "wrench_predictor_ckpt. Predicted wrenches will be zeros."
            )

        # ---- Frozen CVAE encoder (B6 cvae only) ----
        self._cvae_encoder: Optional[torch.nn.Module] = None
        cvae_ckpt = getattr(self.config, "cvae_ckpt", None)
        if self._ap_mode == "cvae" and cvae_ckpt is not None:
            self._cvae_encoder = self._load_cvae(cvae_ckpt)
            logger.info(
                f"[AnticiPoseEnv] Loaded CVAE encoder "
                f"from {cvae_ckpt!r}"
            )
        elif self._ap_mode == "cvae":
            logger.warning(
                "[AnticiPoseEnv] cvae mode but no "
                "cvae_ckpt. CVAE latent will be zeros."
            )

        # ---- Wrench data collector (B1 data-gathering stage) ----
        self._wrench_collector = None
        if self._ap_collect:
            from humanoidverse.utils.wrench_data_collector import (
                WrenchDataCollector,
            )
            capacity = getattr(self.config, "collect_buffer_size", 500_000)
            self._wrench_collector = WrenchDataCollector(
                num_envs=self.num_envs,
                horizon=self._ap_horizon,
                capacity=capacity,
                device=device,
            )
            logger.info(
                f"[AnticiPoseEnv] Wrench data collection enabled "
                f"(H={self._ap_horizon}, {capacity:,} pair capacity)"
            )

        self.init_done = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_body_names(self):
        """Resolve the list of rigid body names from the simulator."""
        if hasattr(self, "body_names"):
            return self.body_names
        if hasattr(self.simulator, "body_names"):
            return self.simulator.body_names
        raise AttributeError(
            "Cannot find body_names on env or simulator. "
            "Ensure the robot URDF is loaded before AnticiPoseEnv init."
        )

    def _resolve_num_bodies(self) -> int:
        """Get the total number of rigid bodies from the simulator."""
        if hasattr(self.simulator, "_rigid_body_pos"):
            return self.simulator._rigid_body_pos.shape[1]
        return len(self._resolve_body_names())

    def _resolve_ee_body_indices(self) -> torch.Tensor:
        """Get rigid body indices for left_rubber_hand and right_rubber_hand."""
        body_names = self._resolve_body_names()
        ee_names = ["left_rubber_hand", "right_rubber_hand"]
        indices = [body_names.index(n) for n in ee_names]
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Buffer initialisation
    # ------------------------------------------------------------------

    def _init_buffers(self):
        """Allocate AnticiPose-specific buffers after parent init."""
        super()._init_buffers()

        H = getattr(self, "_ap_horizon", 5)
        n = self.num_envs

        # Current-step 6D wrench (force + torque at base from arm motion)
        self._current_wrench = torch.zeros(
            n, _WRENCH_DIM, dtype=torch.float32, device=self.device,
        )
        # Predicted wrench buffer: H future predicted wrenches (N, H*6)
        self._predicted_wrench_buf = torch.zeros(
            n, H * _WRENCH_DIM, dtype=torch.float32, device=self.device,
        )
        # Arm plan buffer: H future arm joint targets flattened (N, H*14)
        self._arm_plan_buf = torch.zeros(
            n, H * _ARM_JOINTS, dtype=torch.float32, device=self.device,
        )
        # CVAE latent buffer: 30-dim (same as predicted wrench for parity)
        self._cvae_latent_buf = torch.zeros(
            n, H * _WRENCH_DIM, dtype=torch.float32, device=self.device,
        )
        # Current-step arm targets: (N, 14)
        self._arm_targets = torch.zeros(
            n, _ARM_JOINTS, dtype=torch.float32, device=self.device,
        )
        # Payload mass per env (randomized at reset): (N,)
        self._payload_mass = torch.zeros(
            n, dtype=torch.float32, device=self.device,
        )
        # Cached EE body indices for payload force application
        self._ee_body_indices = self._resolve_ee_body_indices()
        # Payload force tensor for apply_rigid_body_force_at_pos_tensor: (N, num_bodies, 3)
        num_bodies = self._resolve_num_bodies()
        self._payload_force = torch.zeros(
            n, num_bodies, 3, dtype=torch.float32, device=self.device,
        )
        # Distribution shift monitoring (B5 only): throttle to every N steps
        self._pred_monitor_interval = 50
        self._pred_monitor_counter = 0
        # Buffer previous step's t+1 prediction for temporally-aligned comparison
        self._prev_predicted_next_wrench = torch.zeros(
            n, _WRENCH_DIM, dtype=torch.float32, device=self.device,
        )

    # ------------------------------------------------------------------
    # Arm trajectory advancement
    # ------------------------------------------------------------------

    def _advance_arm_trajectory(self):
        """Compute arm targets and future plan for the current timestep.

        Sets:
          self._arm_targets   (N, 14) --- joint targets for this control step
          self._arm_plan_buf  (N, 70) --- flattened [t+1 ... t+H] targets
        """
        t = self.episode_length_buf.float() * self.dt  # steps -> seconds
        self._arm_targets = self._arm_traj_gen.compute_targets(t)
        self._arm_plan_buf[:] = self._arm_traj_gen.get_future_plan(
            t, self._ap_horizon,
        ).reshape(self.num_envs, -1)

    # ------------------------------------------------------------------
    # Pre-physics: inject scripted arm targets into actions
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions):
        """Override to inject scripted arm targets and payload forces.

        The PD controller computes:
          torque = Kp * (action * scale + default_dof_pos - dof_pos) - Kd * dof_vel

        Setting action = (desired - default_dof_pos) / scale makes the
        controller drive the arm joints to the desired trajectory positions.

        Payload: Apply gravitational force at EE rigid bodies to simulate
        carrying an object of mass self._payload_mass[env_i].
        """
        self._advance_arm_trajectory()

        scale = self.config.robot.control.action_scale
        modified = actions.clone()
        modified[:, _ARM_DOF_START:_ARM_DOF_END] = (
            self._arm_targets
            - self.default_dof_pos[:, _ARM_DOF_START:_ARM_DOF_END]
        ) / scale

        super()._pre_physics_step(modified)

        # Apply payload gravitational force at EE bodies
        if self._max_payload_mass > 0.0:
            self._apply_payload_forces()

    # ------------------------------------------------------------------
    # Pre-observation callback: compute wrenches before obs assembly
    # ------------------------------------------------------------------

    def _pre_compute_observations_callback(self):
        """Override to compute wrenches after sim refresh, before obs assembly.

        Called by parent's _post_physics_step() after _refresh_sim_tensors()
        and before _compute_observations().  At this point all rigid body
        states are fresh from the simulator.
        """
        super()._pre_compute_observations_callback()

        # For B1 (reactive), do NOT zero EE forces — leave FALCON's random
        # external force perturbations intact so B1 is truly unmodified.
        # For all other modes, zero them so disturbance comes from arm
        # motion dynamics only.
        if self._ap_mode != "reactive":
            if hasattr(self, "left_ee_apply_force"):
                self.left_ee_apply_force.zero_()
            if hasattr(self, "right_ee_apply_force"):
                self.right_ee_apply_force.zero_()

        # Compute ground-truth wrench from arm rigid body dynamics
        self._current_wrench[:] = self._analytical_wrench.compute(
            payload_mass=self._payload_mass,
        )

        # Run frozen predictor if loaded
        if self._wrench_predictor is not None:
            self._run_wrench_predictor()
            self._monitor_prediction_shift()

        # Run frozen CVAE encoder if loaded
        if self._cvae_encoder is not None:
            self._run_cvae_encoder()

    # ------------------------------------------------------------------
    # Post-observation callback: data collection
    # ------------------------------------------------------------------

    def _post_compute_observations_callback(self):
        """Override to collect wrench supervision data after obs assembly.

        Uses temporal-offset collection: at each step we push
        (obs_t, plan_t, wrench_t) into per-env FIFO queues.  When a queue
        has H+1 entries, the collector yields training pairs:
          (obs_{t-H}, plan_{t-H}) -> [wrench_{t-H+1}, ..., wrench_t]
        """
        super()._post_compute_observations_callback()

        if self._wrench_collector is not None:
            obs_step = self._build_predictor_obs()
            self._wrench_collector.push_step(
                obs_step, self._arm_plan_buf, self._current_wrench,
            )

    # ------------------------------------------------------------------
    # Observation getters (FALCON dispatches _get_obs_<name>() via YAML)
    # ------------------------------------------------------------------

    def _get_obs_predicted_wrench(self) -> torch.Tensor:
        """Predicted future wrenches. Shape: (N, H*6=30). Used by B5."""
        return self._predicted_wrench_buf

    def _get_obs_arm_plan(self) -> torch.Tensor:
        """Raw future arm joint plan. Shape: (N, H*14=70). Used by B4a."""
        return self._arm_plan_buf

    def _get_obs_current_wrench(self) -> torch.Tensor:
        """Current-step base wrench. Shape: (N, 6). Critic-only obs."""
        return self._current_wrench

    def _get_obs_cvae_latent(self) -> torch.Tensor:
        """CVAE latent encoding of arm plan. Shape: (N, 30). Used by B6."""
        return self._cvae_latent_buf

    # ------------------------------------------------------------------
    # Wrench predictor inference
    # ------------------------------------------------------------------

    def _run_wrench_predictor(self):
        """Run the frozen predictor to fill _predicted_wrench_buf.

        Input: obs (115) + arm_plan (70) = 185 dims.
        Output: predicted future wrenches (H*6 = 30 dims).
        """
        obs_step = self._build_predictor_obs()
        with torch.no_grad():
            self._predicted_wrench_buf[:] = self._wrench_predictor(
                obs_step, self._arm_plan_buf,
            )

    def _monitor_prediction_shift(self):
        """Log wrench prediction RMSE to detect distribution shift.

        Uses temporally-aligned comparison: the prediction for t+1 made
        at step t-1 is compared against the ground-truth wrench at step t.
        _prev_predicted_next_wrench stores the step-ahead prediction from
        the previous call, so the comparison is properly aligned.

        Throttled to every _pred_monitor_interval steps to avoid overhead.
        """
        # Always save current t+1 prediction for next step's comparison
        current_pred_next = self._predicted_wrench_buf[:, 0:6].clone()

        self._pred_monitor_counter += 1
        if self._pred_monitor_counter < self._pred_monitor_interval:
            self._prev_predicted_next_wrench[:] = current_pred_next
            return
        self._pred_monitor_counter = 0

        # Compare PREVIOUS step's t+1 prediction against current wrench
        pred = self._prev_predicted_next_wrench
        target = self._current_wrench

        residuals = pred - target
        rmse_overall = (residuals ** 2).mean().sqrt()

        # Force (first 3 dims) vs torque (last 3 dims)
        rmse_force = (residuals[:, :3] ** 2).mean().sqrt()
        rmse_torque = (residuals[:, 3:] ** 2).mean().sqrt()

        if hasattr(self, "log_dict"):
            self.log_dict["Env/pred_wrench_rmse_overall"] = rmse_overall
            self.log_dict["Env/pred_wrench_rmse_force"] = rmse_force
            self.log_dict["Env/pred_wrench_rmse_torque"] = rmse_torque

        # Update buffer for next comparison
        self._prev_predicted_next_wrench[:] = current_pred_next

    def _build_predictor_obs(self) -> torch.Tensor:
        """Assemble 115-dim predictor input from current env state.

        Components match the YAML obs ordering:
          base_ang_vel(3), projected_gravity(3), command_lin_vel(2),
          command_ang_vel(1), command_stand(1), command_waist_dofs(3),
          command_base_height(1), ref_upper_dof_pos(14), dof_pos(29),
          dof_vel(29), actions(29) = 115
        """
        return torch.cat([
            self.base_ang_vel,                              # 3
            self.projected_gravity,                         # 3
            self.commands[:, 0:2],                          # 2  lin vel
            self.commands[:, 2:3],                          # 1  ang vel
            self.commands[:, 4:5],                          # 1  stand
            self.commands[:, 5:8],                          # 3  waist dofs
            self.commands[:, 8:9],                          # 1  base height
            self.ref_upper_dof_pos,                         # 14
            self.simulator.dof_pos - self.default_dof_pos,  # 29
            self.simulator.dof_vel,                         # 29
            self.actions,                                   # 29
        ], dim=-1)                                          # = 115

    # ------------------------------------------------------------------
    # CVAE encoder inference
    # ------------------------------------------------------------------

    def _run_cvae_encoder(self):
        """Run the frozen CVAE encoder to fill _cvae_latent_buf.

        Input: obs (115) + arm_plan (70) = 185 dims.
        Output: deterministic latent mu (30 dims).
        """
        obs_step = self._build_predictor_obs()
        with torch.no_grad():
            self._cvae_latent_buf[:] = self._cvae_encoder(
                obs_step, self._arm_plan_buf,
            )

    def _load_cvae(self, ckpt_path: str) -> torch.nn.Module:
        """Load and freeze a CVAE encoder from checkpoint.

        Tries TorchScript first (fastest inference), then falls back to
        state-dict loading with the standard ArmPlanCVAE architecture.

        Args:
            ckpt_path: Path to saved model (.pt).

        Returns:
            A frozen torch.nn.Module on self.device.
        """
        try:
            encoder = torch.jit.load(ckpt_path, map_location=self.device)
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad_(False)
            logger.info("[AnticiPoseEnv] Loaded TorchScript CVAE encoder.")
            return encoder
        except (RuntimeError, ValueError):
            pass

        from humanoidverse.models.arm_plan_cvae import ArmPlanCVAE
        encoder = ArmPlanCVAE().to(self.device)
        encoder.load_frozen(ckpt_path)
        logger.info("[AnticiPoseEnv] Loaded state-dict CVAE encoder.")
        return encoder

    # ------------------------------------------------------------------
    # Reset handling
    # ------------------------------------------------------------------

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        """Extend parent reset to clear AnticiPose-specific buffers."""
        if len(env_ids) == 0:
            return

        self._arm_traj_gen.reset(env_ids)
        self._analytical_wrench.reset(env_ids)
        self._current_wrench[env_ids] = 0.0
        self._predicted_wrench_buf[env_ids] = 0.0
        self._prev_predicted_next_wrench[env_ids] = 0.0
        self._cvae_latent_buf[env_ids] = 0.0
        self._arm_plan_buf[env_ids] = 0.0
        self._arm_targets[env_ids] = 0.0

        # Randomize payload mass per env: uniform [0, max_payload_mass]
        if self._max_payload_mass > 0.0:
            self._payload_mass[env_ids] = (
                torch.rand(len(env_ids), device=self.device)
                * self._max_payload_mass
            )
        else:
            self._payload_mass[env_ids] = 0.0

        # Flush wrench data collector queues on reset to avoid
        # cross-episode temporal pairs.
        if self._wrench_collector is not None:
            self._wrench_collector.flush_envs(env_ids)

        super().reset_envs_idx(env_ids, target_states, target_buf)

    # ------------------------------------------------------------------
    # Data persistence
    # ------------------------------------------------------------------

    def save_collected_wrench_data(self, path: str) -> None:
        """Save collected wrench supervision data to disk.

        Args:
            path: File path for the .pt output.
        """
        if self._wrench_collector is not None:
            self._wrench_collector.save(path)
            logger.info(
                f"[AnticiPoseEnv] Wrench data saved to {path} "
                f"({len(self._wrench_collector):,} samples)"
            )

    def get_collected_data(self):
        """Return collected (obs, plan, wrench) dataset dict, or None."""
        if self._wrench_collector is not None:
            return self._wrench_collector.get_dataset()
        return None

    # ------------------------------------------------------------------
    # Predictor loading
    # ------------------------------------------------------------------

    def _load_wrench_predictor(self, ckpt_path: str) -> torch.nn.Module:
        """Load and freeze a wrench predictor from checkpoint.

        Tries TorchScript first (fastest inference), then falls back to
        state-dict loading with the standard WrenchPredictor architecture.

        Args:
            ckpt_path: Path to saved model (.pt).

        Returns:
            A frozen torch.nn.Module on self.device.
        """
        try:
            predictor = torch.jit.load(ckpt_path, map_location=self.device)
            predictor.eval()
            for param in predictor.parameters():
                param.requires_grad_(False)
            logger.info("[AnticiPoseEnv] Loaded TorchScript wrench predictor.")
            return predictor
        except Exception:
            pass

        from humanoidverse.models.wrench_predictor import WrenchPredictor
        predictor = WrenchPredictor.from_checkpoint(ckpt_path, device=self.device)
        deriv_str = " (with plan derivatives)" if predictor._use_plan_derivatives else ""
        logger.info(f"[AnticiPoseEnv] Loaded state-dict wrench predictor{deriv_str}.")
        return predictor

    # ------------------------------------------------------------------
    # Payload simulation
    # ------------------------------------------------------------------

    def _apply_payload_forces(self):
        """Apply gravitational force at EE rigid bodies for payload sim.

        Each env carries a payload of mass self._payload_mass[i] kg.
        Force is split equally between left and right EE (rubber_hand).
        Applied as pure -Z force in world frame (gravity).
        """
        self._payload_force.zero_()
        # Split payload equally between two hands: F = -m*g/2 per hand
        per_hand_force = -self._payload_mass * 9.81 * 0.5  # (N,)
        for ee_idx in self._ee_body_indices:
            self._payload_force[:, ee_idx, 2] = per_hand_force
        self.simulator.apply_rigid_body_force_at_pos_tensor(
            self._payload_force, self.apply_force_pos_tensor,
        )

    # ------------------------------------------------------------------
    # Evaluation mode
    # ------------------------------------------------------------------

    def set_is_evaluating(self, command=None):
        """Configure the environment for evaluation."""
        super().set_is_evaluating(command)
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._arm_traj_gen.reset(all_ids)
        logger.info(
            f"[AnticiPoseEnv] Evaluation mode (mode={self._ap_mode})"
        )
