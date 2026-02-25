"""AnticiPose Environment: Anticipatory whole-body control under arm manipulation.

Uses FALCON's YAML-driven observation dispatch so that:
  - pre_process_config() auto-sums obs dims -> PPO gets correct input size
  - History shifting includes extra components in each frame
  - Adding/removing obs components is just a YAML change

Observation getters (FALCON dispatches _get_obs_<name>() automatically):
  _get_obs_oracle_wrench     -> (N, H*6=30)  GT future wrenches (B2)
  _get_obs_predicted_wrench  -> (N, H*6=30)  Predicted wrenches (B5)
  _get_obs_arm_plan          -> (N, H*14=70) Raw arm plan (B4a)
  _get_obs_current_wrench    -> (N, 6)       Current wrench (critic)

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

    Adds scripted arm trajectories and wrench-aware observations on top of
    FALCON's decoupled WBC force environment.  All AnticiPose-specific logic
    is confined to hook overrides --- no parent method is copy-pasted.

    Config keys under ``config.env.config``:
        anticipose_mode (str): "reactive" | "oracle" | "direct_plan" | "anticipose"
        anticipose_horizon (int): H, default 5.
        arm_trajectory_task (str): Task type for trajectory generator.
        collect_wrench_data (bool): Whether to collect supervision data.
        collect_buffer_size (int): Ring buffer capacity for data collection.
        wrench_predictor_ckpt (str | None): Path to frozen predictor checkpoint.
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

        assert self._ap_mode in {
            "reactive", "oracle", "direct_plan", "anticipose"
        }, (
            f"Unknown anticipose_mode: {self._ap_mode!r}. "
            "Choose from: reactive, oracle, direct_plan, anticipose."
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

        # ---- Wrench data collector (B1 data-gathering stage) ----
        self._wrench_collector = None
        if self._ap_collect:
            from humanoidverse.utils.wrench_data_collector import (
                WrenchDataCollector,
            )
            capacity = getattr(self.config, "collect_buffer_size", 500_000)
            self._wrench_collector = WrenchDataCollector(
                capacity=capacity, device=device,
            )
            logger.info(
                f"[AnticiPoseEnv] Wrench data collection enabled "
                f"({capacity:,} capacity)"
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
        # Oracle wrench buffer: H future GT wrenches flattened (N, H*6)
        self._oracle_wrench_buf = torch.zeros(
            n, H * _WRENCH_DIM, dtype=torch.float32, device=self.device,
        )
        # Predicted wrench buffer: same shape, filled by frozen predictor
        self._predicted_wrench_buf = torch.zeros(
            n, H * _WRENCH_DIM, dtype=torch.float32, device=self.device,
        )
        # Arm plan buffer: H future arm joint targets flattened (N, H*14)
        self._arm_plan_buf = torch.zeros(
            n, H * _ARM_JOINTS, dtype=torch.float32, device=self.device,
        )
        # Current-step arm targets: (N, 14)
        self._arm_targets = torch.zeros(
            n, _ARM_JOINTS, dtype=torch.float32, device=self.device,
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
        """Override to inject scripted arm targets before PD control.

        The PD controller computes:
          torque = Kp * (action * scale + default_dof_pos - dof_pos) - Kd * dof_vel

        Setting action = (desired - default_dof_pos) / scale makes the
        controller drive the arm joints to the desired trajectory positions.
        """
        self._advance_arm_trajectory()

        scale = self.config.robot.control.action_scale
        modified = actions.clone()
        modified[:, _ARM_DOF_START:_ARM_DOF_END] = (
            self._arm_targets
            - self.default_dof_pos[:, _ARM_DOF_START:_ARM_DOF_END]
        ) / scale

        super()._pre_physics_step(modified)

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

        # Disable FALCON's external EE forces --- disturbance comes from
        # arm motion dynamics only, not random external forces.
        if hasattr(self, "left_ee_apply_force"):
            self.left_ee_apply_force.zero_()
        if hasattr(self, "right_ee_apply_force"):
            self.right_ee_apply_force.zero_()

        # Compute ground-truth wrench from arm rigid body dynamics
        self._current_wrench[:] = self._analytical_wrench.compute()

        # Fill oracle wrench buffer (tile current wrench H times for MVP;
        # true future wrenches would require forward simulation).
        self._oracle_wrench_buf[:] = self._current_wrench.repeat(
            1, self._ap_horizon,
        )

        # Run frozen predictor if loaded
        if self._wrench_predictor is not None:
            self._run_wrench_predictor()

    # ------------------------------------------------------------------
    # Post-observation callback: data collection
    # ------------------------------------------------------------------

    def _post_compute_observations_callback(self):
        """Override to collect wrench supervision data after obs assembly."""
        super()._post_compute_observations_callback()

        if self._wrench_collector is not None:
            obs_step = self._build_predictor_obs()
            self._wrench_collector.add(
                obs_step, self._arm_plan_buf, self._current_wrench,
            )

    # ------------------------------------------------------------------
    # Observation getters (FALCON dispatches _get_obs_<name>() via YAML)
    # ------------------------------------------------------------------

    def _get_obs_oracle_wrench(self) -> torch.Tensor:
        """GT future wrenches. Shape: (N, H*6=30). Used by B2 oracle."""
        return self._oracle_wrench_buf

    def _get_obs_predicted_wrench(self) -> torch.Tensor:
        """Predicted future wrenches. Shape: (N, H*6=30). Used by B5."""
        return self._predicted_wrench_buf

    def _get_obs_arm_plan(self) -> torch.Tensor:
        """Raw future arm joint plan. Shape: (N, H*14=70). Used by B4a."""
        return self._arm_plan_buf

    def _get_obs_current_wrench(self) -> torch.Tensor:
        """Current-step base wrench. Shape: (N, 6). Critic-only obs."""
        return self._current_wrench

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
    # Reset handling
    # ------------------------------------------------------------------

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        """Extend parent reset to clear AnticiPose-specific buffers."""
        if len(env_ids) == 0:
            return

        self._arm_traj_gen.reset(env_ids)
        self._analytical_wrench.reset(env_ids)
        self._current_wrench[env_ids] = 0.0
        self._oracle_wrench_buf[env_ids] = 0.0
        self._predicted_wrench_buf[env_ids] = 0.0
        self._arm_plan_buf[env_ids] = 0.0
        self._arm_targets[env_ids] = 0.0

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
        predictor = WrenchPredictor().to(self.device)
        predictor.load_frozen(ckpt_path)
        logger.info("[AnticiPoseEnv] Loaded state-dict wrench predictor.")
        return predictor

    # ------------------------------------------------------------------
    # Reward extensions
    # ------------------------------------------------------------------

    def _reward_wrench_anticipation_bonus(self) -> torch.Tensor:
        """Bonus for maintaining stability under high predicted wrench.

        Scale controlled by YAML: reward_scales.wrench_anticipation_bonus.
        """
        pred_reshaped = self._predicted_wrench_buf.reshape(
            self.num_envs, self._ap_horizon, _WRENCH_DIM,
        )
        wrench_norms = torch.norm(pred_reshaped, dim=-1)  # (N, H)
        max_wrench = wrench_norms.max(dim=-1).values       # (N,)

        upright_reward = 1.0 - torch.sum(
            torch.abs(self.projected_gravity[:, :2]), dim=-1,
        )
        wrench_scale = torch.clamp(max_wrench / 50.0, 0.0, 1.0)
        return upright_reward * wrench_scale

    def _reward_penalty_wrench_prediction_error(self) -> torch.Tensor:
        """Penalty proportional to prediction error (analysis only)."""
        error = self._predicted_wrench_buf - self._oracle_wrench_buf
        return torch.norm(error, dim=-1)

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
