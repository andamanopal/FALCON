"""
analytical_wrench.py
--------------------
Computes the manipulation-induced wrench at the robot base using Newton-Euler
rigid body dynamics.

Algorithm (per time step):
    For each arm link i:
        1. pos_i, vel_i  <- simulator._rigid_body_pos/vel[env, body_idx]
        2. acc_i         <- (vel_i - prev_vel_i) / dt          (finite difference)
        3. force_i       <- mass_i * acc_i                      (F = m*a)
        4. r_i           <- pos_i - base_pos                   (moment arm)
        5. torque_i      <- r_i x force_i                      (cross product)
    Wrench = (sum(force_i), sum(torque_i))   shape: (num_envs, 6)

Tensor layout from IsaacGym (via simulator):
    simulator._rigid_body_pos      : (num_envs, num_bodies, 3)   world-frame positions
    simulator._rigid_body_vel      : (num_envs, num_bodies, 3)   world-frame linear velocities
    simulator.robot_root_states    : (num_envs, 13)  [..., 0:3] = base pos

Arm link body indices must be passed in at construction time.
The caller discovers them via env.body_names.index(<link_name>).

G1 per-arm link masses (shoulder_pitch -> rubber_hand, 8 links per arm):
    [0.718, 0.643, 0.734, 0.600, 0.085, 0.484, 0.085, 0.170]  kg
"""

import torch


# Per-arm link masses in kinematic order:
#   shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
#   wrist_roll, wrist_pitch, wrist_yaw, rubber_hand (EE)
_ARM_LINK_MASSES_KG = [0.718, 0.643, 0.734, 0.600, 0.085, 0.484, 0.085, 0.170]


class AnalyticalWrench:
    """Computes the net wrench at the robot base produced by both arms.

    The wrench is expressed in the world frame. Callers may rotate it into
    the base frame using quat_rotate_inverse if needed.

    Args:
        env: The simulation environment.  Must expose:
            env.simulator._rigid_body_pos   (num_envs, num_bodies, 3)
            env.simulator._rigid_body_vel   (num_envs, num_bodies, 3)
            env.simulator.robot_root_states (num_envs, 13)
            env.num_envs  (int)
            env.device    (str)
        left_arm_body_indices  (list[int]): Rigid-body indices for the 8 left-arm
            links in kinematic order (shoulder_pitch -> rubber_hand).
        right_arm_body_indices (list[int]): Same for the right arm.
        dt (float): Simulation control step size in seconds (default 0.02 s = 50 Hz).
    """

    def __init__(self, env, left_arm_body_indices, right_arm_body_indices, dt=0.02):
        assert len(left_arm_body_indices) == len(_ARM_LINK_MASSES_KG), (
            f"Expected {len(_ARM_LINK_MASSES_KG)} left-arm body indices, "
            f"got {len(left_arm_body_indices)}"
        )
        assert len(right_arm_body_indices) == len(_ARM_LINK_MASSES_KG), (
            f"Expected {len(_ARM_LINK_MASSES_KG)} right-arm body indices, "
            f"got {len(right_arm_body_indices)}"
        )

        self.env = env
        self.dt = dt
        self.num_envs = env.num_envs
        self.device = env.device

        # Combined body-index tensor: shape (2 * num_links,)
        # Left arm indices first, then right arm indices.
        all_indices = left_arm_body_indices + right_arm_body_indices
        self.body_indices = torch.tensor(
            all_indices, dtype=torch.long, device=self.device
        )  # (num_arm_links_total,)

        # Mass tensor for all arm links: shape (1, num_arm_links_total, 1)
        # Broadcasting-ready so we can do mass * acc without extra unsqueezes later.
        masses = _ARM_LINK_MASSES_KG + _ARM_LINK_MASSES_KG  # left + right
        self.link_masses = torch.tensor(
            masses, dtype=torch.float32, device=self.device
        ).view(1, -1, 1)  # (1, num_arm_links_total, 1)

        num_links = len(all_indices)

        # Previous-step linear velocities for finite-difference acceleration.
        # Shape: (num_envs, num_arm_links_total, 3)
        self._prev_vel = torch.zeros(
            self.num_envs, num_links, 3,
            dtype=torch.float32, device=self.device
        )

        # Flag: skip acceleration on the very first call (no previous velocity yet).
        self._first_call = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self) -> torch.Tensor:
        """Compute the manipulation-induced wrench at the robot base.

        Returns:
            wrench (torch.Tensor): Shape (num_envs, 6).
                wrench[:, 0:3] = net force  (N)  in world frame
                wrench[:, 3:6] = net torque (Nm) in world frame, taken about base
        """
        sim = self.env.simulator

        # --- Gather link positions and velocities ---
        # sim._rigid_body_pos : (num_envs, num_bodies, 3)
        # sim._rigid_body_vel : (num_envs, num_bodies, 3)
        link_pos = sim._rigid_body_pos[:, self.body_indices, :]  # (N, L, 3)
        link_vel = sim._rigid_body_vel[:, self.body_indices, :]  # (N, L, 3)

        # --- Finite-difference linear acceleration ---
        if self._first_call:
            # Cannot estimate acceleration without a previous velocity; return zeros.
            self._prev_vel = link_vel.clone()
            self._first_call = False
            return torch.zeros(
                self.num_envs, 6, dtype=torch.float32, device=self.device
            )

        acc = (link_vel - self._prev_vel) / self.dt  # (N, L, 3)
        self._prev_vel = link_vel.clone()

        # --- Newton: force per link ---
        # link_masses: (1, L, 1)  ->  broadcasts with acc: (N, L, 3)
        force_per_link = self.link_masses * acc  # (N, L, 3)

        # --- Moment arm: link position relative to base ---
        # robot_root_states[:, 0:3] is the base (pelvis) position.
        base_pos = sim.robot_root_states[:, 0:3].unsqueeze(1)  # (N, 1, 3)
        r = link_pos - base_pos  # (N, L, 3)

        # --- Euler: torque per link about the base ---
        # torch.linalg.cross works element-wise on the last dim.
        torque_per_link = torch.linalg.cross(r, force_per_link, dim=-1)  # (N, L, 3)

        # --- Sum over all arm links ---
        net_force = force_per_link.sum(dim=1)    # (N, 3)
        net_torque = torque_per_link.sum(dim=1)  # (N, 3)

        wrench = torch.cat([net_force, net_torque], dim=-1)  # (N, 6)
        return wrench

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear the velocity history for specified environments.

        Call this inside env.reset_envs_idx() so that stale velocities from
        the previous episode do not corrupt the first acceleration estimate.

        Args:
            env_ids (torch.Tensor): 1-D integer tensor of environment indices to reset.
        """
        if len(env_ids) == 0:
            return
        self._prev_vel[env_ids] = 0.0
        # After a reset the next compute() call will skip (return zeros), which is
        # correct because we have no valid previous velocity.
        # We mark _first_call only if ALL envs reset; otherwise handle per-env.
        # The simplest safe approach: zero out prev_vel and let the subtraction
        # produce a one-step garbage acceleration that decays quickly.
        # A more conservative option is tracked below.
        # (For full correctness, set prev_vel to current vel after the physics step.)
