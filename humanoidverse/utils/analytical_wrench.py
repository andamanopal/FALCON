"""
analytical_wrench.py
--------------------
Computes the manipulation-induced wrench at the robot base using Newton-Euler
rigid body dynamics, expressed in the base (body) frame.

Algorithm (per time step):
    For each arm link i:
        1. pos_i, vel_i  <- simulator._rigid_body_pos/vel[env, body_idx]
        2. acc_i         <- (vel_i - prev_vel_i) / dt          (finite difference)
        3. force_i       <- mass_i * acc_i                      (F = m*a)
        4. r_i           <- pos_i - base_pos                   (moment arm)
        5. torque_i      <- r_i x force_i                      (cross product)
    Wrench_world = (sum(force_i), sum(torque_i))

    Optional payload contribution:
        For each EE link, add F_payload = [0, 0, -m_payload * g / 2]

    Frame rotation (world -> base):
        force_base  = quat_rotate_inverse(base_quat, force_world)
        torque_base = quat_rotate_inverse(base_quat, torque_world)

Tensor layout from IsaacGym (via simulator):
    simulator._rigid_body_pos      : (num_envs, num_bodies, 3)   world-frame positions
    simulator._rigid_body_vel      : (num_envs, num_bodies, 3)   world-frame linear velocities
    simulator.robot_root_states    : (num_envs, 13)  [..., 0:3] = base pos, [3:7] = base quat

G1 per-arm link masses (shoulder_pitch -> rubber_hand, 8 links per arm):
    [0.718, 0.643, 0.734, 0.600, 0.085, 0.484, 0.085, 0.170]  kg
"""

from typing import Optional

import torch


# Per-arm link masses in kinematic order:
#   shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
#   wrist_roll, wrist_pitch, wrist_yaw, rubber_hand (EE)
_ARM_LINK_MASSES_KG = [0.718, 0.643, 0.734, 0.600, 0.085, 0.484, 0.085, 0.170]

# EE link index within each arm (rubber_hand = last link)
_EE_LINK_IDX_IN_ARM = 7


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by the inverse of quaternion q (wxyz convention).

    IsaacGym uses (x, y, z, w) convention in robot_root_states.

    Args:
        q: (N, 4) quaternion in (x, y, z, w) format.
        v: (N, 3) vector to rotate.

    Returns:
        (N, 3) rotated vector.
    """
    q_xyz = q[:, 0:3]
    q_w = q[:, 3:4]
    # v' = v - 2 * w * (q_xyz x v) + 2 * (q_xyz x (q_xyz x v))
    # Using the conjugate rotation formula.
    t = 2.0 * torch.linalg.cross(q_xyz, v, dim=-1)
    return v - q_w * t + torch.linalg.cross(q_xyz, t, dim=-1)


class AnalyticalWrench:
    """Computes the net wrench at the robot base produced by both arms.

    The wrench is expressed in the base (body) frame for consistency
    across different base orientations.

    Args:
        env: The simulation environment.
        left_arm_body_indices  (list[int]): Rigid-body indices for 8 left-arm links.
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
        all_indices = left_arm_body_indices + right_arm_body_indices
        self.body_indices = torch.tensor(
            all_indices, dtype=torch.long, device=self.device,
        )

        # EE indices within the combined array (for payload force)
        num_links_per_arm = len(_ARM_LINK_MASSES_KG)
        self._left_ee_idx = _EE_LINK_IDX_IN_ARM
        self._right_ee_idx = num_links_per_arm + _EE_LINK_IDX_IN_ARM

        # Mass tensor: shape (1, num_arm_links_total, 1)
        masses = _ARM_LINK_MASSES_KG + _ARM_LINK_MASSES_KG
        self.link_masses = torch.tensor(
            masses, dtype=torch.float32, device=self.device,
        ).view(1, -1, 1)

        num_links = len(all_indices)

        # Previous-step linear velocities for finite-difference acceleration.
        self._prev_vel = torch.zeros(
            self.num_envs, num_links, 3,
            dtype=torch.float32, device=self.device,
        )

        self._first_call = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        payload_mass: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the manipulation-induced wrench at the robot base.

        Args:
            payload_mass: Optional (num_envs,) tensor of payload mass per env.
                If provided, adds gravitational payload force at EE links.

        Returns:
            wrench (torch.Tensor): Shape (num_envs, 6) in BASE frame.
                wrench[:, 0:3] = net force  (N)
                wrench[:, 3:6] = net torque (Nm), taken about base
        """
        sim = self.env.simulator

        # --- Gather link positions and velocities ---
        link_pos = sim._rigid_body_pos[:, self.body_indices, :]  # (N, L, 3)
        link_vel = sim._rigid_body_vel[:, self.body_indices, :]  # (N, L, 3)

        # --- Finite-difference linear acceleration ---
        if self._first_call:
            self._prev_vel = link_vel.clone()
            self._first_call = False
            return torch.zeros(
                self.num_envs, 6, dtype=torch.float32, device=self.device,
            )

        acc = (link_vel - self._prev_vel) / self.dt  # (N, L, 3)
        self._prev_vel = link_vel.clone()

        # --- Newton: force per link ---
        force_per_link = self.link_masses * acc  # (N, L, 3)

        # --- Add payload gravitational force at EE links ---
        if payload_mass is not None:
            # payload_mass: (N,) -> (N, 1)
            per_hand_force = -payload_mass.unsqueeze(-1) * 9.81 * 0.5  # (N, 1)
            payload_z = torch.zeros(
                self.num_envs, force_per_link.shape[1], 3,
                dtype=torch.float32, device=self.device,
            )
            payload_z[:, self._left_ee_idx, 2] = per_hand_force.squeeze(-1)
            payload_z[:, self._right_ee_idx, 2] = per_hand_force.squeeze(-1)
            force_per_link = force_per_link + payload_z

        # --- Moment arm: link position relative to base ---
        base_pos = sim.robot_root_states[:, 0:3].unsqueeze(1)  # (N, 1, 3)
        r = link_pos - base_pos  # (N, L, 3)

        # --- Euler: torque per link about the base ---
        torque_per_link = torch.linalg.cross(r, force_per_link, dim=-1)

        # --- Sum over all arm links ---
        net_force_world = force_per_link.sum(dim=1)    # (N, 3)
        net_torque_world = torque_per_link.sum(dim=1)  # (N, 3)

        # --- Rotate from world frame to base (body) frame ---
        base_quat = sim.robot_root_states[:, 3:7]  # (N, 4) in (x,y,z,w)
        net_force_base = _quat_rotate_inverse(base_quat, net_force_world)
        net_torque_base = _quat_rotate_inverse(base_quat, net_torque_world)

        wrench = torch.cat([net_force_base, net_torque_base], dim=-1)
        return wrench

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear the velocity history for specified environments.

        Call this inside env.reset_envs_idx() so that stale velocities from
        the previous episode do not corrupt the first acceleration estimate.

        Args:
            env_ids (torch.Tensor): 1-D integer tensor of environment indices.
        """
        if len(env_ids) == 0:
            return
        self._prev_vel[env_ids] = 0.0
