"""
wrench_data_collector.py
------------------------
Per-env FIFO queues that collect (obs, plan, wrench) tuples during RL rollouts
with temporal-offset alignment for offline training of the WrenchPredictor.

Design
~~~~~~
The predictor must learn: (obs_t, plan_t) -> [wrench_{t+1}, ..., wrench_{t+H}]

To produce correct training pairs, we maintain a circular buffer of length H+1
per environment.  At each step we push (obs_t, plan_t, wrench_t).  When the
buffer reaches H+1 entries, we yield one training pair:

    input:  (obs_{t-H}, plan_{t-H})
    target: [wrench_{t-H+1}, wrench_{t-H+2}, ..., wrench_t]   (H x 6 = 30 dims)

On env reset, we flush that env's queue to avoid cross-episode pairs.

All storage is pre-allocated on GPU.  Completed training pairs are appended
into a large output ring buffer for later retrieval / saving.

Dimensions (from VERIFIED_PARAMS.md):
    obs    : 115  (per-step actor obs)
    plan   :  70  (H=5 x 14 arm joint positions)
    wrench :   6  (force_xyz + torque_xyz)
    target : 30   (H x 6 future wrenches)

Capacity: 500,000 training pairs (~460 MB at float32 on GPU)
"""

from __future__ import annotations

from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Dimension constants
# ---------------------------------------------------------------------------
OBS_DIM = 115
PLAN_DIM = 70
WRENCH_DIM = 6
DEFAULT_HORIZON = 5
TARGET_DIM = DEFAULT_HORIZON * WRENCH_DIM  # 30
CAPACITY = 500_000


class WrenchDataCollector:
    """Per-env FIFO queues with temporal-offset alignment for wrench data.

    Args:
        num_envs (int): Number of parallel environments.
        horizon (int): Prediction horizon H (default 5).
        capacity (int): Max number of aligned training pairs to store.
        device (str | torch.device): Device for all tensors.
    """

    def __init__(
        self,
        num_envs: int = 4096,
        horizon: int = DEFAULT_HORIZON,
        capacity: int = CAPACITY,
        device: str | torch.device = "cuda",
    ):
        self.num_envs = num_envs
        self.horizon = horizon
        self.capacity = capacity
        self.device = torch.device(device)
        self._queue_len = horizon + 1  # Need H+1 steps to form one pair

        # Per-env circular FIFO queues
        # Shape: (num_envs, queue_len, dim)
        self._q_obs = torch.zeros(
            num_envs, self._queue_len, OBS_DIM,
            dtype=torch.float32, device=self.device,
        )
        self._q_plan = torch.zeros(
            num_envs, self._queue_len, PLAN_DIM,
            dtype=torch.float32, device=self.device,
        )
        self._q_wrench = torch.zeros(
            num_envs, self._queue_len, WRENCH_DIM,
            dtype=torch.float32, device=self.device,
        )
        # Per-env write pointer and fill count
        self._q_ptr = torch.zeros(
            num_envs, dtype=torch.long, device=self.device,
        )
        self._q_fill = torch.zeros(
            num_envs, dtype=torch.long, device=self.device,
        )

        # Output ring buffer for aligned training pairs
        self._out_obs = torch.zeros(
            capacity, OBS_DIM, dtype=torch.float32, device=self.device,
        )
        self._out_plan = torch.zeros(
            capacity, PLAN_DIM, dtype=torch.float32, device=self.device,
        )
        self._out_target = torch.zeros(
            capacity, TARGET_DIM, dtype=torch.float32, device=self.device,
        )
        self._out_ptr: int = 0
        self._out_size: int = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def push_step(
        self,
        obs_batch: torch.Tensor,
        plan_batch: torch.Tensor,
        wrench_batch: torch.Tensor,
    ) -> None:
        """Push one timestep of data for all envs.

        Args:
            obs_batch    (torch.Tensor): Shape (num_envs, 115).
            plan_batch   (torch.Tensor): Shape (num_envs, 70).
            wrench_batch (torch.Tensor): Shape (num_envs, 6).
        """
        n = self.num_envs
        ptr = self._q_ptr  # (num_envs,)

        # Write into per-env queues at current pointer
        env_idx = torch.arange(n, device=self.device)
        self._q_obs[env_idx, ptr] = obs_batch.to(self.device)
        self._q_plan[env_idx, ptr] = plan_batch.to(self.device)
        self._q_wrench[env_idx, ptr] = wrench_batch.to(self.device)

        # Advance pointer and fill count
        self._q_ptr = (ptr + 1) % self._queue_len
        self._q_fill = torch.clamp(self._q_fill + 1, max=self._queue_len)

        # Check which envs have full queues (fill == queue_len)
        ready_mask = self._q_fill >= self._queue_len
        if not ready_mask.any():
            return

        ready_ids = ready_mask.nonzero(as_tuple=False).squeeze(-1)
        self._emit_pairs(ready_ids)

    def flush_envs(self, env_ids: torch.Tensor) -> None:
        """Flush queues for specified environments (call on env reset).

        This prevents cross-episode temporal pairs.

        Args:
            env_ids (torch.Tensor): 1-D tensor of env indices to flush.
        """
        if len(env_ids) == 0:
            return
        self._q_ptr[env_ids] = 0
        self._q_fill[env_ids] = 0
        self._q_obs[env_ids] = 0.0
        self._q_plan[env_ids] = 0.0
        self._q_wrench[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Internal: emit aligned training pairs
    # ------------------------------------------------------------------

    def _emit_pairs(self, env_ids: torch.Tensor) -> None:
        """Extract aligned (obs, plan) -> future_wrench pairs from full queues.

        For each ready env, the oldest entry in the circular buffer is the
        input (obs_{t-H}, plan_{t-H}), and the subsequent H entries are
        the target wrench sequence.

        The queue pointer currently points to the NEXT write slot, which
        (in a full circular buffer) is also the oldest slot.
        """
        num_ready = env_ids.shape[0]
        if num_ready == 0:
            return

        # Oldest slot index = current write pointer (about to be overwritten)
        oldest_ptr = self._q_ptr[env_ids]  # (num_ready,)

        # Input: (obs, plan) from the oldest slot
        input_obs = self._q_obs[env_ids, oldest_ptr]    # (num_ready, 115)
        input_plan = self._q_plan[env_ids, oldest_ptr]   # (num_ready, 70)

        # Target: H wrenches from slots [oldest+1, oldest+2, ..., oldest+H]
        # These are the H future wrenches relative to the input timestep.
        target_parts = []
        for h in range(1, self.horizon + 1):
            slot = (oldest_ptr + h) % self._queue_len  # (num_ready,)
            wrench_h = self._q_wrench[env_ids, slot]   # (num_ready, 6)
            target_parts.append(wrench_h)

        # Concatenate: (num_ready, H*6 = 30)
        target_wrench = torch.cat(target_parts, dim=-1)

        # Write into output ring buffer
        self._write_output(input_obs, input_plan, target_wrench)

    def _write_output(
        self,
        obs: torch.Tensor,
        plan: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Append aligned pairs to the output ring buffer."""
        batch_size = obs.shape[0]

        if batch_size >= self.capacity:
            obs = obs[-self.capacity:]
            plan = plan[-self.capacity:]
            target = target[-self.capacity:]
            batch_size = self.capacity

        end = self._out_ptr + batch_size

        if end <= self.capacity:
            indices = torch.arange(
                self._out_ptr, end, device=self.device,
            )
        else:
            first_part = self.capacity - self._out_ptr
            indices = torch.cat([
                torch.arange(
                    self._out_ptr, self.capacity, device=self.device,
                ),
                torch.arange(
                    0, batch_size - first_part, device=self.device,
                ),
            ])

        self._out_obs[indices] = obs
        self._out_plan[indices] = plan
        self._out_target[indices] = target

        self._out_ptr = (self._out_ptr + batch_size) % self.capacity
        self._out_size = min(self._out_size + batch_size, self.capacity)

    # ------------------------------------------------------------------
    # Dataset access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of valid aligned training pairs."""
        return self._out_size

    def get_dataset(self) -> dict[str, torch.Tensor]:
        """Return the valid portion of the output buffer as a dict.

        Returns:
            dict with keys "obs" (N, 115), "plan" (N, 70),
            "wrench" (N, 30) — 30-dim future wrench targets.
        """
        n = self._out_size
        return {
            "obs": self._out_obs[:n],
            "plan": self._out_plan[:n],
            "wrench": self._out_target[:n],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save aligned training pairs to disk.

        The wrench field is 30-dim (H*6) future wrench targets,
        NOT 6-dim current wrenches.

        Args:
            path: File path for the .pt output.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        n = self._out_size
        payload = {
            "obs": self._out_obs[:n].cpu(),
            "plan": self._out_plan[:n].cpu(),
            "wrench": self._out_target[:n].cpu(),
            "meta": {
                "capacity": self.capacity,
                "size": self._out_size,
                "horizon": self.horizon,
                "obs_dim": OBS_DIM,
                "plan_dim": PLAN_DIM,
                "wrench_dim": WRENCH_DIM,
                "target_dim": self.horizon * WRENCH_DIM,
            },
        }
        torch.save(payload, save_path)

    @classmethod
    def load(
        cls,
        path: str,
        device: str | torch.device = "cuda",
    ) -> "WrenchDataCollector":
        """Restore a collector from a previously saved checkpoint.

        Note: Only the output ring buffer is restored.  Per-env FIFO
        queues are reset (they are only needed during live collection).
        """
        raw = torch.load(path, map_location="cpu", weights_only=True)
        meta = raw["meta"]

        collector = cls(
            num_envs=1,  # Queues not needed for offline loading
            horizon=meta["horizon"],
            capacity=meta["capacity"],
            device=device,
        )
        n = meta["size"]

        collector._out_obs[:n].copy_(raw["obs"])
        collector._out_plan[:n].copy_(raw["plan"])
        collector._out_target[:n].copy_(raw["wrench"])

        collector._out_size = n
        collector._out_ptr = n % meta["capacity"]
        return collector

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fill_pct = 100.0 * self._out_size / self.capacity
        return (
            f"WrenchDataCollector("
            f"pairs={self._out_size:,}/{self.capacity:,} ({fill_pct:.1f}%), "
            f"horizon={self.horizon}, "
            f"num_envs={self.num_envs}, "
            f"device={self.device})"
        )
