"""
wrench_data_collector.py
------------------------
GPU ring buffer that collects (obs, plan, wrench) tuples during RL rollouts
for offline training of the WrenchPredictor.

Design decisions
~~~~~~~~~~~~~~~~
- All storage tensors are pre-allocated on GPU at construction time to avoid
  repeated cudaMalloc and Python-level allocation during training.
- A ring buffer (circular) policy is used: once the buffer is full the oldest
  samples are overwritten.  This naturally provides a sliding-window dataset
  as the policy improves.
- add() accepts whole environment batches (num_envs samples at once) and
  inserts them via a vectorised index assignment, keeping GPU utilisation high.
- save() snapshots the current buffer contents to disk as a CPU tensor dict
  so the file can be read without a GPU.
- get_dataset() returns a view of the valid portion of the buffer (still on GPU)
  for in-process training loops.

Dimensions (from VERIFIED_PARAMS.md):
    obs    : 115  (per-step actor obs)
    plan   :  70  (H=5 x 14 arm joint positions)
    wrench :   6  (force_xyz + torque_xyz)

Capacity: 500,000 samples  (~460 MB at float32 on GPU)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch


# ---------------------------------------------------------------------------
# Dimension constants
# ---------------------------------------------------------------------------
OBS_DIM    = 115
PLAN_DIM   = 70
WRENCH_DIM = 6
CAPACITY   = 500_000


class WrenchDataCollector:
    """Pre-allocated GPU ring buffer for wrench supervision data.

    Args:
        capacity (int): Maximum number of samples to store before overwriting.
            Defaults to 500,000.
        device (str | torch.device): Device for all tensors.  Defaults to "cuda".

    Example::
        collector = WrenchDataCollector(device="cuda")

        # Inside the RL step loop:
        collector.add(obs_batch, plan_batch, wrench_batch)

        # Periodically save to disk:
        collector.save("wrench_data.pt")

        # Access for training:
        ds = collector.get_dataset()   # dict of (obs, plan, wrench) tensors
    """

    def __init__(self, capacity: int = CAPACITY, device: str | torch.device = "cuda"):
        self.capacity = capacity
        self.device   = torch.device(device)

        # ------------------------------------------------------------------
        # Pre-allocate ring-buffer storage on GPU.
        # All three tensors share the same write pointer so each sample slot
        # contains (obs[i], plan[i], wrench[i]) at the same index i.
        # ------------------------------------------------------------------
        self._obs    = torch.zeros(capacity, OBS_DIM,    dtype=torch.float32, device=self.device)
        self._plan   = torch.zeros(capacity, PLAN_DIM,   dtype=torch.float32, device=self.device)
        self._wrench = torch.zeros(capacity, WRENCH_DIM, dtype=torch.float32, device=self.device)

        # Write pointer: index at which the next batch of samples will be written.
        self._ptr: int = 0

        # Number of valid (written) samples.  Saturates at capacity.
        self._size: int = 0

    # ------------------------------------------------------------------
    # Core ring-buffer operations
    # ------------------------------------------------------------------

    def add(
        self,
        obs_batch:    torch.Tensor,
        plan_batch:   torch.Tensor,
        wrench_batch: torch.Tensor,
    ) -> None:
        """Insert a batch of samples into the ring buffer.

        Handles wrap-around automatically when the pointer exceeds capacity.

        Args:
            obs_batch    (torch.Tensor): Shape (B, 115).  Raw actor obs.
            plan_batch   (torch.Tensor): Shape (B,  70).  Arm plan.
            wrench_batch (torch.Tensor): Shape (B,   6).  Analytical wrench.
        """
        batch_size = obs_batch.shape[0]

        # Move to the collector device if necessary (handles CPU->GPU transfers).
        obs    = obs_batch.to(self.device, non_blocking=True)
        plan   = plan_batch.to(self.device, non_blocking=True)
        wrench = wrench_batch.to(self.device, non_blocking=True)

        if batch_size >= self.capacity:
            # Edge case: batch larger than the entire buffer.
            # Keep only the last `capacity` samples.
            obs    = obs[-self.capacity:]
            plan   = plan[-self.capacity:]
            wrench = wrench[-self.capacity:]
            self._obs[:]    = obs
            self._plan[:]   = plan
            self._wrench[:] = wrench
            self._ptr  = 0
            self._size = self.capacity
            return

        end = self._ptr + batch_size

        if end <= self.capacity:
            # Common case: fits without wrapping.
            indices = torch.arange(self._ptr, end, device=self.device)
        else:
            # Wrap-around: split into two segments.
            first_part  = self.capacity - self._ptr
            second_part = batch_size - first_part
            indices = torch.cat([
                torch.arange(self._ptr, self.capacity, device=self.device),
                torch.arange(0, second_part,          device=self.device),
            ])

        self._obs[indices]    = obs
        self._plan[indices]   = plan
        self._wrench[indices] = wrench

        self._ptr  = (self._ptr + batch_size) % self.capacity
        self._size = min(self._size + batch_size, self.capacity)

    def __len__(self) -> int:
        """Return the number of valid samples currently stored."""
        return self._size

    # ------------------------------------------------------------------
    # Dataset access
    # ------------------------------------------------------------------

    def get_dataset(self) -> dict[str, torch.Tensor]:
        """Return the valid portion of the buffer as a dict of tensors.

        The tensors remain on GPU and share memory with the internal buffers
        (no copy is made).  Do not write into the returned tensors.

        Returns:
            dict with keys "obs", "plan", "wrench", each of shape (N, dim)
            where N = len(self).

        Note:
            When the buffer is not yet full (self._size < self.capacity), only
            indices [0, self._size) are valid.  When it IS full the entire
            array is valid but samples are in circular order (oldest at self._ptr).
            For offline training order does not matter, so we return a contiguous
            slice without re-ordering.
        """
        n = self._size
        return {
            "obs":    self._obs[:n],
            "plan":   self._plan[:n],
            "wrench": self._wrench[:n],
        }

    def get_dataset_shuffled(self, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """Return the valid portion of the buffer in a random order.

        This creates copies of the data with shuffled rows, useful for
        mini-batch iteration without an external DataLoader.

        Args:
            generator: Optional torch.Generator for reproducibility.

        Returns:
            dict with keys "obs", "plan", "wrench" — shuffled copies.
        """
        n = self._size
        perm = torch.randperm(n, device=self.device, generator=generator)
        return {
            "obs":    self._obs[:n][perm].clone(),
            "plan":   self._plan[:n][perm].clone(),
            "wrench": self._wrench[:n][perm].clone(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the current buffer contents and metadata to disk.

        All tensors are moved to CPU before saving so the file is readable
        on machines without a GPU.

        Args:
            path (str): Destination file path (e.g., "wrench_data.pt").
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        n = self._size
        payload = {
            # Data tensors (CPU copies for portability)
            "obs":    self._obs[:n].cpu(),
            "plan":   self._plan[:n].cpu(),
            "wrench": self._wrench[:n].cpu(),
            # Metadata for reloading the ring-buffer state
            "meta": {
                "capacity": self.capacity,
                "size":     self._size,
                "ptr":      self._ptr,
                "obs_dim":    OBS_DIM,
                "plan_dim":   PLAN_DIM,
                "wrench_dim": WRENCH_DIM,
            },
        }
        torch.save(payload, save_path)

    @classmethod
    def load(cls, path: str, device: str | torch.device = "cuda") -> "WrenchDataCollector":
        """Restore a collector from a previously saved checkpoint.

        Args:
            path   (str): Path to a .pt file created by save().
            device: Target device for the ring buffers.

        Returns:
            A WrenchDataCollector with the saved data pre-loaded.
        """
        raw = torch.load(path, map_location="cpu", weights_only=True)
        meta = raw["meta"]

        collector = cls(capacity=meta["capacity"], device=device)
        n = meta["size"]

        # Copy saved tensors into the pre-allocated GPU buffers.
        collector._obs[:n].copy_(raw["obs"])
        collector._plan[:n].copy_(raw["plan"])
        collector._wrench[:n].copy_(raw["wrench"])

        collector._size = meta["size"]
        collector._ptr  = meta["ptr"]
        return collector

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fill_pct = 100.0 * self._size / self.capacity
        return (
            f"WrenchDataCollector("
            f"size={self._size:,}/{self.capacity:,} ({fill_pct:.1f}%), "
            f"ptr={self._ptr}, "
            f"device={self.device})"
        )

    def memory_bytes(self) -> int:
        """Return approximate GPU memory usage of the ring buffers in bytes."""
        bytes_per_element = 4  # float32
        total_elements = self.capacity * (OBS_DIM + PLAN_DIM + WRENCH_DIM)
        return total_elements * bytes_per_element

    def memory_mb(self) -> float:
        """Return approximate GPU memory usage in megabytes."""
        return self.memory_bytes() / (1024 ** 2)
