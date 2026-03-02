"""
Generate Figure 4: Torque time-series showing anticipatory postural adjustments.

Creates 3 vertically stacked panels (ankle torques, hip torques, base wrench)
comparing B1 (reactive) vs B5 (AnticiPose), aligned to arm motion onset.
If B5 learns APAs, expect torque divergence 50-100ms BEFORE disturbance onset.

Requires torque data from `make collect-torque` or `collect_torque_timeseries.py`.

Usage:
    cd FALCON/
    python scripts/plot_torque_timeseries.py \
        --data_dir torque_data \
        --output_dir paper/figures \
        --task lateral_shelf_pick
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# DOF name mapping (matches collect_torque_timeseries.py)
# Indices into the 11-DOF recorded tensor:
#   0: L_ankle_pitch, 1: L_ankle_roll,
#   2: R_ankle_pitch, 3: R_ankle_roll,
#   4: L_hip_pitch,   5: L_hip_roll,
#   6: R_hip_pitch,   7: R_hip_roll,
#   8: waist_yaw,     9: waist_roll, 10: waist_pitch
ANKLE_INDICES = [0, 1, 2, 3]
HIP_INDICES = [4, 5, 6, 7]
WAIST_INDICES = [8, 9, 10]

# Time grid for alignment
T_MIN = -1.0  # seconds before onset
T_MAX = 3.0   # seconds after onset
T_STEP = 0.02  # 50 Hz grid
MIN_POST_ONSET_S = 2.0  # only include episodes surviving at least this long after onset

B1_COLOR = "#7f7f7f"   # gray, consistent with other plots
B5_COLOR = "#1f77b4"   # blue, consistent with other plots


def load_data(path):
    """Load torque data .pt file."""
    return torch.load(path, map_location="cpu", weights_only=False)


def align_episodes(data, dof_indices, min_post_onset_s=MIN_POST_ONSET_S):
    """Align episodes to onset time, interpolate onto common time grid.

    Returns:
        time_grid: (G,) array
        aligned: (N_episodes, G) array of mean torque across specified DOFs
    """
    time_grid = np.arange(T_MIN, T_MAX + T_STEP / 2, T_STEP)
    episodes = data["episodes"]
    aligned_list = []

    for ep in episodes:
        onset = ep["onset_time_s"]
        time_s = ep["time_s"].numpy()
        torques = ep["torques"].numpy()  # (T, 11)

        # Time relative to onset
        t_rel = time_s - onset

        # Check episode extends enough after onset
        if t_rel[-1] < min_post_onset_s:
            continue

        # Average torque across the selected DOFs
        mean_torque = torques[:, dof_indices].mean(axis=1)

        # Interpolate onto common grid
        interp = np.interp(time_grid, t_rel, mean_torque,
                           left=np.nan, right=np.nan)
        aligned_list.append(interp)

    if not aligned_list:
        return time_grid, np.array([]).reshape(0, len(time_grid))

    return time_grid, np.array(aligned_list)


def align_wrench_component(data, component_idx,
                           min_post_onset_s=MIN_POST_ONSET_S):
    """Align a single wrench component across episodes.

    component_idx: 0=Fx, 1=Fy, 2=Fz, 3=Tx, 4=Ty, 5=Tz
    """
    time_grid = np.arange(T_MIN, T_MAX + T_STEP / 2, T_STEP)
    episodes = data["episodes"]
    aligned_list = []

    for ep in episodes:
        onset = ep["onset_time_s"]
        time_s = ep["time_s"].numpy()
        wrenches = ep["wrenches"].numpy()  # (T, 6)

        t_rel = time_s - onset
        if t_rel[-1] < min_post_onset_s:
            continue

        component = wrenches[:, component_idx]
        interp = np.interp(time_grid, t_rel, component,
                           left=np.nan, right=np.nan)
        aligned_list.append(interp)

    if not aligned_list:
        return time_grid, np.array([]).reshape(0, len(time_grid))

    return time_grid, np.array(aligned_list)


def plot_comparison(ax, time_grid, b1_aligned, b5_aligned,
                    ylabel, title):
    """Plot B1 vs B5 with mean ± std shading."""
    # Compute stats ignoring NaN
    b1_mean = np.nanmean(b1_aligned, axis=0)
    b1_std = np.nanstd(b1_aligned, axis=0)
    b5_mean = np.nanmean(b5_aligned, axis=0)
    b5_std = np.nanstd(b5_aligned, axis=0)

    ax.plot(time_grid, b1_mean, color=B1_COLOR, linewidth=1.5, label="B1 (Reactive)")
    ax.fill_between(time_grid, b1_mean - b1_std, b1_mean + b1_std,
                    color=B1_COLOR, alpha=0.15)

    ax.plot(time_grid, b5_mean, color=B5_COLOR, linewidth=1.5, label="B5 (AnticiPose)")
    ax.fill_between(time_grid, b5_mean - b5_std, b5_mean + b5_std,
                    color=B5_COLOR, alpha=0.15)

    # Onset line
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Anticipatory window shading
    ax.axvspan(-0.5, 0, color="yellow", alpha=0.08)
    ax.text(
        -0.25, ax.get_ylim()[1] * 0.9, "Anticipatory\nwindow",
        ha="center", fontsize=6, color="#888888", style="italic",
    )

    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)


def detect_divergence(time_grid, b1_aligned, b5_aligned):
    """Detect if B5 diverges from B1 before onset (t<0).

    Returns lead_time_ms (positive = anticipatory) or None.
    """
    pre_onset_mask = (time_grid >= -0.5) & (time_grid <= 0)
    if not np.any(pre_onset_mask):
        return None

    b1_mean = np.nanmean(b1_aligned, axis=0)
    b5_mean = np.nanmean(b5_aligned, axis=0)
    diff = np.abs(b5_mean - b1_mean)

    # Baseline diff: well before onset (-1.0 to -0.5)
    baseline_mask = (time_grid >= -1.0) & (time_grid < -0.5)
    if not np.any(baseline_mask):
        return None

    baseline_diff = np.nanmean(diff[baseline_mask])
    baseline_std = np.nanstd(diff[baseline_mask])

    if baseline_std < 1e-6:
        return None

    threshold = baseline_diff + 2 * baseline_std

    # Find first time in [-0.5, 0] where diff exceeds threshold
    pre_indices = np.where(pre_onset_mask)[0]
    for idx in pre_indices:
        if diff[idx] > threshold:
            lead_time_ms = -time_grid[idx] * 1000
            return lead_time_ms

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="torque_data")
    parser.add_argument("--output_dir", default="paper/figures")
    parser.add_argument("--task", default="lateral_shelf_pick")
    parser.add_argument(
        "--seeds", default="42,123,456,789,35",
        help="Seeds to aggregate (comma-separated)",
    )
    parser.add_argument(
        "--b1_pattern", default="B1_{task}_s{seed}.pt",
        help="Filename pattern for B1 data",
    )
    parser.add_argument(
        "--b5_pattern", default="B5_{task}_s{seed}.pt",
        help="Filename pattern for B5 data",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Aggregate episodes across seeds
    b1_episodes = []
    b5_episodes = []
    b1_meta = None
    b5_meta = None

    for seed in seeds:
        b1_file = data_dir / args.b1_pattern.format(task=args.task, seed=seed)
        b5_file = data_dir / args.b5_pattern.format(task=args.task, seed=seed)

        if b1_file.exists():
            d = load_data(b1_file)
            b1_episodes.extend(d["episodes"])
            b1_meta = d["metadata"]
            print(f"Loaded B1 seed {seed}: {len(d['episodes'])} episodes")
        else:
            print(f"[WARN] B1 not found: {b1_file}")

        if b5_file.exists():
            d = load_data(b5_file)
            b5_episodes.extend(d["episodes"])
            b5_meta = d["metadata"]
            print(f"Loaded B5 seed {seed}: {len(d['episodes'])} episodes")
        else:
            print(f"[WARN] B5 not found: {b5_file}")

    if not b1_episodes or not b5_episodes:
        print("[ERROR] Need data for both B1 and B5 to produce plot")
        return

    # Build combined data dicts
    b1_data = {"episodes": b1_episodes, "metadata": b1_meta}
    b5_data = {"episodes": b5_episodes, "metadata": b5_meta}

    print(f"\nTotal B1 episodes: {len(b1_episodes)}")
    print(f"Total B5 episodes: {len(b5_episodes)}")

    # Align data
    t_grid, b1_ankle = align_episodes(b1_data, ANKLE_INDICES)
    _, b5_ankle = align_episodes(b5_data, ANKLE_INDICES)
    _, b1_hip = align_episodes(b1_data, HIP_INDICES)
    _, b5_hip = align_episodes(b5_data, HIP_INDICES)
    _, b1_fy = align_wrench_component(b1_data, 1)  # lateral force
    _, b5_fy = align_wrench_component(b5_data, 1)

    print(f"Aligned B1 ankle episodes: {b1_ankle.shape[0]}")
    print(f"Aligned B5 ankle episodes: {b5_ankle.shape[0]}")

    # Create figure
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    plot_comparison(
        axes[0], t_grid, b1_ankle, b5_ankle,
        ylabel="Torque (Nm)", title="Ankle Torques (avg L+R pitch+roll)",
    )
    plot_comparison(
        axes[1], t_grid, b1_hip, b5_hip,
        ylabel="Torque (Nm)", title="Hip Torques (avg L+R pitch+roll)",
    )
    plot_comparison(
        axes[2], t_grid, b1_fy, b5_fy,
        ylabel="Force (N)", title="Base Lateral Force (Fy)",
    )

    axes[2].set_xlabel("Time relative to arm motion onset (s)", fontsize=10)

    # Detect anticipatory divergence
    lead_ankle = detect_divergence(t_grid, b1_ankle, b5_ankle)
    lead_hip = detect_divergence(t_grid, b1_hip, b5_hip)
    lead_fy = detect_divergence(t_grid, b1_fy, b5_fy)

    annotation_texts = []
    if lead_ankle is not None:
        annotation_texts.append(f"Ankle lead: {lead_ankle:.0f}ms")
        print(f"Ankle anticipatory lead: {lead_ankle:.0f}ms")
    if lead_hip is not None:
        annotation_texts.append(f"Hip lead: {lead_hip:.0f}ms")
        print(f"Hip anticipatory lead: {lead_hip:.0f}ms")
    if lead_fy is not None:
        annotation_texts.append(f"Fy lead: {lead_fy:.0f}ms")
        print(f"Lateral force anticipatory lead: {lead_fy:.0f}ms")

    if annotation_texts:
        fig.text(
            0.02, 0.02, "  |  ".join(annotation_texts),
            fontsize=8, color="#555555", style="italic",
        )

    fig.suptitle(
        f"Torque Time-Series: B1 vs B5 — {args.task.replace('_', ' ').title()}",
        fontsize=12, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    for fmt in ("pdf", "png"):
        path = out_dir / f"torque_timeseries.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
