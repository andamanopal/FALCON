"""
Generate Figure 5: Per-task breakdown grouped bar chart.

Reads per-task eval JSONs and creates grouped bars showing each baseline's
performance on each of the 5 arm tasks.

Usage:
    cd FALCON/
    python scripts/plot_per_task_bars.py \
        --eval_dir logs_eval \
        --output_dir paper/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASELINES = ["B1", "B2", "B3", "B4a", "B4b", "B5", "B6"]
BASELINE_PATTERNS = {
    "B1": "B1_reactive",
    "B2": "B2_extended_history",
    "B3": "B3_current_wrench",
    "B4a": "B4a_direct_plan",
    "B4b": "B4b_direct_plan_critic",
    "B5": "B5_anticipose",
    "B6": "B6_cvae",
}
TASKS_TRAIN = ["frontal_reach_lift", "lateral_shelf_pick", "forward_push"]
TASKS_HELDOUT = ["lateral_slam_down", "cross_body_reach"]
ALL_TASKS = TASKS_TRAIN + TASKS_HELDOUT
TASK_LABELS = {
    "frontal_reach_lift": "Frontal\nReach-Lift",
    "lateral_shelf_pick": "Lateral\nShelf-Pick",
    "forward_push": "Forward\nPush",
    "lateral_slam_down": "Lateral\nSlam-Down*",
    "cross_body_reach": "Cross-Body\nReach*",
}
SEEDS = [42, 123, 456, 789, 35]

COLORS = {
    "B1": "#7f7f7f",
    "B2": "#9467bd",
    "B3": "#2ca02c",
    "B4a": "#ff7f0e",
    "B4b": "#d62728",
    "B5": "#1f77b4",
    "B6": "#8c564b",
}


def load_result(eval_dirs, baseline, task, seed):
    """Load a per-task result JSON."""
    pattern = BASELINE_PATTERNS[baseline]
    name = f"eval_{pattern}_{task}_s{seed}"
    for d in eval_dirs:
        path = Path(d) / name / "results.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def collect_per_task(eval_dirs, metric="mean_reward"):
    """Collect metric per baseline x task, averaged over seeds."""
    means = {}
    stds = {}
    for bl in BASELINES:
        means[bl] = []
        stds[bl] = []
        for task in ALL_TASKS:
            vals = []
            for seed in SEEDS:
                r = load_result(eval_dirs, bl, task, seed)
                if r is not None:
                    vals.append(r[metric])
            if vals:
                means[bl].append(np.mean(vals))
                stds[bl].append(np.std(vals))
            else:
                means[bl].append(np.nan)
                stds[bl].append(np.nan)
    return means, stds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_dirs",
        default="logs_eval,../logs_eval_s42_rsync,../logs_eval_s789_rsync",
        help="Comma-separated eval directories (relative to CWD)",
    )
    parser.add_argument("--output_dir", default="paper/figures")
    parser.add_argument("--metric", default="mean_reward")
    args = parser.parse_args()

    eval_dirs = [d.strip() for d in args.eval_dirs.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    means, stds = collect_per_task(eval_dirs, args.metric)

    # Report data availability
    print("\nPer-task data availability:")
    for bl in BASELINES:
        for i, task in enumerate(ALL_TASKS):
            count = 0
            for seed in SEEDS:
                if load_result(eval_dirs, bl, task, seed) is not None:
                    count += 1
            if count < 5:
                print(f"  {bl} x {task}: {count}/5 seeds")

    # Create figure
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    n_tasks = len(ALL_TASKS)
    n_bl = len(BASELINES)
    bar_width = 0.11
    group_gap = 0.3

    fig, ax = plt.subplots(figsize=(12, 5))

    # Compute x positions with gap between train and held-out
    group_centers = []
    x_pos = 0.0
    for i in range(n_tasks):
        if i == len(TASKS_TRAIN):
            x_pos += group_gap
        group_centers.append(x_pos)
        x_pos += 1.0

    for j, bl in enumerate(BASELINES):
        offsets = [
            gc + (j - n_bl / 2 + 0.5) * bar_width
            for gc in group_centers
        ]
        ax.bar(
            offsets, means[bl], bar_width,
            yerr=stds[bl], capsize=2,
            color=COLORS[bl], alpha=0.8,
            edgecolor="white", linewidth=0.5,
            label=bl,
            error_kw={"linewidth": 0.8},
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [TASK_LABELS[t] for t in ALL_TASKS], fontsize=8,
    )

    # Add separator between train and held-out
    sep_x = (group_centers[len(TASKS_TRAIN) - 1]
             + group_centers[len(TASKS_TRAIN)]) / 2
    ax.axvline(sep_x, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(
        sep_x - 0.05, ax.get_ylim()[1] * 0.95, "Train",
        ha="right", fontsize=7, color="gray", style="italic",
    )
    ax.text(
        sep_x + 0.05, ax.get_ylim()[1] * 0.95, "Held-out",
        ha="left", fontsize=7, color="gray", style="italic",
    )

    ax.set_ylabel("Mean Episode Reward", fontsize=10)
    ax.set_title(
        "Per-Task Baseline Comparison (mean ± std across 5 seeds)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(
        fontsize=8, ncol=n_bl, loc="upper center",
        bbox_to_anchor=(0.5, -0.12), frameon=False,
    )
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    for fmt in ("pdf", "png"):
        path = out_dir / f"per_task_bars.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)
    print("\n* = held-out tasks (never seen during training)")


if __name__ == "__main__":
    main()
