"""
Generate Figure 2: Main results box plot comparing all baselines across seeds.

Reads existing aggregate eval JSONs (train + held-out) and produces a
side-by-side box plot with individual seed points overlaid.

Usage:
    cd FALCON/
    python scripts/plot_main_results_box.py --output_dir paper/figures

    # Custom eval directories:
    python scripts/plot_main_results_box.py \
        --eval_dirs logs_eval_s42_rsync,logs_eval_s789_rsync,logs_eval \
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
BASELINE_LABELS = {
    "B1": "B1\nReactive",
    "B2": "B2\nExt. History",
    "B3": "B3\nWrench FB",
    "B4a": "B4a\nPlan (Actor)",
    "B4b": "B4b\nPlan (Critic)",
    "B5": "B5\nPred. Wrench",
    "B6": "B6\nCVAE Latent",
}
SEEDS = [42, 123, 456, 789, 35]

# B4b (hero) in bold blue; B5 (catastrophic) in red; others muted
COLORS = {
    "B1": "#7f7f7f",
    "B2": "#9467bd",
    "B3": "#2ca02c",
    "B4a": "#ff7f0e",
    "B4b": "#0050A0",
    "B5": "#d62728",
    "B6": "#8c564b",
}


def find_result(eval_dirs, baseline, task_type, seed):
    """Search eval directories for a result JSON."""
    pattern = BASELINE_PATTERNS[baseline]
    name = f"eval_{pattern}_{task_type}_s{seed}"
    for d in eval_dirs:
        path = Path(d) / name / "results.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


def collect_metric(eval_dirs, task_type, metric="mean_reward"):
    """Collect metric values per baseline per seed."""
    data = {}
    for bl in BASELINES:
        vals = []
        for seed in SEEDS:
            r = find_result(eval_dirs, bl, task_type, seed)
            if r is not None:
                vals.append(r[metric])
        data[bl] = vals
    return data


def make_box_panel(ax, data, title, highlight="B4b"):
    """Draw a box plot panel."""
    positions = list(range(len(BASELINES)))
    box_data = [data.get(bl, []) for bl in BASELINES]

    bp = ax.boxplot(
        box_data, positions=positions, widths=0.5,
        patch_artist=True, showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
    )

    for i, bl in enumerate(BASELINES):
        color = COLORS[bl]
        bp["boxes"][i].set_facecolor(color)
        bp["boxes"][i].set_alpha(0.4)
        bp["boxes"][i].set_edgecolor(color)
        if bl == highlight:
            bp["boxes"][i].set_alpha(0.7)
            bp["boxes"][i].set_linewidth(2)

    # Overlay individual seed points
    for i, bl in enumerate(BASELINES):
        vals = data.get(bl, [])
        if vals:
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
            ax.scatter(
                [i + j for j in jitter], vals,
                color=COLORS[bl], s=30, zorder=5,
                edgecolors="white", linewidths=0.5,
            )

    # B1 reference line
    b1_vals = data.get("B1", [])
    if b1_vals:
        ax.axhline(
            np.mean(b1_vals), color=COLORS["B1"],
            linestyle="--", linewidth=1, alpha=0.7, label="B1 mean",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([BASELINE_LABELS[bl] for bl in BASELINES], fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Episode Reward", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_dirs",
        default="../logs_eval_s42_rsync,../logs_eval_s789_rsync,logs_eval",
        help="Comma-separated eval directories to search (relative to CWD)",
    )
    parser.add_argument("--output_dir", default="paper/figures")
    parser.add_argument("--metric", default="mean_reward")
    args = parser.parse_args()

    eval_dirs = [d.strip() for d in args.eval_dirs.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = collect_metric(eval_dirs, "train", args.metric)
    heldout_data = collect_metric(eval_dirs, "heldout", args.metric)

    # Report data availability
    print("\nData availability:")
    for bl in BASELINES:
        n_train = len(train_data.get(bl, []))
        n_held = len(heldout_data.get(bl, []))
        print(f"  {bl}: train={n_train}/5 seeds, held-out={n_held}/5 seeds")

    # Create figure
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    make_box_panel(ax1, train_data, "Training Tasks (random)")
    make_box_panel(ax2, heldout_data, "Held-out Task (lateral_slam_down)")

    fig.suptitle(
        "Held-out and Training Performance Across 5 Seeds",
        fontsize=12, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    for fmt in ("pdf", "png"):
        path = out_dir / f"main_results_box.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
