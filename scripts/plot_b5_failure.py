"""
Generate B5 failure analysis figure: predictor R^2 vs policy held-out reward.

Shows that better wrench prediction does NOT lead to better policy performance,
proving the distribution shift problem is fundamental.

Usage:
    python scripts/plot_b5_failure.py --output_dir ../paper/figures
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Data from experiment logs (5 B5 predictor variants)
VARIANTS = {
    "v1-500K": {"r2": 0.69, "heldout": 24.55, "train": 35.0},
    "v1-2M":   {"r2": 0.77, "heldout": 55.96, "train": 80.0},
    "B5c-v1":  {"r2": 0.80, "heldout": 30.0,  "train": 75.0},
    "v2-h1":   {"r2": 0.82, "heldout": -5.0,  "train": 60.0},
    "v2-full": {"r2": 0.83, "heldout": -11.19, "train": 50.0},
}

# Reference baselines (5-seed means)
BASELINES = {
    "B1 (Reactive)": 57.57,
    "B4b (Priv. Critic)": 86.59,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="../paper/figures")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.8))

    # Plot B5 variants
    r2_vals = [v["r2"] for v in VARIANTS.values()]
    heldout_vals = [v["heldout"] for v in VARIANTS.values()]
    names = list(VARIANTS.keys())

    ax.scatter(r2_vals, heldout_vals, c="#d62728", s=60, zorder=5,
               edgecolors="white", linewidths=0.8, label="B5 variants")

    # Label each point
    offsets = {
        "v1-500K": (-0.01, 5),
        "v1-2M":   (0.01, 4),
        "B5c-v1":  (0.01, 4),
        "v2-h1":   (-0.04, -10),
        "v2-full": (0.01, -10),
    }
    for name, r2, ho in zip(names, r2_vals, heldout_vals):
        dx, dy = offsets[name]
        ax.annotate(name, (r2 + dx, ho + dy), fontsize=6.5,
                    ha="center", color="#555555")

    # Reference lines
    ax.axhline(BASELINES["B1 (Reactive)"], color="#7f7f7f", linestyle="--",
               linewidth=1, alpha=0.7, label="B1 (Reactive)")
    ax.axhline(BASELINES["B4b (Priv. Critic)"], color="#0050A0", linestyle="--",
               linewidth=1, alpha=0.7, label="B4b (Priv. Critic)")

    # Trend arrow showing "better R^2 -> worse policy"
    ax.annotate("", xy=(0.84, -15), xytext=(0.68, 50),
                arrowprops={"arrowstyle": "->", "color": "#d62728",
                            "lw": 1.5, "ls": "--"})
    ax.text(0.76, 15, "Better $R^2$\n$\\neq$ better policy",
            fontsize=7, ha="center", color="#d62728", style="italic")

    ax.set_xlabel("Wrench Predictor $R^2$", fontsize=9)
    ax.set_ylabel("Held-out Task Reward", fontsize=9)
    ax.set_xlim(0.65, 0.87)
    ax.set_ylim(-25, 100)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.2)

    fig.tight_layout()

    for fmt in ("pdf", "png"):
        path = out_dir / f"b5_failure_scatter.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
