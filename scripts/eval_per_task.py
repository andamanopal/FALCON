"""
Orchestrate per-task evaluation across all baselines x tasks x seeds.

Usage:
    cd FALCON/
    python scripts/eval_per_task.py \
        --baselines B1,B2,B3,B4a,B4b,B4c,B5,B6 \
        --tasks frontal_reach_lift,lateral_shelf_pick,forward_push,lateral_slam_down,cross_body_reach,bilateral_asymmetric_lift,overhead_reach,backward_swing \
        --seeds 42,123,456,789,35 \
        --num_episodes 500 \
        --log_dir logs/anticipose_overnight \
        --output_dir logs_eval
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

BASELINE_PATTERNS = {
    "B1": "B1_reactive",
    "B2": "B2_extended_history",
    "B3": "B3_current_wrench",
    "B4a": "B4a_direct_plan",
    "B4b": "B4b_direct_plan_critic",
    "B4c": "B4c_direct_plan_both",
    "B5": "B5_anticipose",
    "B6": "B6_cvae",
}

BASELINE_MODES = {
    "B5": "anticipose",
    "B6": "cvae",
}


def find_checkpoint(log_dir, pattern, seed, num_iters):
    """Find the latest checkpoint matching baseline pattern and seed."""
    search = sorted(
        Path(log_dir).glob(f"*{pattern}_seed{seed}*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for d in search:
        ckpt = d / f"model_{num_iters}.pt"
        if ckpt.exists():
            return ckpt
    return None


def find_predictor_ckpt(log_dir, seed):
    """Find wrench predictor checkpoint for B5."""
    path = Path(log_dir) / f"wrench_predictor_seed{seed}.pt"
    return path if path.exists() else None


def find_cvae_ckpt(log_dir, seed):
    """Find CVAE checkpoint for B6."""
    path = Path(log_dir) / f"arm_plan_cvae_seed{seed}.pt"
    return path if path.exists() else None


def build_eval_name(baseline, task, seed):
    """Build evaluation directory name."""
    pattern = BASELINE_PATTERNS[baseline]
    return f"eval_{pattern}_{task}_s{seed}"


def run_eval(checkpoint, eval_name, task, num_episodes, num_envs,
             max_ep_len_s, output_dir, wrench_ckpt=None, cvae_ckpt=None):
    """Run eval_baselines.py as subprocess."""
    cmd = [
        sys.executable, "scripts/eval_baselines.py",
        "--checkpoint", str(checkpoint),
        "--eval_name", eval_name,
        "--num_episodes", str(num_episodes),
        "--num_envs", str(num_envs),
        "--max_episode_length_s", str(max_ep_len_s),
        "--arm_trajectory_task", task,
        "--output_dir", str(output_dir),
    ]
    if wrench_ckpt is not None:
        cmd.extend(["--wrench_predictor_ckpt", str(wrench_ckpt)])
    if cvae_ckpt is not None:
        cmd.extend(["--cvae_ckpt", str(cvae_ckpt)])

    print(f"\n{'='*70}")
    print(f"  Running: {eval_name}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Task: {task}")
    print(f"{'='*70}")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Per-task evaluation orchestrator")
    parser.add_argument(
        "--baselines", default="B1,B2,B3,B4a,B4b,B4c,B5,B6",
        help="Comma-separated baseline IDs",
    )
    parser.add_argument(
        "--tasks",
        default="frontal_reach_lift,lateral_shelf_pick,forward_push,"
                "lateral_slam_down,cross_body_reach,"
                "bilateral_asymmetric_lift,overhead_reach,backward_swing",
        help="Comma-separated task names",
    )
    parser.add_argument(
        "--seeds", default="42,123,456,789,35",
        help="Comma-separated seeds",
    )
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--max_episode_length_s", type=float, default=20.0)
    parser.add_argument("--num_iters", type=int, default=3000)
    parser.add_argument("--log_dir", default="logs/anticipose_overnight")
    parser.add_argument("--output_dir", default="logs_eval")
    args = parser.parse_args()

    baselines = [b.strip() for b in args.baselines.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    total = len(baselines) * len(tasks) * len(seeds)
    completed = 0
    skipped = 0
    failed = 0

    print(f"\nPer-task evaluation: {len(baselines)} baselines x "
          f"{len(tasks)} tasks x {len(seeds)} seeds = {total} runs\n")

    for seed in seeds:
        pred_ckpt = find_predictor_ckpt(args.log_dir, seed)
        cvae_ckpt = find_cvae_ckpt(args.log_dir, seed)

        for baseline in baselines:
            pattern = BASELINE_PATTERNS.get(baseline)
            if pattern is None:
                print(f"[WARN] Unknown baseline: {baseline}")
                failed += len(tasks)
                continue

            ckpt = find_checkpoint(
                args.log_dir, pattern, seed, args.num_iters,
            )
            if ckpt is None:
                print(f"[SKIP] No checkpoint for {baseline} seed={seed}")
                skipped += len(tasks)
                continue

            wrench = pred_ckpt if baseline == "B5" else None
            cvae = cvae_ckpt if baseline == "B6" else None

            if baseline == "B5" and wrench is None:
                print(f"[SKIP] No predictor ckpt for B5 seed={seed}")
                skipped += len(tasks)
                continue
            if baseline == "B6" and cvae is None:
                print(f"[SKIP] No CVAE ckpt for B6 seed={seed}")
                skipped += len(tasks)
                continue

            for task in tasks:
                eval_name = build_eval_name(baseline, task, seed)
                result_file = (
                    Path(args.output_dir) / eval_name / "results.json"
                )

                if result_file.exists():
                    print(f"[EXISTS] {eval_name} — skipping")
                    skipped += 1
                    continue

                ok = run_eval(
                    checkpoint=ckpt,
                    eval_name=eval_name,
                    task=task,
                    num_episodes=args.num_episodes,
                    num_envs=args.num_envs,
                    max_ep_len_s=args.max_episode_length_s,
                    output_dir=args.output_dir,
                    wrench_ckpt=wrench,
                    cvae_ckpt=cvae,
                )
                if ok:
                    completed += 1
                else:
                    failed += 1
                    print(f"[FAIL] {eval_name}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  Per-Task Evaluation Summary")
    print(f"  Completed: {completed}  Skipped: {skipped}  Failed: {failed}")
    print(f"{'='*70}\n")

    # Print results table from available JSONs
    print(f"{'Baseline':<12} {'Task':<24} {'Seed':<6} {'Reward':>10} "
          f"{'EpLen':>8} {'Surv%':>7} {'Ori°':>7} {'VelErr':>8}")
    print("-" * 84)

    for seed in seeds:
        for baseline in baselines:
            for task in tasks:
                eval_name = build_eval_name(baseline, task, seed)
                result_file = (
                    Path(args.output_dir) / eval_name / "results.json"
                )
                if result_file.exists():
                    with open(result_file) as f:
                        d = json.load(f)
                    ori = d.get('mean_orientation_rms_deg', -1)
                    vel = d.get('mean_vel_tracking_err_rms', -1)
                    surv = d['survival_rate'] * 100

                    print(
                        f"{baseline:<12} {task:<24} {seed:<6} "
                        f"{d['mean_reward']:>10.2f} "
                        f"{d['mean_episode_length']:>8.1f} "
                        f"{surv:>6.1f}% "
                        f"{ori:>7.2f} "
                        f"{vel:>8.3f}"
                    )


if __name__ == "__main__":
    main()
