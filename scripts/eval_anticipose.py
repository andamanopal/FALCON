"""
eval_anticipose.py
------------------
Evaluate all AnticiPose baselines and compute paper metrics.

Baselines (ablation ladder)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
    B1   Reactive baseline:      standard FALCON policy (no wrench info)
    B2   Extended History:       10-step obs history (vs 5)
    B3   Current Wrench:         6-dim current wrench in actor obs
    B4a  Direct Plan (Actor):    raw arm plan in actor obs
    B4b  Direct Plan (Critic):   arm plan in critic only
    B5   AnticiPose:             predicted future wrenches (OUR METHOD)

Metrics (per episode)
~~~~~~~~~~~~~~~~~~~~~
    survival_pct      : % of episodes surviving to motion_length - tolerance
    episode_length    : mean episode length (steps)
    com_error         : mean CoM position error vs. commanded height (m)
    base_orient_error : mean base orientation error (rad, from projected gravity)
    vel_tracking      : mean velocity tracking reward (lin + ang)

Usage
~~~~~
    python scripts/eval_anticipose.py \
        --b1_checkpoint  checkpoints/b1.pt \
        --b5_checkpoint  checkpoints/b5.pt \
        --config_path    checkpoints/b1/config.yaml \
        --num_episodes   100 \
        --output_path    results/eval_results.json \
        --device         cuda
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Survival tolerance: an episode is considered "survived" if its length
# reaches within this many steps of the motion/trajectory length.
_SURVIVAL_TOLERANCE_STEPS = 5


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    """Compute ratio, returning None when denominator is near zero."""
    if abs(denominator) < 1e-8:
        return None
    return numerator / denominator


def compute_pairwise_deltas(
    results: Dict[str, Dict[str, float]],
    baseline_key: str,
    metric_keys: List[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Compute (Bx - B1) for each baseline and metric.

    Returns dict[baseline][metric] = delta value.
    """
    if baseline_key not in results:
        return {}
    b1_metrics = results[baseline_key]
    deltas = {}
    for bname, bmetrics in results.items():
        if bname == baseline_key or bname.startswith("_"):
            continue
        deltas[bname] = {}
        for key in metric_keys:
            if key in bmetrics and key in b1_metrics:
                deltas[bname][key] = bmetrics[key] - b1_metrics[key]
    return deltas


# ---------------------------------------------------------------------------
# Single-policy evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    algo,
    env,
    num_episodes: int,
    device,
    task_label: str = "default",
) -> Dict[str, float]:
    """Roll out `algo` for `num_episodes` complete episodes and collect metrics."""
    import torch

    keys = algo.keys
    num_envs = env.num_envs

    ep_survival = []
    ep_lengths = []
    ep_com_errors = []
    ep_orient_errors = []
    ep_vel_tracking = []

    ep_com_sum = torch.zeros(num_envs, device=device)
    ep_orient_sum = torch.zeros(num_envs, device=device)
    ep_vel_sum = torch.zeros(num_envs, device=device)
    ep_len = torch.zeros(num_envs, dtype=torch.long, device=device)

    # Determine the motion/trajectory length for survival metric.
    # Use motion_length if available (from arm trajectory), else max_episode_length.
    motion_length = getattr(env, "max_episode_length", 500)
    if hasattr(env, "_arm_traj_gen") and hasattr(env._arm_traj_gen, "motion_length_steps"):
        motion_length = env._arm_traj_gen.motion_length_steps
    survival_threshold = max(1, motion_length - _SURVIVAL_TOLERANCE_STEPS)

    completed_episodes = 0
    log.info(
        f"  [{task_label}] Starting rollout: {num_episodes} episodes, "
        f"{num_envs} parallel envs, survival_threshold={survival_threshold}"
    )

    for actor in algo.actors.values():
        actor.eval()
        for p in actor.parameters():
            p.requires_grad_(False)

    obs_dict = env.reset_all()
    for k in obs_dict:
        obs_dict[k] = obs_dict[k].to(device)

    with torch.inference_mode():
        while completed_episodes < num_episodes:
            actor_obs = obs_dict["actor_obs"]

            actions_dict = {}
            for key in keys:
                actions_dict[key] = algo.actors[key].act_inference(actor_obs)
            combined_actions = torch.cat(
                [actions_dict[k] for k in keys], dim=1,
            )

            actor_state = {"actions": combined_actions}
            obs_dict, rewards, dones, infos = env.step(actor_state)
            for k in obs_dict:
                obs_dict[k] = obs_dict[k].to(device)

            ep_len += 1

            # ---- Per-step metric accumulation ----
            if hasattr(env, "base_pos") and hasattr(env, "commands"):
                commanded_height = (
                    env.commands[:, 4]
                    if env.commands.shape[1] > 4
                    else torch.zeros(num_envs, device=device)
                )
                actual_height = env.base_pos[:, 2]
                com_err = (actual_height - (0.8 + commanded_height)).abs()
                ep_com_sum += com_err

            if hasattr(env, "projected_gravity"):
                g_body = env.projected_gravity
                g_ref = torch.tensor([0.0, 0.0, -1.0], device=device)
                cos_angle = (g_body * g_ref).sum(dim=-1).clamp(-1.0, 1.0)
                orient_err = torch.acos(cos_angle)
                ep_orient_sum += orient_err

            if "to_log" in infos and "tracking_lin_vel" in infos["to_log"]:
                vel_t = infos["to_log"]["tracking_lin_vel"].to(device)
                ep_vel_sum += vel_t
            else:
                total_rew = (
                    sum(rewards.values())
                    if isinstance(rewards, dict)
                    else rewards
                )
                ep_vel_sum += total_rew.to(device)

            # ---- Episode completion detection ----
            done_envs = dones.bool().to(device)
            done_indices = done_envs.nonzero(as_tuple=False).squeeze(-1)

            if done_indices.numel() > 0:
                for idx in done_indices:
                    i = idx.item()
                    length = ep_len[i].item()
                    if length == 0:
                        continue

                    # Survival: episode reached motion length (within tolerance)
                    survived = length >= survival_threshold
                    ep_survival.append(float(survived))
                    ep_lengths.append(float(length))
                    ep_com_errors.append((ep_com_sum[i] / length).item())
                    ep_orient_errors.append((ep_orient_sum[i] / length).item())
                    ep_vel_tracking.append((ep_vel_sum[i] / length).item())

                    completed_episodes += 1

                ep_com_sum[done_indices] = 0.0
                ep_orient_sum[done_indices] = 0.0
                ep_vel_sum[done_indices] = 0.0
                ep_len[done_indices] = 0

                if completed_episodes % max(1, num_episodes // 10) == 0:
                    survival_so_far = (
                        100.0 * sum(ep_survival) / len(ep_survival)
                    )
                    log.info(
                        f"  [{task_label}] {completed_episodes}/{num_episodes} eps  "
                        f"survival={survival_so_far:.1f}%  "
                        f"mean_len={sum(ep_lengths)/len(ep_lengths):.1f}"
                    )

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "survival_pct": 100.0 * _mean(ep_survival),
        "episode_length": _mean(ep_lengths),
        "com_error": _mean(ep_com_errors),
        "base_orient_error": _mean(ep_orient_errors),
        "vel_tracking": _mean(ep_vel_tracking),
        "n_episodes": len(ep_survival),
    }


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_policy(checkpoint_path: str, env, device, config):
    """Instantiate and load a PPOMultiActorCritic from a checkpoint."""
    from omegaconf import OmegaConf
    from humanoidverse.utils.helpers import pre_process_config
    from hydra.utils import instantiate

    ckpt = Path(checkpoint_path)
    config_path = ckpt.parent / "config.yaml"
    if not config_path.exists():
        config_path = ckpt.parent.parent / "config.yaml"

    if config_path.exists():
        log.info(f"  Loading policy config from {config_path}")
        with open(config_path) as f:
            policy_config = OmegaConf.load(f)
        merged_config = OmegaConf.merge(policy_config, config)
    else:
        log.warning(
            f"  No config.yaml found near {checkpoint_path}; "
            "using global config (may fail if obs dims differ)."
        )
        merged_config = config

    pre_process_config(merged_config)

    algo = instantiate(
        device=device, env=env, config=merged_config.algo, log_dir=None,
    )
    algo.setup()
    algo.load(checkpoint_path)
    return algo


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "survival_pct": "Survival (%)",
    "episode_length": "Ep. Length (steps)",
    "com_error": "CoM Error (m)",
    "base_orient_error": "Orient. Error (rad)",
    "vel_tracking": "Vel. Tracking",
}
METRIC_HIGHER_IS_BETTER = {
    "survival_pct": True,
    "episode_length": True,
    "com_error": False,
    "base_orient_error": False,
    "vel_tracking": True,
}

# Ordered baseline names for display
BASELINE_ORDER = ["B1", "B2", "B3", "B4a", "B4b", "B5"]


def print_results_table(
    results: Dict[str, Dict[str, float]],
    task_label: str = "default",
):
    """Print a rich comparison table to stdout."""
    col_width = 18
    name_width = 22
    baselines = [b for b in BASELINE_ORDER if b in results]

    sep = "-" * (name_width + col_width * len(baselines) + 2)
    log.info("")
    log.info(f"{'=' * len(sep)}")
    log.info(f"  Results: task={task_label}")
    log.info(sep)

    header = f"{'Metric':<{name_width}}"
    for b in baselines:
        header += f"{b:>{col_width}}"
    log.info(header)
    log.info(sep)

    for mkey, mlabel in METRIC_LABELS.items():
        row = f"{mlabel:<{name_width}}"
        for b in baselines:
            val = results[b].get(mkey, float("nan"))
            row += f"{val:>{col_width}.4f}"
        log.info(row)

    log.info(sep)

    # Delta rows (B5 - B1)
    if "B5" in results and "B1" in results:
        row = f"{'Delta (B5 - B1)':<{name_width}}"
        for mkey in METRIC_LABELS:
            delta = results["B5"].get(mkey, 0) - results["B1"].get(mkey, 0)
            row += f"{delta:>{col_width}.4f}"
        log.info(row)

    log.info(sep)
    log.info(
        f"  n_episodes: "
        + "  ".join(
            f"{b}={results[b].get('n_episodes', 0)}" for b in baselines
        )
    )
    log.info(sep)


# ---------------------------------------------------------------------------
# Main evaluation driver
# ---------------------------------------------------------------------------

def run_evaluation(args):
    try:
        import isaacgym  # noqa: F401
    except ImportError:
        log.warning(
            "isaacgym not importable; proceeding (may fail at env creation)."
        )

    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from humanoidverse.utils.helpers import pre_process_config

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; falling back to CPU.")
    log.info(f"Device: {device}")

    if not Path(args.config_path).exists():
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    log.info(f"Loading base config from: {args.config_path}")
    with open(args.config_path) as f:
        base_config = OmegaConf.load(f)

    eval_overrides = OmegaConf.create({
        "headless": True,
        "num_envs": args.num_envs_eval,
    })
    config = OmegaConf.merge(base_config, eval_overrides)
    pre_process_config(config)

    # Collect baseline -> checkpoint path mapping
    baseline_ckpts: Dict[str, Optional[str]] = {
        "B1": args.b1_checkpoint,
        "B2": args.b2_checkpoint,
        "B3": args.b3_checkpoint,
        "B4a": args.b4a_checkpoint,
        "B4b": args.b4b_checkpoint,
        "B5": args.b5_checkpoint,
    }
    active_baselines = {k: v for k, v in baseline_ckpts.items() if v is not None}
    log.info(f"Active baselines: {list(active_baselines.keys())}")

    if not active_baselines:
        raise ValueError(
            "No checkpoint paths provided. Specify at least --b1_checkpoint."
        )

    # Build task list
    tasks = [{"label": "default", "overrides": {}}]
    if args.generalization_tasks:
        for task_name in args.generalization_tasks:
            task_overrides = {}
            if args.task_config_overrides:
                for item in args.task_config_overrides:
                    key, val = item.split("=", 1)
                    task_overrides[key] = val
            tasks.append({"label": task_name, "overrides": task_overrides})

    log.info(f"Tasks to evaluate: {[t['label'] for t in tasks]}")

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for task in tasks:
        task_label = task["label"]
        task_override = task["overrides"]
        log.info(f"\n{'=' * 60}")
        log.info(f"  Evaluating task: {task_label}")
        log.info(f"{'=' * 60}")

        if task_override:
            task_config = OmegaConf.merge(
                config, OmegaConf.create(task_override),
            )
        else:
            task_config = config

        task_results: Dict[str, Dict[str, float]] = {}

        for baseline_name, ckpt_path in active_baselines.items():
            log.info(
                f"\n--- Evaluating {baseline_name} on task '{task_label}' ---"
            )
            log.info(f"    checkpoint: {ckpt_path}")

            try:
                env = instantiate(
                    config=task_config.env, device=str(device),
                )
                algo = load_policy(
                    ckpt_path, env, str(device), task_config,
                )

                t0 = time.time()
                metrics = evaluate_policy(
                    algo=algo,
                    env=env,
                    num_episodes=args.num_episodes,
                    device=device,
                    task_label=f"{task_label}/{baseline_name}",
                )
                elapsed = time.time() - t0

                log.info(
                    f"  {baseline_name} [{task_label}] done in {elapsed:.1f}s: "
                    f"survival={metrics['survival_pct']:.1f}%  "
                    f"ep_len={metrics['episode_length']:.1f}  "
                    f"com_err={metrics['com_error']:.4f}  "
                    f"orient_err={metrics['base_orient_error']:.4f}  "
                    f"vel_track={metrics['vel_tracking']:.4f}"
                )
                task_results[baseline_name] = metrics

            except Exception as exc:
                log.error(
                    f"  FAILED to evaluate {baseline_name} on '{task_label}': {exc}",
                    exc_info=True,
                )
                task_results[baseline_name] = {
                    "survival_pct": float("nan"),
                    "episode_length": float("nan"),
                    "com_error": float("nan"),
                    "base_orient_error": float("nan"),
                    "vel_tracking": float("nan"),
                    "n_episodes": 0,
                    "error": str(exc),
                }

        all_results[task_label] = task_results

        # Pairwise deltas from B1
        deltas = compute_pairwise_deltas(
            task_results, "B1", list(METRIC_LABELS.keys()),
        )
        all_results[task_label]["_deltas_from_B1"] = deltas

        print_results_table(task_results, task_label=task_label)

    # Save results to JSON
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _to_serializable(obj):
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, float):
            return None if (obj != obj) else obj
        if obj is None:
            return None
        return obj

    with open(output_path, "w") as f:
        json.dump(_to_serializable(all_results), f, indent=2)

    log.info(f"\nResults saved to: {output_path}")

    # Final summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY (all tasks)")
    log.info("=" * 60)
    for task_label, task_results in all_results.items():
        log.info(f"  Task: {task_label}")
        for baseline_name in BASELINE_ORDER:
            if baseline_name not in task_results:
                continue
            metrics = task_results[baseline_name]
            if not isinstance(metrics, dict):
                continue
            log.info(
                f"    {baseline_name:<6}  "
                f"survival={metrics.get('survival_pct', float('nan')):>6.1f}%  "
                f"ep_len={metrics.get('episode_length', float('nan')):>7.1f}  "
                f"com_err={metrics.get('com_error', float('nan')):>8.4f}  "
                f"orient_err={metrics.get('base_orient_error', float('nan')):>8.4f}"
            )
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate AnticiPose baselines (ablation ladder).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--b1_checkpoint", type=str, default=None,
        help="Path to B1 (reactive baseline) checkpoint.",
    )
    parser.add_argument(
        "--b2_checkpoint", type=str, default=None,
        help="Path to B2 (extended history) checkpoint.",
    )
    parser.add_argument(
        "--b3_checkpoint", type=str, default=None,
        help="Path to B3 (current wrench) checkpoint.",
    )
    parser.add_argument(
        "--b4a_checkpoint", type=str, default=None,
        help="Path to B4a (direct plan, actor) checkpoint.",
    )
    parser.add_argument(
        "--b4b_checkpoint", type=str, default=None,
        help="Path to B4b (direct plan, critic) checkpoint.",
    )
    parser.add_argument(
        "--b5_checkpoint", type=str, default=None,
        help="Path to B5 (AnticiPose) checkpoint.",
    )
    parser.add_argument(
        "--config_path", type=str, required=True,
        help="Path to a training config.yaml for environment creation.",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=100,
        help="Number of complete episodes per baseline per task.",
    )
    parser.add_argument(
        "--num_envs_eval", type=int, default=16,
        help="Number of parallel environments during evaluation.",
    )
    parser.add_argument(
        "--output_path", type=str, default="results/eval_results.json",
        help="Path to write the JSON results file.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device.",
    )
    parser.add_argument(
        "--generalization_tasks", type=str, nargs="*", default=[],
        help="Names of held-out generalization tasks.",
    )
    parser.add_argument(
        "--task_config_overrides", type=str, nargs="*", default=[],
        help="key=value config overrides for generalization tasks.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
