"""
eval_anticipose.py
------------------
Evaluate all AnticiPose baselines and compute paper metrics.

Baselines
~~~~~~~~~
    B1  Reactive baseline: standard FALCON policy (no wrench info)
    B2  Oracle baseline:   policy conditioned on GT wrench (upper bound)
    B4a Raw-plan baseline: policy conditioned on raw arm plan
    B5  AnticiPose:        policy conditioned on predicted wrench

Metrics (per episode)
~~~~~~~~~~~~~~~~~~~~~
    survival_pct      : % of episodes that reach max_episode_steps without fall
    episode_length    : mean episode length (steps)
    com_error         : mean CoM position error vs. commanded height (m)
    base_orient_error : mean base orientation error (rad, from projected gravity)
    vel_tracking      : mean velocity tracking reward (lin + ang)

Derived metric
~~~~~~~~~~~~~~
    recovery_ratio    : (B5 - B1) / (B2 - B1)  per metric
                        measures how much of the oracle gap AnticiPose closes.

Usage
~~~~~
    python scripts/eval_anticipose.py \
        --b1_checkpoint  checkpoints/b1.pt \
        --b2_checkpoint  checkpoints/b2.pt \
        --b4a_checkpoint checkpoints/b4a.pt \
        --b5_checkpoint  checkpoints/b5.pt \
        --config_path    checkpoints/b1/config.yaml \
        --num_episodes   100 \
        --output_path    results/eval_results.json \
        --device         cuda

Generalization evaluation on held-out tasks:
    python scripts/eval_anticipose.py \
        --b1_checkpoint  checkpoints/b1.pt \
        ... \
        --generalization_tasks task_heavy_load task_fast_motion \
        --task_config_overrides force_range_high=60.0 max_episode_length_s=30

The script must be run from the FALCON/humanoidverse/ directory so that
Hydra can locate the config files (identical layout to eval_agent.py).
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging setup (before any heavy imports so we see bootstrap messages)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    """Compute ratio, returning None when denominator is near zero."""
    if abs(denominator) < 1e-8:
        return None
    return numerator / denominator


def compute_recovery_ratio(
    b5_metrics: Dict[str, float],
    b1_metrics: Dict[str, float],
    b2_metrics: Dict[str, float],
    metric_keys: List[str],
) -> Dict[str, Optional[float]]:
    """Compute (B5 - B1) / (B2 - B1) for each metric.

    A value of 1.0 means B5 fully closes the gap to oracle B2.
    A value of 0.0 means no improvement over reactive B1.
    Negative values indicate B5 is worse than B1 on that metric.
    """
    ratios = {}
    for key in metric_keys:
        gap   = b2_metrics[key] - b1_metrics[key]
        delta = b5_metrics[key] - b1_metrics[key]
        ratios[key] = _safe_ratio(delta, gap)
    return ratios


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
    """Roll out `algo` for `num_episodes` complete episodes and collect metrics.

    Args:
        algo        : instantiated PPOMultiActorCritic (already loaded checkpoint)
        env         : instantiated BaseTask environment
        num_episodes: number of complete episodes to collect
        device      : torch device
        task_label  : string label for logging

    Returns:
        dict with scalar metric values averaged over completed episodes.
    """
    import torch

    keys          = algo.keys
    num_envs      = env.num_envs
    num_act_lower = algo.num_act_lower_body
    num_act_upper = algo.num_act_upper_body
    total_actions = num_act_lower + num_act_upper

    # Per-episode accumulators (one slot per env)
    ep_survival       = []  # bool per completed episode
    ep_lengths        = []
    ep_com_errors     = []
    ep_orient_errors  = []
    ep_vel_tracking   = []

    # Running accumulators per env
    ep_com_sum    = torch.zeros(num_envs, device=device)
    ep_orient_sum = torch.zeros(num_envs, device=device)
    ep_vel_sum    = torch.zeros(num_envs, device=device)
    ep_len        = torch.zeros(num_envs, dtype=torch.long, device=device)

    completed_episodes = 0
    log.info(
        f"  [{task_label}] Starting rollout: {num_episodes} episodes, "
        f"{num_envs} parallel envs"
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

            # Build combined action
            actions_dict = {}
            for key in keys:
                actions_dict[key] = algo.actors[key].act_inference(actor_obs)
            combined_actions = torch.cat(
                [actions_dict[k] for k in keys], dim=1
            )

            actor_state = {"actions": combined_actions}
            obs_dict, rewards, dones, infos = env.step(actor_state)
            for k in obs_dict:
                obs_dict[k] = obs_dict[k].to(device)

            ep_len += 1

            # ---- Per-step metric accumulation ----
            # CoM / base height error
            if hasattr(env, "base_pos") and hasattr(env, "commands"):
                # commanded height is stored as commands[:, height_cmd_idx]
                # Use projected_gravity norm as orientation proxy
                commanded_height = env.commands[:, 4] if env.commands.shape[1] > 4 else torch.zeros(num_envs, device=device)
                actual_height    = env.base_pos[:, 2]
                com_err = (actual_height - (0.8 + commanded_height)).abs()
                ep_com_sum += com_err
            else:
                ep_com_sum += torch.zeros(num_envs, device=device)

            # Base orientation error (angle from upright)
            if hasattr(env, "projected_gravity"):
                # projected_gravity in body frame; upright = (0, 0, -1)
                g_body      = env.projected_gravity  # (num_envs, 3)
                g_ref       = torch.tensor([0.0, 0.0, -1.0], device=device)
                cos_angle   = (g_body * g_ref).sum(dim=-1).clamp(-1.0, 1.0)
                orient_err  = torch.acos(cos_angle)  # (num_envs,)
                ep_orient_sum += orient_err
            else:
                ep_orient_sum += torch.zeros(num_envs, device=device)

            # Velocity tracking (from to_log if available, else from rewards)
            if "to_log" in infos and "tracking_lin_vel" in infos["to_log"]:
                vel_t = infos["to_log"]["tracking_lin_vel"].to(device)
                ep_vel_sum += vel_t
            else:
                # Approximate from reward signal (sum of decoupled rewards)
                total_rew = sum(rewards.values()) if isinstance(rewards, dict) else rewards
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

                    survived = (
                        length >= env.max_episode_length
                        if hasattr(env, "max_episode_length")
                        else True
                    )
                    ep_survival.append(float(survived))
                    ep_lengths.append(float(length))
                    ep_com_errors.append((ep_com_sum[i] / length).item())
                    ep_orient_errors.append((ep_orient_sum[i] / length).item())
                    ep_vel_tracking.append((ep_vel_sum[i] / length).item())

                    completed_episodes += 1

                # Reset accumulators for finished envs
                ep_com_sum[done_indices]    = 0.0
                ep_orient_sum[done_indices] = 0.0
                ep_vel_sum[done_indices]    = 0.0
                ep_len[done_indices]        = 0

                if completed_episodes % max(1, num_episodes // 10) == 0:
                    survival_so_far = 100.0 * sum(ep_survival) / len(ep_survival)
                    log.info(
                        f"  [{task_label}] {completed_episodes}/{num_episodes} eps  "
                        f"survival={survival_so_far:.1f}%  "
                        f"mean_len={sum(ep_lengths)/len(ep_lengths):.1f}"
                    )

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {
        "survival_pct":       100.0 * _mean(ep_survival),
        "episode_length":     _mean(ep_lengths),
        "com_error":          _mean(ep_com_errors),
        "base_orient_error":  _mean(ep_orient_errors),
        "vel_tracking":       _mean(ep_vel_tracking),
        "n_episodes":         len(ep_survival),
    }


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def load_policy(checkpoint_path: str, env, device, config):
    """Instantiate and load a PPOMultiActorCritic from a checkpoint.

    Follows eval_agent.py: config is loaded from checkpoint.parent/config.yaml,
    merged with the override config, then the algo is instantiated.
    """
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from humanoidverse.utils.helpers import pre_process_config

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
        device=device, env=env, config=merged_config.algo, log_dir=None
    )
    algo.setup()
    algo.load(checkpoint_path)
    return algo


# ---------------------------------------------------------------------------
# Comparison table printing
# ---------------------------------------------------------------------------

METRIC_LABELS = {
    "survival_pct":      "Survival (%)",
    "episode_length":    "Ep. Length (steps)",
    "com_error":         "CoM Error (m)",
    "base_orient_error": "Orient. Error (rad)",
    "vel_tracking":      "Vel. Tracking",
}
METRIC_HIGHER_IS_BETTER = {
    "survival_pct":      True,
    "episode_length":    True,
    "com_error":         False,
    "base_orient_error": False,
    "vel_tracking":      True,
}


def print_results_table(
    results: Dict[str, Dict[str, float]],
    recovery_ratios: Dict[str, Optional[float]],
    task_label: str = "default",
):
    """Print a rich comparison table to stdout."""
    col_width  = 18
    name_width = 22
    baselines  = [b for b in ["B1", "B2", "B4a", "B5"] if b in results]

    sep = "-" * (name_width + col_width * len(baselines) + 2)
    log.info("")
    log.info(f"{'='*len(sep)}")
    log.info(f"  Results: task={task_label}")
    log.info(sep)

    # Header
    header = f"{'Metric':<{name_width}}"
    for b in baselines:
        header += f"{b:>{col_width}}"
    log.info(header)
    log.info(sep)

    # Metric rows
    for mkey, mlabel in METRIC_LABELS.items():
        row = f"{mlabel:<{name_width}}"
        for b in baselines:
            val = results[b].get(mkey, float("nan"))
            row += f"{val:>{col_width}.4f}"
        log.info(row)

    log.info(sep)

    # Recovery ratio row (B5 only)
    if "B5" in results and "B1" in results and "B2" in results:
        log.info(f"{'Recovery Ratio (B5)':>{name_width}}")
        rr_row = f"{'(B5-B1)/(B2-B1)':<{name_width}}"
        for mkey in METRIC_LABELS:
            rr = recovery_ratios.get(mkey)
            if rr is None:
                rr_row += f"{'N/A':>{col_width}}"
            else:
                rr_row += f"{rr:>{col_width}.4f}"
        log.info(rr_row)

    log.info(sep)
    log.info(f"  n_episodes: "
             + "  ".join(
                 f"{b}={results[b].get('n_episodes', 0)}"
                 for b in baselines
             ))
    log.info(sep)


# ---------------------------------------------------------------------------
# Main evaluation driver
# ---------------------------------------------------------------------------

def run_evaluation(args):
    # ------------------------------------------------------------------
    # Simulator imports (IsaacGym must precede torch)
    # ------------------------------------------------------------------
    try:
        import isaacgym  # noqa: F401
    except ImportError:
        log.warning("isaacgym not importable; proceeding (may fail at env creation).")

    import torch
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from humanoidverse.utils.helpers import pre_process_config

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; falling back to CPU.")
    log.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load base training config from the provided config_path
    # ------------------------------------------------------------------
    if not Path(args.config_path).exists():
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    log.info(f"Loading base config from: {args.config_path}")
    with open(args.config_path) as f:
        base_config = OmegaConf.load(f)

    # Apply eval-mode overrides (headless, reduced num_envs for speed)
    eval_overrides = OmegaConf.create({
        "headless": True,
        "num_envs":  args.num_envs_eval,
    })
    config = OmegaConf.merge(base_config, eval_overrides)
    pre_process_config(config)

    # ------------------------------------------------------------------
    # Collect baseline → checkpoint path mapping
    # ------------------------------------------------------------------
    baseline_ckpts: Dict[str, Optional[str]] = {
        "B1":  args.b1_checkpoint,
        "B2":  args.b2_checkpoint,
        "B4a": args.b4a_checkpoint,
        "B5":  args.b5_checkpoint,
    }
    # Filter out baselines not provided
    active_baselines = {k: v for k, v in baseline_ckpts.items() if v is not None}
    log.info(f"Active baselines: {list(active_baselines.keys())}")

    if not active_baselines:
        raise ValueError(
            "No checkpoint paths provided. "
            "Specify at least --b1_checkpoint."
        )

    # ------------------------------------------------------------------
    # Build task list: default task + generalization tasks
    # ------------------------------------------------------------------
    tasks = [{"label": "default", "overrides": {}}]
    if args.generalization_tasks:
        for task_name in args.generalization_tasks:
            task_overrides = {}
            # Apply any paired task_config_overrides
            if args.task_config_overrides:
                for item in args.task_config_overrides:
                    key, val = item.split("=", 1)
                    task_overrides[key] = val
            tasks.append({"label": task_name, "overrides": task_overrides})

    log.info(f"Tasks to evaluate: {[t['label'] for t in tasks]}")

    # ------------------------------------------------------------------
    # Outer results dict: results[task_label][baseline] = metric_dict
    # ------------------------------------------------------------------
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for task in tasks:
        task_label    = task["label"]
        task_override = task["overrides"]
        log.info(f"\n{'='*60}")
        log.info(f"  Evaluating task: {task_label}")
        log.info(f"{'='*60}")

        # Apply task-specific overrides to config
        if task_override:
            task_config = OmegaConf.merge(
                config, OmegaConf.create(task_override)
            )
        else:
            task_config = config

        task_results: Dict[str, Dict[str, float]] = {}

        for baseline_name, ckpt_path in active_baselines.items():
            log.info(f"\n--- Evaluating {baseline_name} on task '{task_label}' ---")
            log.info(f"    checkpoint: {ckpt_path}")

            try:
                # Instantiate a fresh environment for each baseline to avoid
                # state contamination.
                env = instantiate(config=task_config.env, device=str(device))

                algo = load_policy(ckpt_path, env, str(device), task_config)

                t0      = time.time()
                metrics = evaluate_policy(
                    algo        = algo,
                    env         = env,
                    num_episodes= args.num_episodes,
                    device      = device,
                    task_label  = f"{task_label}/{baseline_name}",
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
                    "survival_pct":      float("nan"),
                    "episode_length":    float("nan"),
                    "com_error":         float("nan"),
                    "base_orient_error": float("nan"),
                    "vel_tracking":      float("nan"),
                    "n_episodes":        0,
                    "error":             str(exc),
                }

        all_results[task_label] = task_results

        # ---- Recovery ratios ----
        recovery_ratios: Dict[str, Optional[float]] = {}
        if "B1" in task_results and "B2" in task_results and "B5" in task_results:
            recovery_ratios = compute_recovery_ratio(
                b5_metrics = task_results["B5"],
                b1_metrics = task_results["B1"],
                b2_metrics = task_results["B2"],
                metric_keys = list(METRIC_LABELS.keys()),
            )

        print_results_table(task_results, recovery_ratios, task_label=task_label)
        all_results[task_label]["_recovery_ratios"] = {
            k: v for k, v in recovery_ratios.items()
        }

    # ------------------------------------------------------------------
    # Save results to JSON
    # ------------------------------------------------------------------
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _to_serializable(obj):
        """Recursively make all values JSON-serializable."""
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, float):
            return None if (obj != obj) else obj   # NaN -> None
        if obj is None:
            return None
        return obj

    with open(output_path, "w") as f:
        json.dump(_to_serializable(all_results), f, indent=2)

    log.info(f"\nResults saved to: {output_path}")

    # ------------------------------------------------------------------
    # Final summary across tasks
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("SUMMARY (all tasks)")
    log.info("=" * 60)
    for task_label, task_results in all_results.items():
        rr = task_results.get("_recovery_ratios", {})
        log.info(f"  Task: {task_label}")
        for baseline_name, metrics in task_results.items():
            if baseline_name.startswith("_"):
                continue
            if not isinstance(metrics, dict):
                continue
            log.info(
                f"    {baseline_name:<6}  "
                f"survival={metrics.get('survival_pct', float('nan')):>6.1f}%  "
                f"ep_len={metrics.get('episode_length', float('nan')):>7.1f}  "
                f"com_err={metrics.get('com_error', float('nan')):>8.4f}  "
                f"orient_err={metrics.get('base_orient_error', float('nan')):>8.4f}"
            )
        if rr and any(v is not None for v in rr.values()):
            rr_str = "  ".join(
                f"{k}={v:.3f}" if v is not None else f"{k}=N/A"
                for k, v in rr.items()
            )
            log.info(f"    Recovery Ratio (B5): {rr_str}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate AnticiPose baselines (B1, B2, B4a, B5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoint paths
    parser.add_argument(
        "--b1_checkpoint",
        type=str,
        default=None,
        help="Path to B1 (reactive baseline) checkpoint .pt file.",
    )
    parser.add_argument(
        "--b2_checkpoint",
        type=str,
        default=None,
        help="Path to B2 (oracle) checkpoint .pt file.",
    )
    parser.add_argument(
        "--b4a_checkpoint",
        type=str,
        default=None,
        help="Path to B4a (raw-plan) checkpoint .pt file.",
    )
    parser.add_argument(
        "--b5_checkpoint",
        type=str,
        default=None,
        help="Path to B5 (AnticiPose) checkpoint .pt file.",
    )

    # Config
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to a training config.yaml to use for environment creation. "
             "Usually found at <experiment_dir>/config.yaml.",
    )

    # Evaluation settings
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of complete episodes per baseline per task.",
    )
    parser.add_argument(
        "--num_envs_eval",
        type=int,
        default=16,
        help="Number of parallel environments during evaluation.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/eval_results.json",
        help="Path to write the JSON results file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device: 'cuda', 'cuda:0', 'cpu', etc.",
    )

    # Generalization evaluation
    parser.add_argument(
        "--generalization_tasks",
        type=str,
        nargs="*",
        default=[],
        help="Names of held-out generalization tasks to evaluate.",
    )
    parser.add_argument(
        "--task_config_overrides",
        type=str,
        nargs="*",
        default=[],
        help="key=value config overrides applied to all generalization tasks. "
             "Example: --task_config_overrides apply_force_x_range=60.0 max_episode_length_s=30",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
