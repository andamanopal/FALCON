"""
Measure advantage variance across baselines to verify that privileged
critic information (B4b) reduces advantage estimation variance vs B1.

Approach:
  1. Load trained checkpoints for each baseline × seed
  2. Run 24 rollout steps of inference (no gradient)
  3. Compute raw GAE advantages (gamma=0.99, lambda=0.95) before normalisation
  4. Report per-step advantage variance per condition × seed
  5. Statistical comparison via Welch's t-test

Usage:
    cd FALCON/
    python scripts/measure_advantage_variance.py \
        --baselines B1,B4a,B4b,B4c \
        --seeds 42,123,456,789,35 \
        --num_rollout_steps 24 \
        --log_dir logs/anticipose_overnight \
        --output_dir logs_eval/advantage_variance
"""
import argparse
import json
import math
import sys
from pathlib import Path

from omegaconf import OmegaConf

# Register FALCON's custom OmegaConf resolvers before any config loading
try:
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
    OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
    OmegaConf.register_new_resolver("sqrt", lambda x: math.sqrt(float(x)))
    OmegaConf.register_new_resolver("sum", lambda x: sum(x))
    OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(x))
    OmegaConf.register_new_resolver("int", lambda x: int(x))
    OmegaConf.register_new_resolver("len", lambda x: len(x))
    OmegaConf.register_new_resolver("sum_list", lambda lst: sum(lst))
except Exception:
    pass  # resolvers already registered


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


def compute_raw_gae(values, rewards, dones, last_values, gamma, lam):
    """
    Compute raw (un-normalised) GAE advantages.

    Replicates the exact logic from PPOMultiActorCritic._compute_returns()
    but returns the raw advantages before mean/std normalisation.

    Args:
        values      : (num_steps, num_envs, 1) critic value estimates
        rewards     : (num_steps, num_envs, 1) rewards per step
        dones       : (num_steps, num_envs, 1) done flags
        last_values : (num_envs, 1) critic value at terminal obs
        gamma       : discount factor
        lam         : GAE lambda

    Returns:
        raw_advantages : (num_steps, num_envs, 1)
    """
    num_steps = values.shape[0]
    returns = values.clone()
    advantage = 0

    for step in reversed(range(num_steps)):
        if step == num_steps - 1:
            next_values = last_values
        else:
            next_values = values[step + 1]
        next_is_not_terminal = 1.0 - dones[step].float()
        delta = (
            rewards[step]
            + next_is_not_terminal * gamma * next_values
            - values[step]
        )
        advantage = (
            delta + next_is_not_terminal * gamma * lam * advantage
        )
        returns[step] = advantage + values[step]

    raw_advantages = returns - values
    return raw_advantages


def run_rollout_and_measure(ckpt_path, num_rollout_steps, device):
    """
    Load a checkpoint, run inference rollout, compute raw advantage variance.

    Returns dict with per-body-key advantage variance and overall stats.
    """
    import isaacgym  # noqa: F401
    import torch
    from hydra.utils import instantiate
    from humanoidverse.utils.helpers import pre_process_config

    ckpt_path = Path(ckpt_path)
    config_path = ckpt_path.parent / "config.yaml"
    if not config_path.exists():
        config_path = ckpt_path.parent.parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found near {ckpt_path}")

    config = OmegaConf.load(config_path)

    # Apply eval overrides
    num_envs = 64
    config.headless = True
    config.num_envs = num_envs
    config.env.config.num_envs = num_envs
    config.env.config.headless = True

    pre_process_config(config)

    env = instantiate(config.env, device=device)
    algo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(str(ckpt_path))

    # Switch to eval mode
    for key in algo.actors:
        algo.actors[key].eval()
    for key in algo.critics:
        algo.critics[key].eval()

    gamma = algo.gamma
    lam = algo.lam
    keys = algo.keys

    # Run rollout: collect values, rewards, dones per step
    obs_dict = env.reset_all()
    for obs_key in obs_dict:
        obs_dict[obs_key] = obs_dict[obs_key].to(device)

    # Storage: per body key
    collected = {
        key: {"values": [], "rewards": [], "dones": []}
        for key in keys
    }

    with torch.inference_mode():
        for step_i in range(num_rollout_steps):
            # Compute actions and values
            act_parts = {}
            for key in keys:
                act_parts[key] = algo.actors[key].act_inference(
                    obs_dict["actor_obs"]
                )
            actions = torch.cat(
                [act_parts[key] for key in keys], dim=1
            )

            # Record critic values before step
            for key in keys:
                value = algo.critics[key].evaluate(
                    obs_dict["critic_obs"]
                ).detach()
                collected[key]["values"].append(value)

            # Step environment
            actor_state = {"actions": actions}
            obs_dict, rewards, dones, infos = env.step(actor_state)

            for obs_key in obs_dict:
                obs_dict[obs_key] = obs_dict[obs_key].to(device)
            dones = dones.to(device)

            # Record rewards and dones
            for key in keys:
                rw = rewards[key].to(device).clone().unsqueeze(1)
                dn = dones.clone().unsqueeze(1)
                # Bootstrap on time outs (same as training)
                if "time_outs" in infos:
                    rw = (
                        rw
                        + gamma
                        * collected[key]["values"][-1]
                        * infos["time_outs"].unsqueeze(1).to(device)
                    )
                collected[key]["rewards"].append(rw)
                collected[key]["dones"].append(dn)

        # Get last values for GAE bootstrap
        last_values = {}
        for key in keys:
            last_values[key] = algo.critics[key].evaluate(
                obs_dict["critic_obs"]
            ).detach()

    # Compute raw GAE advantages per body key
    results = {}
    for key in keys:
        values_t = torch.stack(collected[key]["values"], dim=0)
        rewards_t = torch.stack(collected[key]["rewards"], dim=0)
        dones_t = torch.stack(collected[key]["dones"], dim=0)

        raw_adv = compute_raw_gae(
            values_t, rewards_t, dones_t, last_values[key], gamma, lam
        )

        # Variance across all (steps, envs) — scalar
        adv_flat = raw_adv.reshape(-1)
        results[key] = {
            "advantage_variance": adv_flat.var().item(),
            "advantage_mean": adv_flat.mean().item(),
            "advantage_std": adv_flat.std().item(),
            "advantage_abs_mean": adv_flat.abs().mean().item(),
            "num_samples": adv_flat.numel(),
        }

    # Overall (average across body keys)
    all_vars = [results[k]["advantage_variance"] for k in keys]
    results["overall"] = {
        "advantage_variance_mean": sum(all_vars) / len(all_vars),
        "per_key_variances": {k: results[k]["advantage_variance"] for k in keys},
    }

    # Clean up GPU memory
    del env, algo
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Measure advantage variance across baselines"
    )
    parser.add_argument(
        "--baselines",
        default="B1,B4a,B4b,B4c",
        help="Comma-separated baseline IDs",
    )
    parser.add_argument(
        "--seeds",
        default="42,123,456,789,35",
        help="Comma-separated seeds",
    )
    parser.add_argument("--num_rollout_steps", type=int, default=24)
    parser.add_argument("--num_iters", type=int, default=3000)
    parser.add_argument("--log_dir", default="logs/anticipose_overnight")
    parser.add_argument("--output_dir", default="logs_eval/advantage_variance")
    args = parser.parse_args()

    import torch

    baselines = [b.strip() for b in args.baselines.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for baseline in baselines:
        pattern = BASELINE_PATTERNS.get(baseline)
        if pattern is None:
            print(f"[WARN] Unknown baseline: {baseline}")
            continue

        all_results[baseline] = {}

        for seed in seeds:
            ckpt = find_checkpoint(
                args.log_dir, pattern, seed, args.num_iters
            )
            if ckpt is None:
                print(f"[SKIP] No checkpoint for {baseline} seed={seed}")
                continue

            print(f"\n{'='*60}")
            print(f"  {baseline} seed={seed}")
            print(f"  Checkpoint: {ckpt}")
            print(f"  Rollout steps: {args.num_rollout_steps}")
            print(f"{'='*60}")

            result = run_rollout_and_measure(
                ckpt, args.num_rollout_steps, device
            )
            all_results[baseline][str(seed)] = result

            var_str = result["overall"]["advantage_variance_mean"]
            print(f"  -> Overall advantage variance: {var_str:.6f}")
            for k, v in result["overall"]["per_key_variances"].items():
                print(f"     {k}: {v:.6f}")

    # Save full results
    results_path = output_dir / "advantage_variance_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  Advantage Variance Summary")
    print(f"{'='*70}")
    print(f"{'Baseline':<10} {'Seed':<8} {'Variance':>12} {'Std':>12} {'|Adv| mean':>12}")
    print("-" * 56)

    for baseline in baselines:
        if baseline not in all_results:
            continue
        for seed in seeds:
            seed_str = str(seed)
            if seed_str not in all_results[baseline]:
                continue
            r = all_results[baseline][seed_str]
            var_avg = r["overall"]["advantage_variance_mean"]
            # Average std across keys
            stds = [r[k]["advantage_std"] for k in r if k != "overall"]
            abs_means = [r[k]["advantage_abs_mean"] for k in r if k != "overall"]
            avg_std = sum(stds) / len(stds) if stds else 0
            avg_abs = sum(abs_means) / len(abs_means) if abs_means else 0
            print(
                f"{baseline:<10} {seed:<8} "
                f"{var_avg:>12.6f} {avg_std:>12.6f} {avg_abs:>12.6f}"
            )

    # Cross-baseline variance comparison
    print(f"\n{'='*70}")
    print(f"  Cross-Baseline Comparison (mean variance across seeds)")
    print(f"{'='*70}")
    print(f"{'Baseline':<10} {'Mean Var':>12} {'Std Var':>12} {'n seeds':>8}")
    print("-" * 44)

    baseline_summary = {}
    for baseline in baselines:
        if baseline not in all_results:
            continue
        vars_list = [
            all_results[baseline][s]["overall"]["advantage_variance_mean"]
            for s in all_results[baseline]
        ]
        if len(vars_list) == 0:
            continue
        mean_var = sum(vars_list) / len(vars_list)
        if len(vars_list) > 1:
            var_of_vars = sum((v - mean_var) ** 2 for v in vars_list) / (
                len(vars_list) - 1
            )
            std_var = var_of_vars ** 0.5
        else:
            std_var = 0.0
        baseline_summary[baseline] = {
            "mean": mean_var,
            "std": std_var,
            "n": len(vars_list),
        }
        print(
            f"{baseline:<10} {mean_var:>12.6f} {std_var:>12.6f} {len(vars_list):>8}"
        )

    # Welch's t-test: B4b vs B1
    if "B1" in baseline_summary and "B4b" in baseline_summary:
        b1 = baseline_summary["B1"]
        b4b = baseline_summary["B4b"]
        if b1["n"] > 1 and b4b["n"] > 1:
            # Welch's t-statistic
            se = (b1["std"] ** 2 / b1["n"] + b4b["std"] ** 2 / b4b["n"]) ** 0.5
            if se > 0:
                t_stat = (b1["mean"] - b4b["mean"]) / se
                print(f"\nWelch's t-test (B1 vs B4b): t = {t_stat:.3f}")
                print(
                    f"  B1 mean var:  {b1['mean']:.6f}"
                    f"  B4b mean var: {b4b['mean']:.6f}"
                )
                reduction_pct = (
                    (b1["mean"] - b4b["mean"]) / b1["mean"] * 100
                    if b1["mean"] > 0
                    else 0
                )
                print(f"  Variance reduction: {reduction_pct:.1f}%")


if __name__ == "__main__":
    main()
