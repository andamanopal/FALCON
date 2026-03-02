"""
Evaluate AnticiPose baselines: run N episodes, collect metrics, save results.

Usage:
    cd FALCON/
    python scripts/eval_baselines.py \
        --checkpoint logs/anticipose_overnight/.../model_3000.pt \
        --eval_name eval_B1_reactive_train \
        --num_episodes 50 \
        --max_episode_length_s 20 \
        --arm_trajectory_task random

    # For B5 anticipose, add:
        --wrench_predictor_ckpt logs/anticipose_overnight/wrench_predictor_seed35.pt
"""
import argparse
import json
import math
import sys
import os
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_name", required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_episode_length_s", type=float, default=20.0)
    parser.add_argument("--arm_trajectory_task", default="random")
    parser.add_argument("--wrench_predictor_ckpt", default=None)
    parser.add_argument("--cvae_ckpt", default=None)
    parser.add_argument("--output_dir", default="logs_eval")
    parser.add_argument(
        "--walking_speeds", nargs="*", type=float, default=None,
        help="Walking speed sweep: eval at each speed (m/s). "
             "E.g., --walking_speeds 0.0 0.3 0.6 1.0",
    )
    args = parser.parse_args()

    # Must import isaacgym before torch
    import isaacgym  # noqa: F401
    import torch
    from hydra.utils import instantiate
    from humanoidverse.utils.helpers import pre_process_config
    from loguru import logger

    # Load training config from checkpoint directory
    ckpt_path = Path(args.checkpoint)
    config_path = ckpt_path.parent / "config.yaml"
    if not config_path.exists():
        config_path = ckpt_path.parent.parent / "config.yaml"
    if not config_path.exists():
        print(f"[ERROR] Config not found near {ckpt_path}")
        sys.exit(1)

    config = OmegaConf.load(config_path)

    # Apply eval overrides
    config.headless = True
    config.num_envs = args.num_envs
    config.env.config.num_envs = args.num_envs
    config.env.config.headless = True
    config.env.config.max_episode_length_s = args.max_episode_length_s
    config.env.config.arm_trajectory_task = args.arm_trajectory_task
    if args.wrench_predictor_ckpt is not None:
        config.env.config.wrench_predictor_ckpt = args.wrench_predictor_ckpt
    if args.cvae_ckpt is not None:
        config.env.config.cvae_ckpt = args.cvae_ckpt

    pre_process_config(config)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create env and algo
    env = instantiate(config.env, device=device)
    algo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(str(ckpt_path))

    # Switch to eval mode
    for key in algo.actors:
        algo.actors[key].eval()

    sim_fps = config.simulator.config.sim.fps
    control_dec = config.simulator.config.sim.control_decimation
    max_steps = int(args.max_episode_length_s * sim_fps / control_dec)
    num_envs = args.num_envs
    num_actions = config.robot.lower_body_actions_dim + config.robot.upper_body_actions_dim

    # Determine walking speeds to evaluate
    walking_speeds = args.walking_speeds if args.walking_speeds is not None else [None]

    all_speed_results = []
    for speed in walking_speeds:
        # Configure eval mode with optional walking speed command
        if speed is not None:
            env.set_is_evaluating(command=[speed, 0.0, 0.0])
            speed_label = f"speed_{speed:.1f}"
            eval_label = f"{args.eval_name}_{speed_label}"
        else:
            env.set_is_evaluating()
            speed_label = None
            eval_label = args.eval_name

        print(f"\n[Eval] {eval_label}")
        print(f"[Eval] checkpoint: {args.checkpoint}")
        print(f"[Eval] task: {args.arm_trajectory_task}")
        if speed is not None:
            print(f"[Eval] walking speed: {speed:.1f} m/s")
        print(f"[Eval] num_envs={num_envs}, num_episodes={args.num_episodes}, max_steps={max_steps}")

        # Run episodes
        episode_rewards = []
        episode_lengths = []
        episode_survived = []
        episode_orientation_rms = []
        episode_vel_err_rms = []
        episode_angvel_err_rms = []
        episode_peak_force = []
        episode_peak_torque = []

        total_episodes_done = 0
        obs_dict = env.reset_all()
        cumulative_reward = torch.zeros(num_envs, device=device)
        episode_len = torch.zeros(num_envs, device=device)

        # Physical metric accumulators (per-env, GPU)
        sum_orientation_sq = torch.zeros(num_envs, device=device)
        sum_vel_err_sq = torch.zeros(num_envs, device=device)
        sum_angvel_err_sq = torch.zeros(num_envs, device=device)
        peak_wrench_force = torch.zeros(num_envs, device=device)
        peak_wrench_torque = torch.zeros(num_envs, device=device)
        has_wrench = hasattr(env, '_current_wrench')

        while total_episodes_done < args.num_episodes:
            with torch.no_grad():
                actor_obs = obs_dict["actor_obs"]
                act_parts = {}
                for key in algo.actors:
                    act_parts[key] = algo.actors[key].act_inference(actor_obs)
                actions = torch.cat([act_parts[key] for key in algo.keys], dim=1)

            actor_state = {"actions": actions}
            obs_dict, rewards, dones, extras = env.step(actor_state)

            cumulative_reward += sum(rewards.values())
            episode_len += 1

            # Physical metrics accumulation
            roll = env.rpy[:, 0]
            pitch = env.rpy[:, 1]
            sum_orientation_sq += roll ** 2 + pitch ** 2

            cmd_vx = env.commands[:, 0]
            cmd_vy = env.commands[:, 1]
            sum_vel_err_sq += (cmd_vx - env.base_lin_vel[:, 0]) ** 2 + \
                              (cmd_vy - env.base_lin_vel[:, 1]) ** 2

            cmd_wz = env.commands[:, 2]
            sum_angvel_err_sq += (cmd_wz - env.base_ang_vel[:, 2]) ** 2

            if has_wrench:
                wrench = env._current_wrench
                force_norm = torch.norm(wrench[:, :3], dim=1)
                torque_norm = torch.norm(wrench[:, 3:], dim=1)
                peak_wrench_force = torch.max(peak_wrench_force, force_norm)
                peak_wrench_torque = torch.max(peak_wrench_torque, torque_norm)

            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_indices:
                i = idx.item()
                ep_reward = cumulative_reward[i].item()
                ep_len = episode_len[i].item()
                survived = bool(extras["time_outs"][i].item())

                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_len)
                episode_survived.append(survived)

                steps = max(ep_len, 1)
                ori_rms_deg = math.degrees(math.sqrt(sum_orientation_sq[i].item() / steps))
                vel_err = math.sqrt(sum_vel_err_sq[i].item() / steps)
                angvel_err = math.sqrt(sum_angvel_err_sq[i].item() / steps)

                episode_orientation_rms.append(ori_rms_deg)
                episode_vel_err_rms.append(vel_err)
                episode_angvel_err_rms.append(angvel_err)
                episode_peak_force.append(peak_wrench_force[i].item() if has_wrench else 0.0)
                episode_peak_torque.append(peak_wrench_torque[i].item() if has_wrench else 0.0)

                total_episodes_done += 1

                if total_episodes_done % 10 == 0:
                    print(f"  Episodes: {total_episodes_done}/{args.num_episodes}")

                cumulative_reward[i] = 0.0
                episode_len[i] = 0.0
                sum_orientation_sq[i] = 0.0
                sum_vel_err_sq[i] = 0.0
                sum_angvel_err_sq[i] = 0.0
                peak_wrench_force[i] = 0.0
                peak_wrench_torque[i] = 0.0

                if total_episodes_done >= args.num_episodes:
                    break

        # Compute metrics
        rewards_t = torch.tensor(episode_rewards)
        lengths_t = torch.tensor(episode_lengths)
        survived_t = torch.tensor(episode_survived, dtype=torch.float32)

        ori_t = torch.tensor(episode_orientation_rms)
        vel_t = torch.tensor(episode_vel_err_rms)
        angvel_t = torch.tensor(episode_angvel_err_rms)
        force_t = torch.tensor(episode_peak_force)
        torque_t = torch.tensor(episode_peak_torque)

        results = {
            "eval_name": eval_label,
            "checkpoint": args.checkpoint,
            "arm_trajectory_task": args.arm_trajectory_task,
            "walking_speed": speed,
            "num_episodes": len(episode_rewards),
            "max_steps": max_steps,
            "mean_reward": rewards_t.mean().item(),
            "std_reward": rewards_t.std().item(),
            "mean_episode_length": lengths_t.mean().item(),
            "std_episode_length": lengths_t.std().item(),
            "survival_rate": survived_t.mean().item(),
            "min_reward": rewards_t.min().item(),
            "max_reward": rewards_t.max().item(),
            "mean_orientation_rms_deg": ori_t.mean().item(),
            "std_orientation_rms_deg": ori_t.std().item(),
            "mean_vel_tracking_err_rms": vel_t.mean().item(),
            "std_vel_tracking_err_rms": vel_t.std().item(),
            "mean_angvel_tracking_err_rms": angvel_t.mean().item(),
            "std_angvel_tracking_err_rms": angvel_t.std().item(),
            "mean_peak_wrench_force_N": force_t.mean().item(),
            "std_peak_wrench_force_N": force_t.std().item(),
            "mean_peak_wrench_torque_Nm": torque_t.mean().item(),
            "std_peak_wrench_torque_Nm": torque_t.std().item(),
        }
        all_speed_results.append(results)

        # Print results
        print(f"\n{'='*60}")
        print(f"  {eval_label}")
        print(f"{'='*60}")
        print(f"  Mean reward:      {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
        print(f"  Mean ep length:   {results['mean_episode_length']:.1f} +/- {results['std_episode_length']:.1f}")
        print(f"  Survival rate:    {results['survival_rate']*100:.1f}%")
        print(f"  Reward range:     [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
        print(f"  Orientation RMS:  {results['mean_orientation_rms_deg']:.2f} +/- {results['std_orientation_rms_deg']:.2f} deg")
        print(f"  Vel tracking err: {results['mean_vel_tracking_err_rms']:.3f} +/- {results['std_vel_tracking_err_rms']:.3f} m/s")
        print(f"  Angvel track err: {results['mean_angvel_tracking_err_rms']:.3f} +/- {results['std_angvel_tracking_err_rms']:.3f} rad/s")
        print(f"  Peak wrench F:    {results['mean_peak_wrench_force_N']:.1f} +/- {results['std_peak_wrench_force_N']:.1f} N")
        print(f"  Peak wrench T:    {results['mean_peak_wrench_torque_Nm']:.1f} +/- {results['std_peak_wrench_torque_Nm']:.1f} Nm")
        print(f"{'='*60}\n")

        # Save per-speed results
        out_dir = Path(args.output_dir) / eval_label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "results.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Eval] Results saved to {out_file}")

    # Save aggregate speed sweep results if multiple speeds were tested
    if len(all_speed_results) > 1:
        sweep_dir = Path(args.output_dir) / args.eval_name
        sweep_dir.mkdir(parents=True, exist_ok=True)
        sweep_file = sweep_dir / "speed_sweep_results.json"
        with open(sweep_file, "w") as f:
            json.dump(all_speed_results, f, indent=2)
        print(f"[Eval] Speed sweep aggregate saved to {sweep_file}")


if __name__ == "__main__":
    main()
