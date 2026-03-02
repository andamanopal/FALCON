"""
Collect per-step torque and wrench time-series for APA analysis.

Records lower-body joint torques, base wrench, and timing aligned to arm
motion onset for B1 vs B5 comparison.

Usage:
    cd FALCON/
    python scripts/collect_torque_timeseries.py \
        --checkpoint logs/anticipose_overnight/.../model_3000.pt \
        --task lateral_shelf_pick \
        --num_episodes 50 \
        --num_envs 64 \
        --output torque_data/B1_lateral_shelf_pick_s42.pt
"""
import argparse
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
    pass

# Lower-body DOF indices in the 29-DOF G1
DOF_INDICES = [4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 14]
DOF_NAMES = [
    "L_ankle_pitch", "L_ankle_roll",
    "R_ankle_pitch", "R_ankle_roll",
    "L_hip_pitch", "L_hip_roll",
    "R_hip_pitch", "R_hip_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
]


def main():
    parser = argparse.ArgumentParser(
        description="Collect torque time-series for APA analysis",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True,
                        help="Arm trajectory task name (e.g. lateral_shelf_pick)")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--max_episode_length_s", type=float, default=20.0)
    parser.add_argument("--wrench_predictor_ckpt", default=None)
    parser.add_argument("--cvae_ckpt", default=None)
    parser.add_argument("--output", required=True,
                        help="Output .pt file path")
    args = parser.parse_args()

    # Must import isaacgym before torch
    import isaacgym  # noqa: F401
    import torch
    from hydra.utils import instantiate
    from humanoidverse.utils.helpers import pre_process_config
    from loguru import logger

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
    config.env.config.arm_trajectory_task = args.task
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

    for key in algo.actors:
        algo.actors[key].eval()

    sim_fps = config.simulator.config.sim.fps
    control_dec = config.simulator.config.sim.control_decimation
    dt = control_dec / sim_fps
    max_steps = int(args.max_episode_length_s * sim_fps / control_dec)
    num_envs = args.num_envs

    print(f"\n[Torque Collection]")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Task: {args.task}")
    print(f"  dt: {dt:.4f}s  max_steps: {max_steps}")
    print(f"  num_envs: {num_envs}  target episodes: {args.num_episodes}")

    # Per-env accumulators
    torque_accum = [[] for _ in range(num_envs)]
    wrench_accum = [[] for _ in range(num_envs)]
    time_accum = [[] for _ in range(num_envs)]

    # Per-env step counter (resets per episode, tracks episode-local time)
    env_step_count = torch.zeros(num_envs, dtype=torch.long, device="cpu")

    # Completed episodes
    episodes = []
    total_done = 0

    # Reset all envs
    obs_dict = env.reset_all()

    while total_done < args.num_episodes:
        # Snapshot onset_time BEFORE step (reset happens within step,
        # so by the time step() returns, done envs already have NEW onset)
        pre_step_onset = env._arm_traj_gen.onset_time.clone().cpu()

        # Inference
        with torch.no_grad():
            actor_obs = obs_dict["actor_obs"]
            act_parts = {}
            for key in algo.actors:
                act_parts[key] = algo.actors[key].act_inference(actor_obs)
            actions = torch.cat([act_parts[key] for key in algo.keys], dim=1)

        actor_state = {"actions": actions}
        obs_dict, rewards, dones, extras = env.step(actor_state)
        env_step_count += 1

        # Record data for all envs (using per-env episode time)
        torques_step = env.torques[:, DOF_INDICES].detach().cpu()
        wrench_step = env._current_wrench.detach().cpu()

        for i in range(num_envs):
            ep_time_s = env_step_count[i].item() * dt
            torque_accum[i].append(torques_step[i])
            wrench_accum[i].append(wrench_step[i])
            time_accum[i].append(ep_time_s)

        # Check dones
        time_out_buf = extras["time_outs"]
        done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

        for idx in done_indices:
            i = idx.item()
            if total_done >= args.num_episodes:
                break

            # Use pre-step onset time (env already reset by now)
            ep_onset = pre_step_onset[i].item()
            survived = bool(time_out_buf[i].item())
            ep_len = len(torque_accum[i])

            if ep_len > 0:
                episode_data = {
                    "torques": torch.stack(torque_accum[i]),
                    "wrenches": torch.stack(wrench_accum[i]),
                    "time_s": torch.tensor(time_accum[i]),
                    "onset_time_s": ep_onset,
                    "survived": survived,
                    "length_steps": ep_len,
                }
                episodes.append(episode_data)
                total_done += 1

                if total_done % 10 == 0:
                    surv_count = sum(
                        1 for ep in episodes if ep["survived"]
                    )
                    print(f"  Episodes: {total_done}/{args.num_episodes} "
                          f"(survived: {surv_count})")

            # Clear accumulators and reset step counter for this env
            torque_accum[i] = []
            wrench_accum[i] = []
            time_accum[i] = []
            env_step_count[i] = 0

    # Build output
    output = {
        "episodes": episodes,
        "metadata": {
            "dt": dt,
            "task": args.task,
            "checkpoint": args.checkpoint,
            "dof_names": DOF_NAMES,
            "dof_indices": DOF_INDICES,
            "num_episodes": len(episodes),
            "max_steps": max_steps,
        },
    }

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)

    surv_count = sum(1 for ep in episodes if ep["survived"])
    print(f"\n[Done] Saved {len(episodes)} episodes to {out_path}")
    print(f"  Survived: {surv_count}/{len(episodes)} "
          f"({100*surv_count/len(episodes):.1f}%)")
    print(f"  Mean length: "
          f"{sum(ep['length_steps'] for ep in episodes)/len(episodes):.1f} steps")


if __name__ == "__main__":
    main()
