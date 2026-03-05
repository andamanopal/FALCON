"""
Record video clips from trained policies using Isaac Gym viewer.

On a headless VM, start Xvfb BEFORE running this script:
    Xvfb :99 -screen 0 1280x720x24 &
    export DISPLAY=:99
    export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

Usage:
    cd FALCON/
    python scripts/record_video.py \
        --checkpoint /workspace/Experiments/SEED_42/B4b_direct_plan_critic/model_3000.pt \
        --task lateral_slam_down \
        --output_dir /workspace/video_clips/B4b_lateral_slam_down_s42 \
        --num_steps 400 \
        --seed 42
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

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


def main():
    parser = argparse.ArgumentParser(
        description="Record video from a trained policy"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", default="lateral_slam_down")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--cvae_ckpt", default=None)
    parser.add_argument("--wrench_predictor_ckpt", default=None)
    parser.add_argument(
        "--cam_pos", nargs=3, type=float, default=[2.0, -2.0, 1.0],
    )
    parser.add_argument(
        "--cam_target", nargs=3, type=float, default=[0.0, 0.0, 0.5],
    )
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()

    # Verify DISPLAY is set (needed for Isaac Gym viewer on headless VM)
    if "DISPLAY" not in os.environ:
        print("[ERROR] DISPLAY not set. On a headless VM, run:")
        print("  Xvfb :99 -screen 0 1280x720x24 &")
        print("  export DISPLAY=:99")
        sys.exit(1)

    # Must import isaacgym before torch
    import isaacgym  # noqa: F401
    from isaacgym import gymapi
    import torch
    import cv2
    from hydra.utils import instantiate
    from humanoidverse.utils.helpers import pre_process_config
    from loguru import logger

    # Load training config
    ckpt_path = Path(args.checkpoint)
    config_path = ckpt_path.parent / "config.yaml"
    if not config_path.exists():
        config_path = ckpt_path.parent.parent / "config.yaml"
    if not config_path.exists():
        print(f"[ERROR] Config not found near {ckpt_path}")
        sys.exit(1)

    config = OmegaConf.load(config_path)

    # headless=False: viewer renders into Xvfb virtual display
    config.headless = False
    config.num_envs = 1
    config.env.config.num_envs = 1
    config.env.config.headless = False
    config.env.config.max_episode_length_s = 10000
    config.env.config.arm_trajectory_task = args.task
    config.env.config.save_rendering_dir = args.output_dir
    config.use_wandb = False

    if args.seed is not None:
        config.seed = args.seed
    if args.cvae_ckpt is not None:
        config.env.config.cvae_ckpt = args.cvae_ckpt
    if args.wrench_predictor_ckpt is not None:
        config.env.config.wrench_predictor_ckpt = args.wrench_predictor_ckpt

    pre_process_config(config)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"[Record] DISPLAY={os.environ.get('DISPLAY')}")
    print(f"[Record] VK_ICD_FILENAMES={os.environ.get('VK_ICD_FILENAMES', 'not set')}")
    print("[Record] Creating environment...")
    env = instantiate(config.env, device=device)
    print("[Record] Environment created")

    gym = env.simulator.gym
    sim = env.simulator.sim

    # Load policy
    algo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(str(ckpt_path))

    for key in algo.actors:
        algo.actors[key].eval()

    env.set_is_evaluating()

    # Create output directory
    out_dir = Path(args.output_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    obs_dict = env.reset_all()

    # Force short onset AFTER reset_all (reset re-randomizes onset_time)
    traj = env._arm_traj_gen
    if hasattr(traj, 'generators'):
        for gen in traj.generators.values():
            gen.onset_time[:] = 0.3
    else:
        traj.onset_time[:] = 0.3
    print(f"[Record] Forced onset_time = 0.3s (after reset)")

    # --- Phase detection setup ---
    gen = traj.generators[args.task] if hasattr(traj, 'generators') else traj
    onset = 0.3  # We just forced it
    dt = gen.dt

    # Compute phase boundaries from generator parameters
    # Most tasks use trapezoid: ramp_up -> hold -> ramp_down
    # Get speed and dominant amplitude to compute t_ramp
    if hasattr(gen, 'pitch_amp'):
        amp = gen.pitch_amp[0].item()
    elif hasattr(gen, 'roll_amp'):
        amp = gen.roll_amp[0].item()
    elif hasattr(gen, 'push_amp'):
        amp = gen.push_amp[0].item()
    else:
        amp = 1.0

    spd = gen.speed[0].item() if hasattr(gen, 'speed') else 2.0
    t_ramp = amp / max(spd, 1e-6)
    t_hold = gen.hold_time[0].item() if hasattr(gen, 'hold_time') else 1.0

    # Task-specific phase detection
    is_slam = args.task == "lateral_slam_down"
    is_periodic = args.task == "gangnam_style"
    if is_slam:
        t_arrest = gen.t_arrest[0].item() if hasattr(gen, 't_arrest') else 0.15
    if is_periodic:
        freq = gen.freq[0].item() if hasattr(gen, 'freq') else 2.5

    def get_phase_label(step_idx):
        """Return (phase_name, progress_0_to_1) for the current step."""
        t = step_idx * dt
        elapsed = t - onset
        if elapsed < 0:
            return "Idle", 0.0
        if is_periodic:
            if elapsed < 0.3:
                return "Ramp In", elapsed / 0.3
            # Show beat count and phase within current beat
            beat = elapsed * freq
            beat_frac = beat % 1.0
            beat_num = int(beat) + 1
            return f"Beat {beat_num}", beat_frac
        if is_slam:
            if elapsed < t_ramp:
                return "Swing", elapsed / max(t_ramp, 1e-6)
            elif elapsed < t_ramp + t_arrest:
                return "Arrest", (elapsed - t_ramp) / max(t_arrest, 1e-6)
            else:
                return "Hold", 1.0
        else:
            t_total = 2.0 * t_ramp + t_hold
            if elapsed < t_ramp:
                return "Ramp Up", elapsed / max(t_ramp, 1e-6)
            elif elapsed < t_ramp + t_hold:
                return "Hold", (elapsed - t_ramp) / max(t_hold, 1e-6)
            elif elapsed < t_total:
                return "Ramp Down", (elapsed - t_ramp - t_hold) / max(t_ramp, 1e-6)
            else:
                return "Done", 1.0

    # Also compute arm target magnitude per step for the intensity bar
    phase_labels = []

    print(f"[Record] Checkpoint: {ckpt_path}")
    print(f"[Record] Task: {args.task}")
    print(f"[Record] Steps: {args.num_steps}")
    print(f"[Record] Onset: {onset:.2f}s, Ramp: {t_ramp:.2f}s, Hold: {t_hold:.2f}s")
    print(f"[Record] Recording...")

    for step in range(args.num_steps):
        with torch.no_grad():
            actor_obs = obs_dict["actor_obs"]
            act_parts = {}
            for key in algo.actors:
                act_parts[key] = algo.actors[key].act_inference(actor_obs)
            actions = torch.cat([act_parts[key] for key in algo.keys], dim=1)

        actor_state = {"actions": actions}
        obs_dict, rewards, dones, extras = env.step(actor_state)

        # Track phase
        label, progress = get_phase_label(step)
        phase_labels.append((label, progress))

        # Capture frame from viewer
        gym.step_graphics(sim)
        gym.draw_viewer(env.viewer, sim, True)

        frame_path = str(frames_dir / f"{step:05d}.png")
        gym.write_viewer_image_to_file(env.viewer, frame_path)

        if (step + 1) % 100 == 0:
            print(f"  Frame {step + 1}/{args.num_steps} [{label}]")

    print(f"[Record] Saved {args.num_steps} frames to {frames_dir}")

    # Stitch to video with phase overlays
    video_path = str(out_dir / "clip.mp4")
    print(f"[Record] Stitching video with phase labels...")

    # Condition label from checkpoint path (e.g. "B4b_direct_plan_critic" -> "B4b: Privileged Critic")
    CONDITION_LABELS = {
        "B1_reactive": "B1: Reactive",
        "B2_extended_history": "B2: Extended History",
        "B4a_direct_plan": "B4a: Plan in Actor",
        "B4b_direct_plan_critic": "B4b: Plan in Critic",
        "B4c_direct_plan_both": "B4c: Plan in Both",
    }
    cond_name = ckpt_path.parent.name
    cond_label = CONDITION_LABELS.get(cond_name, cond_name)
    print(f"[Record] Condition dir: '{cond_name}' -> label: '{cond_label}'")

    # Task display name
    TASK_LABELS = {
        "overhead_reach": "Extreme Arm Pitch (Held-Out)",
        "bilateral_asymmetric_lift": "Asymmetric Bilateral Lift (Held-Out)",
        "lateral_slam_down": "Lateral Slam + Arrest (Held-Out)",
        "cross_body_reach": "Cross-Body Reach (Held-Out)",
        "forward_push": "Impulsive Push (Training)",
        "frontal_reach_lift": "Frontal Reach Lift (Training)",
        "lateral_shelf_pick": "Lateral Shelf Pick (Training)",
        "backward_swing": "Backward Swing (Held-Out)",
        "frontal_raise": "Frontal Raise (Held-Out)",
        "gangnam_style": "Gangnam Style (Held-Out)",
    }
    task_label = TASK_LABELS.get(args.task, args.task)

    # Phase colors (BGR for OpenCV)
    PHASE_COLORS = {
        "Idle":      (128, 128, 128),  # gray
        "Ramp Up":   (0, 200, 255),    # orange
        "Swing":     (0, 200, 255),    # orange
        "Hold":      (0, 0, 255),      # red
        "Arrest":    (0, 0, 200),      # dark red
        "Ramp Down": (200, 200, 0),    # cyan
        "Done":      (128, 128, 128),  # gray
    }

    images = sorted(frames_dir.glob("*.png"))
    if images:
        sample = cv2.imread(str(images[0]))
        h, w, _ = sample.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_path, fourcc, args.fps, (w, h))

        font = cv2.FONT_HERSHEY_SIMPLEX

        for i, img_path in enumerate(images):
            frame = cv2.imread(str(img_path))

            label, progress = phase_labels[i] if i < len(phase_labels) else ("", 0.0)
            color = PHASE_COLORS.get(label, (255, 255, 255))
            t_sec = i / args.fps

            # --- Top-left: condition + task ---
            cv2.putText(frame, cond_label, (16, 36),
                        font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, task_label, (16, 66),
                        font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

            # --- Bottom-left: phase label + time ---
            phase_text = f"{label}  ({t_sec:.1f}s)"
            cv2.putText(frame, phase_text, (16, h - 50),
                        font, 0.7, color, 2, cv2.LINE_AA)

            # --- Bottom: progress bar ---
            bar_x, bar_y = 16, h - 30
            bar_w, bar_h = w - 32, 14
            # Background
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h),
                          (50, 50, 50), -1)
            # Fill
            fill_w = int(bar_w * progress)
            if fill_w > 0:
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + fill_w, bar_y + bar_h),
                              color, -1)
            # Border
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + bar_w, bar_y + bar_h),
                          (180, 180, 180), 1)

            video.write(frame)
        video.release()

        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        duration_s = len(images) / args.fps
        print(f"[Record] Video: {video_path} ({duration_s:.1f}s, {size_mb:.1f} MB)")
    else:
        print("[Record] No frames found")

    print("[Record] Done.")


if __name__ == "__main__":
    main()
