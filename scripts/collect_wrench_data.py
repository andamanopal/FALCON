"""
collect_wrench_data.py
----------------------
Collect (obs, plan, wrench) supervision data by rolling out a trained
reactive (B1) policy inside the AnticiPose IsaacGym environment.

The script uses the AnticiPoseEnv which internally:
  1. Generates scripted arm trajectories and stores the arm plan
  2. Computes analytical wrenches from arm rigid body dynamics
  3. Collects (obs, plan, wrench) tuples in its WrenchDataCollector

After the rollout, the collected data is saved to disk via
env.save_collected_wrench_data().

Usage (from FALCON/):
    python scripts/collect_wrench_data.py \
        +exp=anticipose \
        +robot=g1/g1_29dof_waist_fakehand \
        +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma \
        +simulator=isaacgym \
        +domain_rand=domain_rand_rl_gym \
        +rewards=dec_loco/reward_dec_loco_stand_height_ma_diff_force \
        +terrain=terrain_locomotion_plane \
        checkpoint=<path/to/b1_checkpoint.pt> \
        +output_path=data/wrench_data.pt \
        +num_samples=500000 \
        num_envs=4096 \
        headless=True \
        env.config.anticipose_mode=reactive \
        env.config.collect_wrench_data=true
"""

import os
import sys
from pathlib import Path

# Add humanoidverse/ to sys.path so train_agent-style imports resolve.
# This mirrors how FALCON's train_agent.py works (it lives inside
# humanoidverse/ and imports from utils.config_utils).
_SCRIPT_DIR = Path(__file__).resolve().parent
_FALCON_ROOT = _SCRIPT_DIR.parent
_HUMANOIDVERSE_DIR = _FALCON_ROOT / "humanoidverse"
if str(_HUMANOIDVERSE_DIR) not in sys.path:
    sys.path.insert(0, str(_HUMANOIDVERSE_DIR))
if str(_FALCON_ROOT) not in sys.path:
    sys.path.insert(0, str(_FALCON_ROOT))

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

import logging
from loguru import logger

# Bring config utilities into scope (same as train_agent.py)
from utils.config_utils import *  # noqa: E402, F403

# ---------------------------------------------------------------------------
# Constants matching VERIFIED_PARAMS.md
# ---------------------------------------------------------------------------
OBS_DIM       = 115
ARM_JOINTS    = 14
HORIZON       = 5
PLAN_DIM      = HORIZON * ARM_JOINTS  # 70
WRENCH_DIM    = 6
DEFAULT_SAMPLES = 500_000


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(
    config_path="../humanoidverse/config",
    config_name="base",
    version_base="1.1",
)
def main(config: OmegaConf):
    # ------------------------------------------------------------------
    # Logging (mirrors train_agent.py)
    # ------------------------------------------------------------------
    hydra_log_path = os.path.join(
        HydraConfig.get().runtime.output_dir, "collect_wrench.log"
    )
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)

    os.chdir(hydra.utils.get_original_cwd())

    # ------------------------------------------------------------------
    # Simulator-specific imports (IsaacGym must be imported before torch)
    # ------------------------------------------------------------------
    simulator_type = config.simulator["_target_"].split(".")[-1]
    if simulator_type == "IsaacGym":
        import isaacgym  # noqa: F401

    import torch
    from utils.common import seeding
    from humanoidverse.utils.helpers import pre_process_config

    # ------------------------------------------------------------------
    # Resolve collection-specific config keys
    # ------------------------------------------------------------------
    checkpoint_path = config.get("checkpoint", None)
    output_path     = config.get("output_path", "data/wrench_data.pt")
    num_samples     = int(config.get("num_samples", DEFAULT_SAMPLES))

    if checkpoint_path is None:
        raise ValueError(
            "No checkpoint specified. "
            "Pass +checkpoint=<path/to/b1.pt> on the command line."
        )

    logger.info(f"B1 checkpoint:  {checkpoint_path}")
    logger.info(f"Output path:    {output_path}")
    logger.info(f"Target samples: {num_samples:,}")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if hasattr(config, "device") and config.device is not None:
        device = config.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # Seeding and config pre-processing
    # ------------------------------------------------------------------
    if config.seed is not None:
        seeding(
            config.seed,
            torch_deterministic=config.get("torch_deterministic", False),
        )

    # Ensure wrench data collection is enabled in the env config
    config.env.config.collect_wrench_data = True
    config.env.config.collect_buffer_size = min(num_samples, 1_000_000)

    pre_process_config(config)

    # ------------------------------------------------------------------
    # Environment (AnticiPoseEnv with wrench collection enabled)
    # ------------------------------------------------------------------
    logger.info("Instantiating environment ...")
    env = instantiate(config=config.env, device=device)
    num_envs = env.num_envs
    logger.info(f"Environment ready. num_envs={num_envs}")

    # ------------------------------------------------------------------
    # Load B1 policy (PPOMultiActorCritic)
    # ------------------------------------------------------------------
    logger.info("Instantiating and loading B1 policy ...")
    algo = instantiate(
        device=device, env=env, config=config.algo, log_dir=None,
    )
    algo.setup()
    algo.load(checkpoint_path)

    # Freeze actors for inference
    for actor in algo.actors.values():
        actor.eval()
        for p in actor.parameters():
            p.requires_grad_(False)

    keys = algo.keys
    logger.info(f"Policy keys: {keys}")

    # ------------------------------------------------------------------
    # Rollout loop
    # ------------------------------------------------------------------
    logger.info("Starting data collection rollouts ...")
    obs_dict = env.reset_all()

    for k in obs_dict:
        obs_dict[k] = obs_dict[k].to(device)

    steps_collected = 0
    log_interval = max(1, num_samples // (num_envs * 20))

    with torch.inference_mode():
        step_idx = 0
        while steps_collected < num_samples:
            # ---- Actor inference ----
            actor_obs = obs_dict["actor_obs"]

            actions_dict = {}
            for key in keys:
                actions_dict[key] = algo.actors[key].act_inference(actor_obs)

            # ---- Environment step ----
            # The AnticiPoseEnv internally:
            #   1. Advances arm trajectory (_pre_physics_step)
            #   2. Computes analytical wrench (_pre_compute_observations_callback)
            #   3. Collects (obs, plan, wrench) data (_post_compute_observations_callback)
            actor_state = {
                "actions": torch.cat(
                    [actions_dict[k] for k in keys], dim=1,
                ),
            }
            obs_dict, _, _, _ = env.step(actor_state)
            for k in obs_dict:
                obs_dict[k] = obs_dict[k].to(device)

            steps_collected += num_envs
            step_idx += 1

            if step_idx % log_interval == 0:
                fill_pct = (
                    100.0 * min(steps_collected, num_samples) / num_samples
                )
                collector_info = (
                    repr(env._wrench_collector)
                    if env._wrench_collector is not None
                    else "N/A"
                )
                logger.info(
                    f"  Step {step_idx:>6}  |  "
                    f"Collected {min(steps_collected, num_samples):>9,} / "
                    f"{num_samples:,} ({fill_pct:.1f}%)  |  {collector_info}"
                )

    # ------------------------------------------------------------------
    # Save dataset
    # ------------------------------------------------------------------
    logger.info(f"Collection complete. Saving to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    env.save_collected_wrench_data(output_path)

    if env._wrench_collector is not None:
        actual_size = len(env._wrench_collector)
        logger.info(
            f"  obs    shape: ({actual_size}, {OBS_DIM})\n"
            f"  plan   shape: ({actual_size}, {PLAN_DIM})\n"
            f"  wrench shape: ({actual_size}, {WRENCH_DIM})"
        )
    logger.info("Done.")


if __name__ == "__main__":
    main()
