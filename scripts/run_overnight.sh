#!/usr/bin/env bash
# run_overnight.sh -- AnticiPose overnight experiment (~8-10 h, 1 GPU)
#
# Pipeline:
#   Stage 1   B1 reactive, 1 seed, ${NUM_ITERS} iters
#   Stage 1b  Collect wrench data using trained B1 checkpoint
#   Stage 2   Train wrench predictor (supervised, offline)
#   Stage 3   B2 oracle, 1 seed, ${NUM_ITERS} iters (go / no-go gate)
#   Stage 4a  B5 anticipose + frozen predictor, ${NUM_ITERS} iters
#   Stage 4b  B4a direct_plan baseline, ${NUM_ITERS} iters
#   Stage 5   Eval all 4 checkpoints on training + held-out arm tasks
#
# Usage:
#   chmod +x scripts/run_overnight.sh
#   cd FALCON/
#   # 4090 (tight overnight):
#   ./scripts/run_overnight.sh --seed 35 --num-envs 8192 --iters 3000
#   # H100 SXM (full run):
#   ./scripts/run_overnight.sh --seed 35 --num-envs 8192 --iters 10000
#
# Prerequisites:
#   - IsaacGym installed and importable
#   - conda / venv with FALCON dependencies active
#   - Run from the FALCON/ repo root

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI flags)
# ---------------------------------------------------------------------------
SEED=42
NUM_ENVS=4096
NUM_ITERS=10000
BASE_LOG_DIR="logs"
PREDICTOR_BATCH_SIZE=4096
PREDICTOR_EPOCHS=100
PREDICTOR_PATIENCE=10
COLLECT_SAMPLES=500000

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)        SEED="$2";          shift 2 ;;
    --num-envs)    NUM_ENVS="$2";      shift 2 ;;
    --iters)       NUM_ITERS="$2";     shift 2 ;;
    --base-log-dir) BASE_LOG_DIR="$2"; shift 2 ;;
    *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Shared Hydra overrides for all stages
# ---------------------------------------------------------------------------
COMMON_OVERRIDES=(
  "+simulator=isaacgym"
  "+domain_rand=domain_rand_rl_gym"
  "+rewards=dec_loco/reward_dec_loco_stand_height_ma_diff_force"
  "+robot=g1/g1_29dof_waist_fakehand"
  "+terrain=terrain_locomotion_plane"
  "num_envs=${NUM_ENVS}"
  "seed=${SEED}"
  "base_dir=${BASE_LOG_DIR}"
  "use_wandb=True"
  "+opt=wandb"
  "wandb.wandb_entity=andaman-l"
  "wandb.wandb_project=AnticiPose"
)

# Base obs config (B1 reactive -- no extra obs)
OBS_BASE="+obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma"
# Per-mode obs configs
OBS_ORACLE="+obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma_oracle"
OBS_ANTICIPOSE="+obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma_anticipose"
OBS_DIRECT_PLAN="+obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma_direct_plan"

TRAIN_CMD="python humanoidverse/train_agent.py +exp=anticipose"

log_stage() {
  echo ""
  echo "========================================================================"
  echo "  $1"
  echo "  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "========================================================================"
}

# Re-log TensorBoard events to WandB for a completed training run.
# WandB's sync_tensorboard=True often stops mid-run; this reads the
# TB events file directly and logs all data to a fresh WandB run.
sync_wandb() {
  local run_dir="$1"
  local run_name="$2"
  echo "[wandb] Re-logging TensorBoard data from ${run_dir}"
  python - "${run_dir}" "${run_name}" <<'PYEOF'
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import wandb

run_dir = sys.argv[1]
run_name = sys.argv[2]

ea = EventAccumulator(run_dir)
ea.Reload()
tags = ea.Tags().get("scalars", [])
if not tags:
    print(f"[wandb] No scalar tags found in {run_dir}")
    sys.exit(0)

step_data = {}
for tag in tags:
    for event in ea.Scalars(tag):
        step = event.step
        if step not in step_data:
            step_data[step] = {}
        step_data[step][tag] = event.value

print(f"[wandb] Found {len(tags)} tags, {len(step_data)} steps")
run = wandb.init(
    project="AnticiPose",
    entity="andaman-l",
    name=run_name,
    tags=["resync"],
)
for step in sorted(step_data.keys()):
    wandb.log(step_data[step], step=step)
wandb.finish()
print("[wandb] Sync complete")
PYEOF
}

# ---------------------------------------------------------------------------
# Stage 1: B1 reactive (~2.5 h)
# ---------------------------------------------------------------------------
log_stage "Stage 1/5 -- B1 reactive (${NUM_ITERS} iters)"

${TRAIN_CMD} \
  "${COMMON_OVERRIDES[@]}" \
  "${OBS_BASE}" \
  "project_name=anticipose_overnight" \
  "experiment_name=B1_reactive_seed${SEED}" \
  "env.config.anticipose_mode=reactive" \
  "algo.config.num_learning_iterations=${NUM_ITERS}"

# Locate the checkpoint written by the run above.
B1_RUN_DIR=$(ls -td "${BASE_LOG_DIR}/anticipose_overnight/"*"B1_reactive_seed${SEED}"* | head -1)
echo "[Stage 1] Run dir: ${B1_RUN_DIR}"

sync_wandb "${B1_RUN_DIR}" "B1_reactive_seed${SEED}"

B1_CHECKPOINT="${B1_RUN_DIR}/model_${NUM_ITERS}.pt"

if [[ ! -f "${B1_CHECKPOINT}" ]]; then
  echo "[ERROR] B1 checkpoint not found at ${B1_CHECKPOINT}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Stage 1b: Collect wrench data using B1 checkpoint (~15 min)
# ---------------------------------------------------------------------------
log_stage "Stage 1b -- Collect wrench supervision data (${COLLECT_SAMPLES} samples)"

WRENCH_DATA_PATH="${BASE_LOG_DIR}/anticipose_overnight/wrench_data_seed${SEED}.pt"

python scripts/collect_wrench_data.py \
  +exp=anticipose \
  "${COMMON_OVERRIDES[@]}" \
  "${OBS_BASE}" \
  "project_name=anticipose_overnight" \
  "experiment_name=collect_wrench_seed${SEED}" \
  "env.config.anticipose_mode=reactive" \
  "env.config.collect_wrench_data=true" \
  "env.config.collect_buffer_size=${COLLECT_SAMPLES}" \
  "checkpoint=${B1_CHECKPOINT}" \
  "+output_path=${WRENCH_DATA_PATH}" \
  "+num_samples=${COLLECT_SAMPLES}" \
  "headless=true"

if [[ ! -f "${WRENCH_DATA_PATH}" ]]; then
  echo "[ERROR] Wrench data not found at ${WRENCH_DATA_PATH}"
  exit 1
fi
echo "[Stage 1b] Wrench data saved to: ${WRENCH_DATA_PATH}"

# ---------------------------------------------------------------------------
# Stage 2: Train wrench predictor (offline supervised) (~30 min)
# ---------------------------------------------------------------------------
log_stage "Stage 2/5 -- Train wrench predictor on collected data"

PREDICTOR_CKPT="${BASE_LOG_DIR}/anticipose_overnight/wrench_predictor_seed${SEED}.pt"

python scripts/train_wrench_predictor.py \
  --data_path "${WRENCH_DATA_PATH}" \
  --save_path "${PREDICTOR_CKPT}" \
  --epochs "${PREDICTOR_EPOCHS}" \
  --batch_size "${PREDICTOR_BATCH_SIZE}" \
  --patience "${PREDICTOR_PATIENCE}" \
  --device cuda

if [[ ! -f "${PREDICTOR_CKPT}" ]]; then
  echo "[ERROR] Wrench predictor checkpoint not found at ${PREDICTOR_CKPT}"
  exit 1
fi
echo "[Stage 2] Predictor saved to: ${PREDICTOR_CKPT}"

# ---------------------------------------------------------------------------
# Stage 3: B2 oracle -- go / no-go gate (~2.5 h)
# ---------------------------------------------------------------------------
log_stage "Stage 3/5 -- B2 oracle (${NUM_ITERS} iters, go/no-go gate)"

${TRAIN_CMD} \
  "${COMMON_OVERRIDES[@]}" \
  "${OBS_ORACLE}" \
  "project_name=anticipose_overnight" \
  "experiment_name=B2_oracle_seed${SEED}" \
  "env.config.anticipose_mode=oracle" \
  "algo.config.num_learning_iterations=${NUM_ITERS}"

B2_RUN_DIR=$(ls -td "${BASE_LOG_DIR}/anticipose_overnight/"*"B2_oracle_seed${SEED}"* | head -1)
echo "[Stage 3] Run dir: ${B2_RUN_DIR}"
sync_wandb "${B2_RUN_DIR}" "B2_oracle_seed${SEED}"

# ---------------------------------------------------------------------------
# Stage 4a: B5 anticipose (~2.5 h)
# ---------------------------------------------------------------------------
log_stage "Stage 4a/5 -- B5 anticipose (${NUM_ITERS} iters)"

${TRAIN_CMD} \
  "${COMMON_OVERRIDES[@]}" \
  "${OBS_ANTICIPOSE}" \
  "project_name=anticipose_overnight" \
  "experiment_name=B5_anticipose_seed${SEED}" \
  "env.config.anticipose_mode=anticipose" \
  "++env.config.wrench_predictor_ckpt=${PREDICTOR_CKPT}" \
  "algo.config.num_learning_iterations=${NUM_ITERS}"

B5_RUN_DIR=$(ls -td "${BASE_LOG_DIR}/anticipose_overnight/"*"B5_anticipose_seed${SEED}"* | head -1)
sync_wandb "${B5_RUN_DIR}" "B5_anticipose_seed${SEED}"

# ---------------------------------------------------------------------------
# Stage 4b: B4a direct_plan (~2.5 h)
# ---------------------------------------------------------------------------
log_stage "Stage 4b/5 -- B4a direct_plan (${NUM_ITERS} iters)"

${TRAIN_CMD} \
  "${COMMON_OVERRIDES[@]}" \
  "${OBS_DIRECT_PLAN}" \
  "project_name=anticipose_overnight" \
  "experiment_name=B4a_direct_plan_seed${SEED}" \
  "env.config.anticipose_mode=direct_plan" \
  "algo.config.num_learning_iterations=${NUM_ITERS}"

B4A_RUN_DIR=$(ls -td "${BASE_LOG_DIR}/anticipose_overnight/"*"B4a_direct_plan_seed${SEED}"* | head -1)
sync_wandb "${B4A_RUN_DIR}" "B4a_direct_plan_seed${SEED}"

# ---------------------------------------------------------------------------
# Stage 5: Evaluation -- training tasks + held-out arm tasks (500 episodes)
# ---------------------------------------------------------------------------
log_stage "Stage 5/5 -- Evaluation (all 4 baselines, 500 episodes each)"

EVAL_CMD="python scripts/eval_baselines.py"
EVAL_EPISODES=500
EVAL_ENVS=64
EVAL_MAX_S=20

# --- B1 reactive ---
${EVAL_CMD} \
  --checkpoint "${B1_CHECKPOINT}" \
  --eval_name "eval_B1_reactive_train_s${SEED}" \
  --num_episodes ${EVAL_EPISODES} --num_envs ${EVAL_ENVS} \
  --max_episode_length_s ${EVAL_MAX_S} \
  --arm_trajectory_task random

${EVAL_CMD} \
  --checkpoint "${B1_CHECKPOINT}" \
  --eval_name "eval_B1_reactive_heldout_s${SEED}" \
  --num_episodes ${EVAL_EPISODES} --num_envs ${EVAL_ENVS} \
  --max_episode_length_s ${EVAL_MAX_S} \
  --arm_trajectory_task lateral_slam_down

# --- B2 oracle ---
${EVAL_CMD} \
  --checkpoint "${B2_RUN_DIR}/model_${NUM_ITERS}.pt" \
  --eval_name "eval_B2_oracle_train_s${SEED}" \
  --num_episodes ${EVAL_EPISODES} --num_envs ${EVAL_ENVS} \
  --max_episode_length_s ${EVAL_MAX_S} \
  --arm_trajectory_task random

${EVAL_CMD} \
  --checkpoint "${B2_RUN_DIR}/model_${NUM_ITERS}.pt" \
  --eval_name "eval_B2_oracle_heldout_s${SEED}" \
  --num_episodes ${EVAL_EPISODES} --num_envs ${EVAL_ENVS} \
  --max_episode_length_s ${EVAL_MAX_S} \
  --arm_trajectory_task lateral_slam_down

# --- B5 anticipose ---
${EVAL_CMD} \
  --checkpoint "${B5_RUN_DIR}/model_${NUM_ITERS}.pt" \
  --eval_name "eval_B5_anticipose_train_s${SEED}" \
  --num_episodes ${EVAL_EPISODES} --num_envs ${EVAL_ENVS} \
  --max_episode_length_s ${EVAL_MAX_S} \
  --arm_trajectory_task random \
  --wrench_predictor_ckpt "${PREDICTOR_CKPT}"

${EVAL_CMD} \
  --checkpoint "${B5_RUN_DIR}/model_${NUM_ITERS}.pt" \
  --eval_name "eval_B5_anticipose_heldout_s${SEED}" \
  --num_episodes ${EVAL_EPISODES} --num_envs ${EVAL_ENVS} \
  --max_episode_length_s ${EVAL_MAX_S} \
  --arm_trajectory_task lateral_slam_down \
  --wrench_predictor_ckpt "${PREDICTOR_CKPT}"

# --- B4a direct_plan ---
${EVAL_CMD} \
  --checkpoint "${B4A_RUN_DIR}/model_${NUM_ITERS}.pt" \
  --eval_name "eval_B4a_direct_plan_train_s${SEED}" \
  --num_episodes ${EVAL_EPISODES} --num_envs ${EVAL_ENVS} \
  --max_episode_length_s ${EVAL_MAX_S} \
  --arm_trajectory_task random

${EVAL_CMD} \
  --checkpoint "${B4A_RUN_DIR}/model_${NUM_ITERS}.pt" \
  --eval_name "eval_B4a_direct_plan_heldout_s${SEED}" \
  --num_episodes ${EVAL_EPISODES} --num_envs ${EVAL_ENVS} \
  --max_episode_length_s ${EVAL_MAX_S} \
  --arm_trajectory_task lateral_slam_down

log_stage "All stages complete"
echo "Log root : ${BASE_LOG_DIR}/anticipose_overnight/"
echo "Predictor: ${PREDICTOR_CKPT}"
