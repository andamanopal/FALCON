#!/usr/bin/env bash
# run_all_baselines.sh — AnticiPose full baseline sweep (3 seeds, parallel where safe)
#
# Baselines:
#   B1  reactive     — standard FALCON, no extra obs         (actor: 575)
#   B2  oracle       — ground-truth future wrenches          (actor: 605)
#   B4a direct_plan  — raw arm plan appended                 (actor: 645)
#   B5  anticipose   — predicted wrenches, frozen predictor  (actor: 605)
#
# Pipeline:
#   Phase 0   B1 × 3 seeds (parallel) + wrench data collection per seed  ~2.5 h
#   Phase 1   Train wrench predictor per seed (parallel)                  ~0.5 h
#   Phase 2   B2 × 3 seeds (parallel, go/no-go)                          ~2.5 h
#   Phase 3   B5 × 3 seeds (parallel)                                     ~2.5 h
#   Phase 4   B4a × 3 seeds (parallel)                                    ~2.5 h
#   Phase 5   Eval all baselines × seeds × tasks (parallel per baseline)  ~1 h
#   ─────────────────────────────────────────────────────────────────────────────
#   Total estimate on 1 GPU (sequential phases):                           ~12 h
#   Total estimate on 4 GPUs (one per parallel job, all phases parallel):   ~3 h
#
# Environment variables:
#   GPU_IDS       Comma-separated GPU IDs to cycle across parallel jobs (default: "0")
#   NUM_ENVS      Parallel envs per job                                   (default: 4096)
#   BASE_LOG_DIR  Root directory for run logs                             (default: logs)
#
# Usage:
#   chmod +x scripts/run_all_baselines.sh
#   cd FALCON/
#
#   # Single GPU, sequential phases:
#   ./scripts/run_all_baselines.sh
#
#   # 4 GPUs, assign one GPU per parallel job:
#   GPU_IDS="0,1,2,3" ./scripts/run_all_baselines.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEEDS=(42 123 456)
GPU_IDS="${GPU_IDS:-0}"
NUM_ENVS="${NUM_ENVS:-4096}"
BASE_LOG_DIR="${BASE_LOG_DIR:-logs}"
PROJECT_NAME="anticipose_full"

PREDICTOR_TRAIN_STEPS=50000
PREDICTOR_BATCH_SIZE=1024

# ---------------------------------------------------------------------------
# GPU assignment helper (cycles through available GPU IDs)
# ---------------------------------------------------------------------------
IFS=',' read -ra _GPU_LIST <<< "${GPU_IDS}"
_GPU_COUNT=${#_GPU_LIST[@]}
_GPU_IDX=0

next_gpu() {
  echo "${_GPU_LIST[$_GPU_IDX]}"
  _GPU_IDX=$(( (_GPU_IDX + 1) % _GPU_COUNT ))
}

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
log_phase() {
  echo ""
  echo "========================================================================"
  echo "  $1"
  echo "  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "========================================================================"
}

# ---------------------------------------------------------------------------
# Shared Hydra overrides
# ---------------------------------------------------------------------------
base_overrides() {
  local seed="$1"
  echo \
    "+simulator=isaacgym" \
    "+domain_rand=domain_rand_rl_gym" \
    "+rewards=dec_loco/reward_dec_loco_stand_height_ma_diff_force" \
    "+robot=g1/g1_29dof_waist_fakehand" \
    "+terrain=terrain_locomotion_plane" \
    "+obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma" \
    "num_envs=${NUM_ENVS}" \
    "seed=${seed}" \
    "base_dir=${BASE_LOG_DIR}" \
    "project_name=${PROJECT_NAME}" \
    "algo.config.num_learning_iterations=10000"
}

TRAIN_CMD="python humanoidverse/train_agent.py +exp=anticipose"
EVAL_CMD="python humanoidverse/eval_agent.py +exp=anticipose"

# ---------------------------------------------------------------------------
# Helper: find the most-recent run directory matching a pattern
# ---------------------------------------------------------------------------
find_run_dir() {
  local pattern="$1"
  ls -td "${BASE_LOG_DIR}/${PROJECT_NAME}/"*"${pattern}"* 2>/dev/null | head -1
}

# ---------------------------------------------------------------------------
# Phase 0: B1 reactive × 3 seeds — parallel  (~2.5 h per GPU)
# Timing: seeds run in parallel if multiple GPUs available; otherwise sequential.
# ---------------------------------------------------------------------------
log_phase "Phase 0/5 — B1 reactive × ${#SEEDS[@]} seeds (+ wrench data collection)"

declare -a B1_PIDS=()
declare -A B1_GPU

for seed in "${SEEDS[@]}"; do
  gpu=$(next_gpu)
  B1_GPU[$seed]=$gpu
  CUDA_VISIBLE_DEVICES=$gpu ${TRAIN_CMD} \
    $(base_overrides "$seed") \
    "experiment_name=B1_reactive_seed${seed}" \
    "env.config.anticipose_mode=reactive" \
    "env.config.collect_wrench_data=true" \
    "env.config.collect_buffer_size=1000000" \
    &
  B1_PIDS+=($!)
  echo "[Phase 0] Launched B1 seed=${seed} on GPU ${gpu} (PID ${B1_PIDS[-1]})"
done

for pid in "${B1_PIDS[@]}"; do
  wait "$pid" && echo "[Phase 0] PID $pid finished OK" || { echo "[ERROR] PID $pid failed"; exit 1; }
done
log_phase "Phase 0 complete"

# ---------------------------------------------------------------------------
# Phase 1: Train wrench predictor × 3 seeds — parallel  (~30 min)
# ---------------------------------------------------------------------------
log_phase "Phase 1/5 — Train wrench predictor × ${#SEEDS[@]} seeds"

declare -a PRED_PIDS=()
declare -A PREDICTOR_CKPT

for seed in "${SEEDS[@]}"; do
  B1_RUN_DIR=$(find_run_dir "B1_reactive_seed${seed}")
  if [[ -z "${B1_RUN_DIR}" ]]; then
    echo "[ERROR] Could not find B1 run dir for seed ${seed}"; exit 1
  fi
  echo "[Phase 1] B1 run dir (seed ${seed}): ${B1_RUN_DIR}"

  WRENCH_DATA_DIR="${B1_RUN_DIR}/output/wrench_data"
  PRED_OUT="${BASE_LOG_DIR}/${PROJECT_NAME}/wrench_predictor_seed${seed}.pt"
  PREDICTOR_CKPT[$seed]="${PRED_OUT}"

  python humanoidverse/train_wrench_predictor.py \
    --data-dir "${WRENCH_DATA_DIR}" \
    --output-path "${PRED_OUT}" \
    --train-steps "${PREDICTOR_TRAIN_STEPS}" \
    --batch-size "${PREDICTOR_BATCH_SIZE}" \
    --input-dim 185 \
    --output-dim 30 \
    --hidden-dims "256,256,128" \
    --seed "${seed}" \
    &
  PRED_PIDS+=($!)
  echo "[Phase 1] Launched predictor training seed=${seed} (PID ${PRED_PIDS[-1]})"
done

for pid in "${PRED_PIDS[@]}"; do
  wait "$pid" && echo "[Phase 1] PID $pid finished OK" || { echo "[ERROR] PID $pid failed"; exit 1; }
done
log_phase "Phase 1 complete"

# Verify predictor checkpoints exist
for seed in "${SEEDS[@]}"; do
  if [[ ! -f "${PREDICTOR_CKPT[$seed]}" ]]; then
    echo "[ERROR] Predictor checkpoint missing for seed ${seed}: ${PREDICTOR_CKPT[$seed]}"
    exit 1
  fi
  echo "[Phase 1] Predictor (seed ${seed}): ${PREDICTOR_CKPT[$seed]}"
done

# ---------------------------------------------------------------------------
# Phase 2: B2 oracle × 3 seeds — parallel  (~2.5 h per GPU)
# ---------------------------------------------------------------------------
log_phase "Phase 2/5 — B2 oracle × ${#SEEDS[@]} seeds"

declare -a B2_PIDS=()

for seed in "${SEEDS[@]}"; do
  gpu=$(next_gpu)
  CUDA_VISIBLE_DEVICES=$gpu ${TRAIN_CMD} \
    $(base_overrides "$seed") \
    "experiment_name=B2_oracle_seed${seed}" \
    "env.config.anticipose_mode=oracle" \
    &
  B2_PIDS+=($!)
  echo "[Phase 2] Launched B2 oracle seed=${seed} on GPU ${gpu} (PID ${B2_PIDS[-1]})"
done

for pid in "${B2_PIDS[@]}"; do
  wait "$pid" && echo "[Phase 2] PID $pid finished OK" || { echo "[ERROR] PID $pid failed"; exit 1; }
done
log_phase "Phase 2 complete"

# ---------------------------------------------------------------------------
# Phase 3: B5 anticipose × 3 seeds — parallel  (~2.5 h per GPU)
# ---------------------------------------------------------------------------
log_phase "Phase 3/5 — B5 anticipose × ${#SEEDS[@]} seeds"

declare -a B5_PIDS=()

for seed in "${SEEDS[@]}"; do
  gpu=$(next_gpu)
  CUDA_VISIBLE_DEVICES=$gpu ${TRAIN_CMD} \
    $(base_overrides "$seed") \
    "experiment_name=B5_anticipose_seed${seed}" \
    "env.config.anticipose_mode=anticipose" \
    "++env.config.wrench_predictor_ckpt=${PREDICTOR_CKPT[$seed]}" \
    &
  B5_PIDS+=($!)
  echo "[Phase 3] Launched B5 anticipose seed=${seed} on GPU ${gpu} (PID ${B5_PIDS[-1]})"
done

for pid in "${B5_PIDS[@]}"; do
  wait "$pid" && echo "[Phase 3] PID $pid finished OK" || { echo "[ERROR] PID $pid failed"; exit 1; }
done
log_phase "Phase 3 complete"

# ---------------------------------------------------------------------------
# Phase 4: B4a direct_plan × 3 seeds — parallel  (~2.5 h per GPU)
# ---------------------------------------------------------------------------
log_phase "Phase 4/5 — B4a direct_plan × ${#SEEDS[@]} seeds"

declare -a B4_PIDS=()

for seed in "${SEEDS[@]}"; do
  gpu=$(next_gpu)
  CUDA_VISIBLE_DEVICES=$gpu ${TRAIN_CMD} \
    $(base_overrides "$seed") \
    "experiment_name=B4a_direct_plan_seed${seed}" \
    "env.config.anticipose_mode=direct_plan" \
    &
  B4_PIDS+=($!)
  echo "[Phase 4] Launched B4a direct_plan seed=${seed} on GPU ${gpu} (PID ${B4_PIDS[-1]})"
done

for pid in "${B4_PIDS[@]}"; do
  wait "$pid" && echo "[Phase 4] PID $pid finished OK" || { echo "[ERROR] PID $pid failed"; exit 1; }
done
log_phase "Phase 4 complete"

# ---------------------------------------------------------------------------
# Phase 5: Evaluation — all baselines × 3 seeds × (training + held-out tasks)
# ---------------------------------------------------------------------------
log_phase "Phase 5/5 — Evaluation"

EVAL_OVERRIDES_BASE=(
  "+simulator=isaacgym"
  "+domain_rand=domain_rand_rl_gym"
  "+rewards=dec_loco/reward_dec_loco_stand_height_ma_diff_force"
  "+robot=g1/g1_29dof_waist_fakehand"
  "+terrain=terrain_locomotion_plane"
  "+obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma"
  "num_envs=64"
  "headless=true"
  "auto_load_latest=false"
  "base_dir=${BASE_LOG_DIR}"
  "project_name=${PROJECT_NAME}"
)

declare -a EVAL_PIDS=()

for seed in "${SEEDS[@]}"; do
  # Resolve checkpoint paths
  B1_DIR=$(find_run_dir "B1_reactive_seed${seed}")
  B2_DIR=$(find_run_dir "B2_oracle_seed${seed}")
  B4A_DIR=$(find_run_dir "B4a_direct_plan_seed${seed}")
  B5_DIR=$(find_run_dir "B5_anticipose_seed${seed}")

  for dir_var in B1_DIR B2_DIR B4A_DIR B5_DIR; do
    val="${!dir_var}"
    if [[ -z "$val" ]]; then
      echo "[ERROR] Run dir not found for ${dir_var} seed=${seed}"; exit 1
    fi
  done

  gpu=$(next_gpu)

  # B1 reactive — training tasks
  CUDA_VISIBLE_DEVICES=$gpu ${EVAL_CMD} \
    "${EVAL_OVERRIDES_BASE[@]}" \
    "seed=${seed}" \
    "experiment_name=eval_B1_reactive_train_seed${seed}" \
    "env.config.anticipose_mode=reactive" \
    "env.config.arm_trajectory_task=random" \
    "checkpoint=${B1_DIR}/output/model_10000.pt" \
    &
  EVAL_PIDS+=($!)

  # B1 reactive — held-out tasks
  CUDA_VISIBLE_DEVICES=$gpu ${EVAL_CMD} \
    "${EVAL_OVERRIDES_BASE[@]}" \
    "seed=${seed}" \
    "experiment_name=eval_B1_reactive_heldout_seed${seed}" \
    "env.config.anticipose_mode=reactive" \
    "env.config.arm_trajectory_task=held_out_wave" \
    "checkpoint=${B1_DIR}/output/model_10000.pt" \
    &
  EVAL_PIDS+=($!)

  # B2 oracle — training tasks
  CUDA_VISIBLE_DEVICES=$gpu ${EVAL_CMD} \
    "${EVAL_OVERRIDES_BASE[@]}" \
    "seed=${seed}" \
    "experiment_name=eval_B2_oracle_train_seed${seed}" \
    "env.config.anticipose_mode=oracle" \
    "env.config.arm_trajectory_task=random" \
    "checkpoint=${B2_DIR}/output/model_10000.pt" \
    &
  EVAL_PIDS+=($!)

  # B2 oracle — held-out tasks
  CUDA_VISIBLE_DEVICES=$gpu ${EVAL_CMD} \
    "${EVAL_OVERRIDES_BASE[@]}" \
    "seed=${seed}" \
    "experiment_name=eval_B2_oracle_heldout_seed${seed}" \
    "env.config.anticipose_mode=oracle" \
    "env.config.arm_trajectory_task=held_out_wave" \
    "checkpoint=${B2_DIR}/output/model_10000.pt" \
    &
  EVAL_PIDS+=($!)

  # B4a direct_plan — training tasks
  CUDA_VISIBLE_DEVICES=$gpu ${EVAL_CMD} \
    "${EVAL_OVERRIDES_BASE[@]}" \
    "seed=${seed}" \
    "experiment_name=eval_B4a_direct_plan_train_seed${seed}" \
    "env.config.anticipose_mode=direct_plan" \
    "env.config.arm_trajectory_task=random" \
    "checkpoint=${B4A_DIR}/output/model_10000.pt" \
    &
  EVAL_PIDS+=($!)

  # B4a direct_plan — held-out tasks
  CUDA_VISIBLE_DEVICES=$gpu ${EVAL_CMD} \
    "${EVAL_OVERRIDES_BASE[@]}" \
    "seed=${seed}" \
    "experiment_name=eval_B4a_direct_plan_heldout_seed${seed}" \
    "env.config.anticipose_mode=direct_plan" \
    "env.config.arm_trajectory_task=held_out_wave" \
    "checkpoint=${B4A_DIR}/output/model_10000.pt" \
    &
  EVAL_PIDS+=($!)

  # B5 anticipose — training tasks
  CUDA_VISIBLE_DEVICES=$gpu ${EVAL_CMD} \
    "${EVAL_OVERRIDES_BASE[@]}" \
    "seed=${seed}" \
    "experiment_name=eval_B5_anticipose_train_seed${seed}" \
    "env.config.anticipose_mode=anticipose" \
    "env.config.arm_trajectory_task=random" \
    "++env.config.wrench_predictor_ckpt=${PREDICTOR_CKPT[$seed]}" \
    "checkpoint=${B5_DIR}/output/model_10000.pt" \
    &
  EVAL_PIDS+=($!)

  # B5 anticipose — held-out tasks
  CUDA_VISIBLE_DEVICES=$gpu ${EVAL_CMD} \
    "${EVAL_OVERRIDES_BASE[@]}" \
    "seed=${seed}" \
    "experiment_name=eval_B5_anticipose_heldout_seed${seed}" \
    "env.config.anticipose_mode=anticipose" \
    "env.config.arm_trajectory_task=held_out_wave" \
    "++env.config.wrench_predictor_ckpt=${PREDICTOR_CKPT[$seed]}" \
    "checkpoint=${B5_DIR}/output/model_10000.pt" \
    &
  EVAL_PIDS+=($!)
done

for pid in "${EVAL_PIDS[@]}"; do
  wait "$pid" && echo "[Phase 5] PID $pid finished OK" || { echo "[ERROR] eval PID $pid failed"; exit 1; }
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log_phase "All phases complete"
echo ""
echo "Results under: ${BASE_LOG_DIR}/${PROJECT_NAME}/"
echo ""
echo "Baseline checkpoints:"
for seed in "${SEEDS[@]}"; do
  echo "  Seed ${seed}:"
  echo "    B1  reactive  : $(find_run_dir "B1_reactive_seed${seed}")/output/model_10000.pt"
  echo "    B2  oracle    : $(find_run_dir "B2_oracle_seed${seed}")/output/model_10000.pt"
  echo "    B4a dir_plan  : $(find_run_dir "B4a_direct_plan_seed${seed}")/output/model_10000.pt"
  echo "    B5  anticipose: $(find_run_dir "B5_anticipose_seed${seed}")/output/model_10000.pt"
  echo "    Predictor     : ${PREDICTOR_CKPT[$seed]}"
done
echo ""
echo "Timing estimates (single GPU, sequential phases):"
echo "  Phase 0 B1 training × 3 seeds :  ~7.5 h (sequential) | ~2.5 h (3 GPUs)"
echo "  Phase 1 Predictor training × 3 : ~0.5 h (parallel)"
echo "  Phase 2 B2 training × 3 seeds :  ~7.5 h (sequential) | ~2.5 h (3 GPUs)"
echo "  Phase 3 B5 training × 3 seeds :  ~7.5 h (sequential) | ~2.5 h (3 GPUs)"
echo "  Phase 4 B4a training × 3 seeds : ~7.5 h (sequential) | ~2.5 h (3 GPUs)"
echo "  Phase 5 Evaluation             :  ~1 h (parallel, light)"
echo "  ────────────────────────────────────────────────────────"
echo "  Total (1 GPU)  : ~32 h   |   Total (4 GPUs): ~10 h"
