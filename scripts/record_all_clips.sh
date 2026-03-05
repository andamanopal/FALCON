#!/bin/bash
# =============================================================================
# Record all video clips for IROS submission video.
#
# Run on the remote GPU VM:
#   cd /workspace/AnticiPose/FALCON
#   bash scripts/record_all_clips.sh
#
# Adjust SEED, TASK, and NUM_STEPS as needed.
# Output goes to /workspace/video_clips/<condition>_<task>_s<seed>/clip.mp4
# =============================================================================

set -euo pipefail

# --- Configuration ---
SEED=42
TASK="frontal_raise"               # Arms raise forward — visually clear
NUM_STEPS=250                      # ~5s at 50fps (onset forced to 0.3s)
WIDTH=1280
HEIGHT=720
FPS=50

EXPERIMENTS_DIR="/workspace/Experiments"
OUTPUT_BASE="/workspace/video_clips"
SCRIPT="scripts/record_video.py"

# Camera angle: front-side view to see arm motion clearly
CAM_POS="2.0 -1.5 1.0"
CAM_TARGET="0.0 0.0 0.7"

# --- Conditions to record ---
# Segment 2: B4b vs B1 (money shot)
# Segment 3: B4a vs B4b vs B4c (core factorial)
# Segment 4: B2 collapse
# Segment 5: B4c catastrophic failure (can reuse B4c from segment 3)
CONDITIONS=(
    "B1_reactive"
    "B2_extended_history"
    "B4a_direct_plan"
    "B4b_direct_plan_critic"
    "B4c_direct_plan_both"
)

# Record both most visual tasks
TASKS=("frontal_raise" "bilateral_asymmetric_lift" "gangnam_style")

mkdir -p "${OUTPUT_BASE}"

echo "=============================================="
echo "  IROS Video Recording"
echo "  Seed:       ${SEED}"
echo "  Tasks:      ${TASKS[*]}"
echo "  Steps:      ${NUM_STEPS} (~$(echo "scale=1; ${NUM_STEPS}/${FPS}" | bc)s)"
echo "  Resolution: ${WIDTH}x${HEIGHT}"
echo "  Conditions: ${#CONDITIONS[@]}"
echo "=============================================="

TOTAL=0
FAILED=0

for TASK_NAME in "${TASKS[@]}"; do
    for COND in "${CONDITIONS[@]}"; do
        CKPT="${EXPERIMENTS_DIR}/SEED_${SEED}/${COND}/model_3000.pt"
        OUT="${OUTPUT_BASE}/${COND}_${TASK_NAME}_s${SEED}"

        # Skip if already recorded
        if [ -f "${OUT}/clip.mp4" ]; then
            echo "[SKIP] ${COND} / ${TASK_NAME} — already exists"
            continue
        fi

        if [ ! -f "${CKPT}" ]; then
            echo "[WARN] Checkpoint not found: ${CKPT}"
            FAILED=$((FAILED + 1))
            continue
        fi

        # Build extra args for B5/B6
        EXTRA_ARGS=""
        if [[ "${COND}" == "B5_anticipose"* ]]; then
            PRED="${EXPERIMENTS_DIR}/SEED_${SEED}/predictors/wrench_predictor_seed${SEED}.pt"
            if [ -f "${PRED}" ]; then
                EXTRA_ARGS="--wrench_predictor_ckpt ${PRED}"
            else
                echo "[WARN] Wrench predictor not found for B5: ${PRED}"
                FAILED=$((FAILED + 1))
                continue
            fi
        fi
        if [[ "${COND}" == "B6_cvae"* ]]; then
            CVAE="${EXPERIMENTS_DIR}/SEED_${SEED}/predictors/arm_plan_cvae_seed${SEED}.pt"
            if [ -f "${CVAE}" ]; then
                EXTRA_ARGS="--cvae_ckpt ${CVAE}"
            else
                echo "[WARN] CVAE ckpt not found for B6: ${CVAE}"
                FAILED=$((FAILED + 1))
                continue
            fi
        fi

        echo ""
        echo ">>> Recording: ${COND} / ${TASK_NAME} / seed ${SEED}"
        echo "    Checkpoint: ${CKPT}"
        echo "    Output:     ${OUT}"

        python ${SCRIPT} \
            --checkpoint "${CKPT}" \
            --task "${TASK_NAME}" \
            --output_dir "${OUT}" \
            --num_steps ${NUM_STEPS} \
            --seed ${SEED} \
            --width ${WIDTH} \
            --height ${HEIGHT} \
            --cam_pos ${CAM_POS} \
            --cam_target ${CAM_TARGET} \
            --fps ${FPS} \
            ${EXTRA_ARGS}

        if [ $? -eq 0 ]; then
            TOTAL=$((TOTAL + 1))
            SIZE=$(du -h "${OUT}/clip.mp4" 2>/dev/null | cut -f1)
            echo "    [OK] ${SIZE}"
        else
            FAILED=$((FAILED + 1))
            echo "    [FAIL]"
        fi
    done
done

echo ""
echo "=============================================="
echo "  Recording Complete"
echo "  Recorded: ${TOTAL}  Failed: ${FAILED}"
echo "  Output:   ${OUTPUT_BASE}/"
echo "=============================================="
echo ""
echo "Clips recorded:"
ls -lh ${OUTPUT_BASE}/*/clip.mp4 2>/dev/null || echo "  (none)"
echo ""
echo "Next: download clips to your Mac with download_clips.sh"
