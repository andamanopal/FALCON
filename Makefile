# ============================================================================
# AnticiPose Makefile
# ============================================================================
#
# Training (launches 4 tmux panes, one per baseline):
#   make train-all SEED=42
#   make train-all SEED=42 NUM_ITERS=3000 NUM_ENVS=8192
#
# Train individual baselines:
#   make train-b1 SEED=42
#   make train-b2 SEED=42
#   make train-b5 SEED=42  (requires predictor trained first)
#   make train-b4a SEED=42
#
# Full sequential pipeline on 1 GPU (B1 -> collect -> predictor -> B2 -> B5 -> B4a -> eval):
#   make train-pipeline SEED=42
#
# Evaluation:
#   make eval-all SEED=35 NUM_EPISODES=500
#   make eval-b1 SEED=35 NUM_EPISODES=500
#
# Sync WandB:
#   make sync-wandb SEED=35
#
# Collect results:
#   make results SEED=35
# ============================================================================

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED          ?= 42
NUM_ENVS      ?= 8192
NUM_ITERS     ?= 3000
NUM_EPISODES  ?= 500
MAX_EP_LEN_S  ?= 20
EVAL_NUM_ENVS ?= 64
OUTPUT_DIR    ?= logs_eval
LOG_DIR       ?= logs
PROJECT       ?= anticipose_overnight
PRED_EPOCHS   ?= 100
PRED_BATCH    ?= 4096
PRED_PATIENCE ?= 10
COLLECT_SAMPLES ?= 500000
VENV            ?= /workspace/AnticiPose/.venv/bin/activate

# Derived paths
PRED_CKPT     = $(LOG_DIR)/$(PROJECT)/wrench_predictor_seed$(SEED).pt
WRENCH_DATA   = $(LOG_DIR)/$(PROJECT)/wrench_data_seed$(SEED).pt

# Training command
TRAIN_CMD = python humanoidverse/train_agent.py +exp=anticipose

# Shared Hydra overrides
COMMON = +simulator=isaacgym \
         +domain_rand=domain_rand_rl_gym \
         +rewards=dec_loco/reward_dec_loco_stand_height_ma_diff_force \
         +robot=g1/g1_29dof_waist_fakehand \
         +terrain=terrain_locomotion_plane \
         num_envs=$(NUM_ENVS) \
         seed=$(SEED) \
         base_dir=$(LOG_DIR) \
         use_wandb=True \
         +opt=wandb \
         wandb.wandb_entity=andaman-l \
         wandb.wandb_project=AnticiPose

# Obs configs per mode
OBS_BASE        = +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma
OBS_ORACLE      = +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma_oracle
OBS_ANTICIPOSE  = +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma_anticipose
OBS_DIRECT_PLAN = +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma_direct_plan

# Eval command
EVAL_CMD = python scripts/eval_baselines.py

# Helper: find latest run dir for a given baseline pattern
find_dir = $(shell ls -td $(LOG_DIR)/$(PROJECT)/*$(1)_seed$(SEED)* 2>/dev/null | head -1)
find_ckpt = $(call find_dir,$(1))/model_$(NUM_ITERS).pt

# ============================================================================
# DEFAULT TARGET
# ============================================================================

.DEFAULT_GOAL := help

.PHONY: help train-all train-b1 train-b2 train-b5 train-b4a train-pipeline
.PHONY: collect-wrench train-predictor
.PHONY: eval-all eval-b1 eval-b2 eval-b5 eval-b4a
.PHONY: sync-wandb results

# ============================================================================
# TRAINING TARGETS
# ============================================================================

## train-all: Launch all 4 baselines in separate tmux sessions
##   NOTE: B5 requires a pre-trained predictor. If PRED_CKPT doesn't exist,
##   B5 session will error and remind you to run `make train-pipeline` instead.
train-all:
	@echo "==> Launching 4 training sessions (seed=$(SEED))"
	@tmux new-session -d -s B1_train \
		'source $(VENV) && \
		 echo "=== B1 reactive (seed=$(SEED)) ===" && \
		 $(TRAIN_CMD) $(COMMON) $(OBS_BASE) \
		   project_name=$(PROJECT) \
		   experiment_name=B1_reactive_seed$(SEED) \
		   env.config.anticipose_mode=reactive \
		   algo.config.num_learning_iterations=$(NUM_ITERS) && \
		 echo "B1 DONE" ; exec bash'
	@tmux new-session -d -s B2_train \
		'source $(VENV) && \
		 echo "=== B2 oracle (seed=$(SEED)) ===" && \
		 $(TRAIN_CMD) $(COMMON) $(OBS_ORACLE) \
		   project_name=$(PROJECT) \
		   experiment_name=B2_oracle_seed$(SEED) \
		   env.config.anticipose_mode=oracle \
		   algo.config.num_learning_iterations=$(NUM_ITERS) && \
		 echo "B2 DONE" ; exec bash'
	@tmux new-session -d -s B4a_train \
		'source $(VENV) && \
		 echo "=== B4a direct_plan (seed=$(SEED)) ===" && \
		 $(TRAIN_CMD) $(COMMON) $(OBS_DIRECT_PLAN) \
		   project_name=$(PROJECT) \
		   experiment_name=B4a_direct_plan_seed$(SEED) \
		   env.config.anticipose_mode=direct_plan \
		   algo.config.num_learning_iterations=$(NUM_ITERS) && \
		 echo "B4a DONE" ; exec bash'
	@if [ -f "$(PRED_CKPT)" ]; then \
	  tmux new-session -d -s B5_train \
		'source $(VENV) && \
		 echo "=== B5 anticipose (seed=$(SEED)) ===" && \
		 $(TRAIN_CMD) $(COMMON) $(OBS_ANTICIPOSE) \
		   project_name=$(PROJECT) \
		   experiment_name=B5_anticipose_seed$(SEED) \
		   env.config.anticipose_mode=anticipose \
		   ++env.config.wrench_predictor_ckpt=$(PRED_CKPT) \
		   algo.config.num_learning_iterations=$(NUM_ITERS) && \
		 echo "B5 DONE" ; exec bash' ; \
	else \
	  echo "WARNING: Predictor not found at $(PRED_CKPT) -- skipping B5" ; \
	  echo "  Run: make train-pipeline SEED=$(SEED)" ; \
	fi
	@echo "==> Sessions created: B1_train, B2_train, B4a_train, B5_train"
	@echo "==> Attach with: tmux attach -t B1_train  (or B2_train, B4a_train, B5_train)"
	@echo "==> List all:    tmux ls"

## train-pipeline: Full sequential pipeline in one tmux session (like run_overnight.sh)
##   B1 -> collect -> predictor -> B2 -> B5 -> B4a -> eval (all sequential, 1 GPU)
train-pipeline:
	@echo "==> Pipeline: B1 -> collect -> predictor -> B2 -> B5 -> B4a -> eval (seed=$(SEED))"
	@tmux kill-session -t pipeline_s$(SEED) 2>/dev/null || true
	@tmux new-session -d -s pipeline_s$(SEED) \
		'source $(VENV) && \
		 echo "=== FULL PIPELINE (seed=$(SEED), $(NUM_ITERS) iters, $(NUM_ENVS) envs) ===" && \
		 echo "" && \
		 echo "[Stage 1/7] Training B1 reactive..." && \
		 $(TRAIN_CMD) $(COMMON) $(OBS_BASE) \
		   project_name=$(PROJECT) \
		   experiment_name=B1_reactive_seed$(SEED) \
		   env.config.anticipose_mode=reactive \
		   algo.config.num_learning_iterations=$(NUM_ITERS) && \
		 B1_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B1_reactive_seed$(SEED)* | head -1) && \
		 echo "[Stage 1] B1 done: $${B1_DIR}" && \
		 python scripts/sync_wandb.py $${B1_DIR} B1_reactive_seed$(SEED) && \
		 echo "" && \
		 echo "[Stage 2/7] Collecting wrench data..." && \
		 python scripts/collect_wrench_data.py \
		   +exp=anticipose $(COMMON) $(OBS_BASE) \
		   project_name=$(PROJECT) \
		   experiment_name=collect_wrench_seed$(SEED) \
		   env.config.anticipose_mode=reactive \
		   env.config.collect_wrench_data=true \
		   env.config.collect_buffer_size=$(COLLECT_SAMPLES) \
		   checkpoint=$${B1_DIR}/model_$(NUM_ITERS).pt \
		   +output_path=$(WRENCH_DATA) \
		   +num_samples=$(COLLECT_SAMPLES) \
		   headless=true && \
		 echo "[Stage 2] Wrench data: $(WRENCH_DATA)" && \
		 echo "" && \
		 echo "[Stage 3/7] Training wrench predictor..." && \
		 python scripts/train_wrench_predictor.py \
		   --data_path $(WRENCH_DATA) \
		   --save_path $(PRED_CKPT) \
		   --epochs $(PRED_EPOCHS) \
		   --batch_size $(PRED_BATCH) \
		   --patience $(PRED_PATIENCE) \
		   --device cuda && \
		 echo "[Stage 3] Predictor: $(PRED_CKPT)" && \
		 echo "" && \
		 echo "[Stage 4/7] Training B2 oracle..." && \
		 $(TRAIN_CMD) $(COMMON) $(OBS_ORACLE) \
		   project_name=$(PROJECT) \
		   experiment_name=B2_oracle_seed$(SEED) \
		   env.config.anticipose_mode=oracle \
		   algo.config.num_learning_iterations=$(NUM_ITERS) && \
		 B2_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B2_oracle_seed$(SEED)* | head -1) && \
		 python scripts/sync_wandb.py $${B2_DIR} B2_oracle_seed$(SEED) && \
		 echo "" && \
		 echo "[Stage 5/7] Training B5 anticipose..." && \
		 $(TRAIN_CMD) $(COMMON) $(OBS_ANTICIPOSE) \
		   project_name=$(PROJECT) \
		   experiment_name=B5_anticipose_seed$(SEED) \
		   env.config.anticipose_mode=anticipose \
		   ++env.config.wrench_predictor_ckpt=$(PRED_CKPT) \
		   algo.config.num_learning_iterations=$(NUM_ITERS) && \
		 B5_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B5_anticipose_seed$(SEED)* | head -1) && \
		 python scripts/sync_wandb.py $${B5_DIR} B5_anticipose_seed$(SEED) && \
		 echo "" && \
		 echo "[Stage 6/7] Training B4a direct_plan..." && \
		 $(TRAIN_CMD) $(COMMON) $(OBS_DIRECT_PLAN) \
		   project_name=$(PROJECT) \
		   experiment_name=B4a_direct_plan_seed$(SEED) \
		   env.config.anticipose_mode=direct_plan \
		   algo.config.num_learning_iterations=$(NUM_ITERS) && \
		 B4A_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B4a_direct_plan_seed$(SEED)* | head -1) && \
		 python scripts/sync_wandb.py $${B4A_DIR} B4a_direct_plan_seed$(SEED) && \
		 echo "" && \
		 echo "[Stage 7/7] Evaluating all baselines ($(NUM_EPISODES) episodes)..." && \
		 python scripts/eval_baselines.py \
		   --checkpoint $${B1_DIR}/model_$(NUM_ITERS).pt \
		   --eval_name eval_B1_reactive_train_s$(SEED) \
		   --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
		   --max_episode_length_s $(MAX_EP_LEN_S) \
		   --arm_trajectory_task random \
		   --output_dir $(OUTPUT_DIR) && \
		 python scripts/eval_baselines.py \
		   --checkpoint $${B1_DIR}/model_$(NUM_ITERS).pt \
		   --eval_name eval_B1_reactive_heldout_s$(SEED) \
		   --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
		   --max_episode_length_s $(MAX_EP_LEN_S) \
		   --arm_trajectory_task lateral_slam_down \
		   --output_dir $(OUTPUT_DIR) && \
		 python scripts/eval_baselines.py \
		   --checkpoint $${B2_DIR}/model_$(NUM_ITERS).pt \
		   --eval_name eval_B2_oracle_train_s$(SEED) \
		   --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
		   --max_episode_length_s $(MAX_EP_LEN_S) \
		   --arm_trajectory_task random \
		   --output_dir $(OUTPUT_DIR) && \
		 python scripts/eval_baselines.py \
		   --checkpoint $${B2_DIR}/model_$(NUM_ITERS).pt \
		   --eval_name eval_B2_oracle_heldout_s$(SEED) \
		   --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
		   --max_episode_length_s $(MAX_EP_LEN_S) \
		   --arm_trajectory_task lateral_slam_down \
		   --output_dir $(OUTPUT_DIR) && \
		 python scripts/eval_baselines.py \
		   --checkpoint $${B5_DIR}/model_$(NUM_ITERS).pt \
		   --eval_name eval_B5_anticipose_train_s$(SEED) \
		   --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
		   --max_episode_length_s $(MAX_EP_LEN_S) \
		   --arm_trajectory_task random \
		   --wrench_predictor_ckpt $(PRED_CKPT) \
		   --output_dir $(OUTPUT_DIR) && \
		 python scripts/eval_baselines.py \
		   --checkpoint $${B5_DIR}/model_$(NUM_ITERS).pt \
		   --eval_name eval_B5_anticipose_heldout_s$(SEED) \
		   --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
		   --max_episode_length_s $(MAX_EP_LEN_S) \
		   --arm_trajectory_task lateral_slam_down \
		   --wrench_predictor_ckpt $(PRED_CKPT) \
		   --output_dir $(OUTPUT_DIR) && \
		 python scripts/eval_baselines.py \
		   --checkpoint $${B4A_DIR}/model_$(NUM_ITERS).pt \
		   --eval_name eval_B4a_direct_plan_train_s$(SEED) \
		   --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
		   --max_episode_length_s $(MAX_EP_LEN_S) \
		   --arm_trajectory_task random \
		   --output_dir $(OUTPUT_DIR) && \
		 python scripts/eval_baselines.py \
		   --checkpoint $${B4A_DIR}/model_$(NUM_ITERS).pt \
		   --eval_name eval_B4a_direct_plan_heldout_s$(SEED) \
		   --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
		   --max_episode_length_s $(MAX_EP_LEN_S) \
		   --arm_trajectory_task lateral_slam_down \
		   --output_dir $(OUTPUT_DIR) && \
		 echo "" && \
		 echo "========================================" && \
		 echo "  ALL DONE (seed=$(SEED))" && \
		 echo "========================================" ; exec bash'
	@echo "==> tmux session pipeline_s$(SEED) created"
	@echo "==> Attach with: tmux attach -t pipeline_s$(SEED)"

## train-b1: Train B1 reactive only
train-b1:
	$(TRAIN_CMD) $(COMMON) $(OBS_BASE) \
	  project_name=$(PROJECT) \
	  experiment_name=B1_reactive_seed$(SEED) \
	  env.config.anticipose_mode=reactive \
	  algo.config.num_learning_iterations=$(NUM_ITERS)

## train-b2: Train B2 oracle only
train-b2:
	$(TRAIN_CMD) $(COMMON) $(OBS_ORACLE) \
	  project_name=$(PROJECT) \
	  experiment_name=B2_oracle_seed$(SEED) \
	  env.config.anticipose_mode=oracle \
	  algo.config.num_learning_iterations=$(NUM_ITERS)

## train-b5: Train B5 anticipose only (requires PRED_CKPT)
train-b5:
	@test -f "$(PRED_CKPT)" || (echo "ERROR: Predictor not found at $(PRED_CKPT). Run make train-pipeline first." && exit 1)
	$(TRAIN_CMD) $(COMMON) $(OBS_ANTICIPOSE) \
	  project_name=$(PROJECT) \
	  experiment_name=B5_anticipose_seed$(SEED) \
	  env.config.anticipose_mode=anticipose \
	  ++env.config.wrench_predictor_ckpt=$(PRED_CKPT) \
	  algo.config.num_learning_iterations=$(NUM_ITERS)

## train-b4a: Train B4a direct_plan only
train-b4a:
	$(TRAIN_CMD) $(COMMON) $(OBS_DIRECT_PLAN) \
	  project_name=$(PROJECT) \
	  experiment_name=B4a_direct_plan_seed$(SEED) \
	  env.config.anticipose_mode=direct_plan \
	  algo.config.num_learning_iterations=$(NUM_ITERS)

## collect-wrench: Collect wrench data from B1 checkpoint
collect-wrench:
	@B1_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B1_reactive_seed$(SEED)* | head -1) && \
	echo "Using B1 checkpoint: $${B1_DIR}/model_$(NUM_ITERS).pt" && \
	python scripts/collect_wrench_data.py \
	  +exp=anticipose $(COMMON) $(OBS_BASE) \
	  project_name=$(PROJECT) \
	  experiment_name=collect_wrench_seed$(SEED) \
	  env.config.anticipose_mode=reactive \
	  env.config.collect_wrench_data=true \
	  env.config.collect_buffer_size=$(COLLECT_SAMPLES) \
	  checkpoint=$${B1_DIR}/model_$(NUM_ITERS).pt \
	  +output_path=$(WRENCH_DATA) \
	  +num_samples=$(COLLECT_SAMPLES) \
	  headless=true

## train-predictor: Train wrench predictor from collected data
train-predictor:
	@test -f "$(WRENCH_DATA)" || (echo "ERROR: Wrench data not found at $(WRENCH_DATA). Run make collect-wrench first." && exit 1)
	python scripts/train_wrench_predictor.py \
	  --data_path $(WRENCH_DATA) \
	  --save_path $(PRED_CKPT) \
	  --epochs $(PRED_EPOCHS) \
	  --batch_size $(PRED_BATCH) \
	  --patience $(PRED_PATIENCE) \
	  --device cuda

# ============================================================================
# EVALUATION TARGETS
# ============================================================================

## eval-all: Evaluate all 4 baselines on train + held-out tasks (8 runs)
eval-all: eval-b1 eval-b2 eval-b5 eval-b4a
	@echo ""
	@echo "========================================"
	@echo "  ALL EVALUATIONS COMPLETE (seed=$(SEED))"
	@echo "========================================"
	@$(MAKE) --no-print-directory results SEED=$(SEED)

## eval-b1: Evaluate B1 reactive (train + held-out)
eval-b1:
	@B1_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B1_reactive_seed$(SEED)* | head -1) && \
	$(EVAL_CMD) \
	  --checkpoint $${B1_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B1_reactive_train_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) \
	  --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task random \
	  --output_dir $(OUTPUT_DIR) && \
	$(EVAL_CMD) \
	  --checkpoint $${B1_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B1_reactive_heldout_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) \
	  --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task lateral_slam_down \
	  --output_dir $(OUTPUT_DIR)

## eval-b2: Evaluate B2 oracle (train + held-out)
eval-b2:
	@B2_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B2_oracle_seed$(SEED)* | head -1) && \
	$(EVAL_CMD) \
	  --checkpoint $${B2_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B2_oracle_train_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) \
	  --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task random \
	  --output_dir $(OUTPUT_DIR) && \
	$(EVAL_CMD) \
	  --checkpoint $${B2_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B2_oracle_heldout_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) \
	  --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task lateral_slam_down \
	  --output_dir $(OUTPUT_DIR)

## eval-b5: Evaluate B5 anticipose (train + held-out)
eval-b5:
	@B5_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B5_anticipose_seed$(SEED)* | head -1) && \
	$(EVAL_CMD) \
	  --checkpoint $${B5_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B5_anticipose_train_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) \
	  --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task random \
	  --wrench_predictor_ckpt $(PRED_CKPT) \
	  --output_dir $(OUTPUT_DIR) && \
	$(EVAL_CMD) \
	  --checkpoint $${B5_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B5_anticipose_heldout_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) \
	  --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task lateral_slam_down \
	  --wrench_predictor_ckpt $(PRED_CKPT) \
	  --output_dir $(OUTPUT_DIR)

## eval-b4a: Evaluate B4a direct_plan (train + held-out)
eval-b4a:
	@B4A_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B4a_direct_plan_seed$(SEED)* | head -1) && \
	$(EVAL_CMD) \
	  --checkpoint $${B4A_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B4a_direct_plan_train_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) \
	  --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task random \
	  --output_dir $(OUTPUT_DIR) && \
	$(EVAL_CMD) \
	  --checkpoint $${B4A_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B4a_direct_plan_heldout_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) \
	  --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task lateral_slam_down \
	  --output_dir $(OUTPUT_DIR)

# ============================================================================
# UTILITIES
# ============================================================================

## sync-wandb: Re-log TensorBoard data to WandB for all baselines of a seed
sync-wandb:
	@for mode in B1_reactive B2_oracle B5_anticipose B4a_direct_plan; do \
	  DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*$${mode}_seed$(SEED)* 2>/dev/null | head -1) ; \
	  if [ -n "$$DIR" ]; then \
	    python scripts/sync_wandb.py "$$DIR" "$${mode}_seed$(SEED)" ; \
	  fi ; \
	done

## results: Print all eval results for a seed as a table
results:
	@echo ""
	@echo "=== Eval Results (seed=$(SEED)) ==="
	@echo ""
	@printf "%-35s %12s %12s %12s\n" "Eval Name" "Mean Reward" "Mean EpLen" "Survival%"
	@printf "%-35s %12s %12s %12s\n" "-----------------------------------" "------------" "------------" "------------"
	@for f in $(OUTPUT_DIR)/eval_*_s$(SEED)/results.json; do \
	  if [ -f "$$f" ]; then \
	    python -c " \
import json, sys; \
d = json.load(open('$$f')); \
print(f\"{d['eval_name']:<35s} {d['mean_reward']:>12.2f} {d['mean_episode_length']:>12.1f} {d['survival_rate']*100:>11.1f}%\")" ; \
	  fi ; \
	done
	@echo ""

## help: Show available targets
help:
	@echo "AnticiPose Makefile"
	@echo ""
	@echo "Usage: make <target> [SEED=N] [NUM_ITERS=N] [NUM_ENVS=N] [NUM_EPISODES=N]"
	@echo ""
	@echo "Variables (with defaults):"
	@echo "  SEED=$(SEED)  NUM_ENVS=$(NUM_ENVS)  NUM_ITERS=$(NUM_ITERS)"
	@echo "  NUM_EPISODES=$(NUM_EPISODES)  MAX_EP_LEN_S=$(MAX_EP_LEN_S)  EVAL_NUM_ENVS=$(EVAL_NUM_ENVS)"
	@echo ""
	@grep -E '^##' Makefile | sed 's/^## /  /'
