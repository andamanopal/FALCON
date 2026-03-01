# ============================================================================
# AnticiPose Makefile — Single source of truth for all experiments
# ============================================================================
#
# Baselines (ablation ladder):
#   B1   Reactive (FALCON)       — no extra obs
#   B2   Extended History         — 10-step obs history (vs 5)
#   B3   Current Wrench           — 6-dim current wrench in actor obs
#   B4a  Direct Plan (Actor)      — 70-dim arm plan in actor obs
#   B4b  Direct Plan (Critic)     — 70-dim arm plan in critic only
#   B5   AnticiPose (ours)        — 30-dim predicted future wrench
#   B6   CVAE Latent              — 30-dim CVAE latent encoding of arm plan
#
# Quick iteration (1 seed, sequential):
#   make train-pipeline SEED=42
#   make train-pipeline SEED=42 NUM_ITERS=3000 NUM_ENVS=8192
#
# Individual targets:
#   make train-b1 SEED=42
#   make train-b5 SEED=42
#   make eval-all SEED=42
#
# Multi-seed final run:
#   for S in 42 123 456 789 35; do make train-pipeline SEED=$S; done
# ============================================================================

# ---------------------------------------------------------------------------
# Config (override on command line: make train-pipeline SEED=35 NUM_ITERS=5000)
# ---------------------------------------------------------------------------
SEED            ?= 42
NUM_ENVS        ?= 8192
NUM_ITERS       ?= 3000
NUM_EPISODES    ?= 500
MAX_EP_LEN_S    ?= 20
EVAL_NUM_ENVS   ?= 64
OUTPUT_DIR      ?= logs_eval
LOG_DIR         ?= logs
PROJECT         ?= anticipose_overnight
# Predictor/CVAE training (increase PRED_EPOCHS for longer training to
# improve R²; the default 100 may bottleneck before convergence).
PRED_EPOCHS     ?= 100
PRED_V2_EPOCHS  ?= 500
PRED_BATCH      ?= 4096
PRED_PATIENCE   ?= 10
PRED_V2_PATIENCE ?= 20
WANDB_ENTITY    ?= andaman-l
WANDB_PROJECT   ?= AnticiPose
COLLECT_SAMPLES ?= 500000
VENV            ?= /workspace/AnticiPose/.venv/bin/activate
EVAL_EXTRA_ARGS ?=

# Derived paths (overridable for quick iteration, e.g. reusing existing data)
WRENCH_DATA       ?= $(LOG_DIR)/$(PROJECT)/wrench_data_seed$(SEED).pt
PRED_CKPT         ?= $(LOG_DIR)/$(PROJECT)/wrench_predictor_seed$(SEED).pt
CVAE_CKPT         ?= $(LOG_DIR)/$(PROJECT)/arm_plan_cvae_seed$(SEED).pt
PIPELINE_PROGRESS  = $(LOG_DIR)/$(PROJECT)/pipeline_progress_seed$(SEED).txt

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

# Obs configs per baseline
OBS_BASE          = +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma
OBS_EXTENDED_HIST = +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma_extended_history
OBS_CURRENT_WR    = +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma_current_wrench
OBS_DIRECT_PLAN   = +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma_direct_plan
OBS_DIRECT_CRIT   = +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma_direct_plan_critic
OBS_ANTICIPOSE    = +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma_anticipose
OBS_ANTICIPOSE_CW = +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma_anticipose_current_wrench
OBS_ANTICIPOSE_CW_DELTA  = +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma_anticipose_cw_delta
OBS_ANTICIPOSE_CW_H1     = +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma_anticipose_cw_h1
OBS_ANTICIPOSE_CW_H1D    = +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma_anticipose_cw_h1_delta
OBS_CVAE          = +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma_cvae

# Eval command
EVAL_CMD = python scripts/eval_baselines.py

# Helper: find latest run dir for a given baseline pattern
find_dir = $(shell ls -td $(LOG_DIR)/$(PROJECT)/*$(1)_seed$(SEED)* 2>/dev/null | head -1)
find_ckpt = $(call find_dir,$(1))/model_$(NUM_ITERS).pt

# ============================================================================
# DEFAULT TARGET
# ============================================================================

.DEFAULT_GOAL := help

.PHONY: help
.PHONY: train-b1 train-b2 train-b3 train-b4a train-b4b train-b5 train-b5c train-b5c-delta train-b5c-h1 train-b5c-h1-delta train-b6
.PHONY: train-pipeline train-pipeline-tmux pipeline-status pipeline-resume
.PHONY: collect-wrench train-predictor eval-predictor train-cvae
.PHONY: collect-wrench-v2 train-predictor-v2 eval-predictor-v2 train-b5c-v2
.PHONY: eval-all eval-b1 eval-b2 eval-b3 eval-b4a eval-b4b eval-b5 eval-b5c eval-b5c-v2 eval-b5c-delta eval-b5c-h1 eval-b5c-h1-delta eval-b6
.PHONY: smoke-test retrain-b5
.PHONY: sync-wandb results

# ============================================================================
# INDIVIDUAL TRAINING TARGETS
# ============================================================================

## train-b1: Train B1 reactive baseline
train-b1:
	$(TRAIN_CMD) $(COMMON) $(OBS_BASE) \
	  project_name=$(PROJECT) \
	  experiment_name=B1_reactive_seed$(SEED) \
	  env.config.anticipose_mode=reactive \
	  algo.config.num_learning_iterations=$(NUM_ITERS)

## train-b2: Train B2 extended history (10-step)
train-b2:
	$(TRAIN_CMD) $(COMMON) $(OBS_EXTENDED_HIST) \
	  project_name=$(PROJECT) \
	  experiment_name=B2_extended_history_seed$(SEED) \
	  env.config.anticipose_mode=reactive \
	  algo.config.num_learning_iterations=$(NUM_ITERS)

## train-b3: Train B3 current wrench
train-b3:
	$(TRAIN_CMD) $(COMMON) $(OBS_CURRENT_WR) \
	  project_name=$(PROJECT) \
	  experiment_name=B3_current_wrench_seed$(SEED) \
	  env.config.anticipose_mode=reactive \
	  algo.config.num_learning_iterations=$(NUM_ITERS)

## train-b4a: Train B4a direct plan (actor)
train-b4a:
	$(TRAIN_CMD) $(COMMON) $(OBS_DIRECT_PLAN) \
	  project_name=$(PROJECT) \
	  experiment_name=B4a_direct_plan_seed$(SEED) \
	  env.config.anticipose_mode=direct_plan \
	  algo.config.num_learning_iterations=$(NUM_ITERS)

## train-b4b: Train B4b direct plan (critic only)
train-b4b:
	$(TRAIN_CMD) $(COMMON) $(OBS_DIRECT_CRIT) \
	  project_name=$(PROJECT) \
	  experiment_name=B4b_direct_plan_critic_seed$(SEED) \
	  env.config.anticipose_mode=direct_plan \
	  algo.config.num_learning_iterations=$(NUM_ITERS)

## train-b5: Train B5 anticipose (requires predictor checkpoint)
train-b5:
	@test -f "$(PRED_CKPT)" || (echo "ERROR: Predictor not found at $(PRED_CKPT). Run: make collect-wrench && make train-predictor" && exit 1)
	$(TRAIN_CMD) $(COMMON) $(OBS_ANTICIPOSE) \
	  project_name=$(PROJECT) \
	  experiment_name=B5_anticipose_seed$(SEED) \
	  env.config.anticipose_mode=anticipose \
	  ++env.config.wrench_predictor_ckpt=$(PRED_CKPT) \
	  algo.config.num_learning_iterations=$(NUM_ITERS)

## train-b5c: Train B5c anticipose + current wrench anchor (requires predictor checkpoint)
train-b5c:
	@test -f "$(PRED_CKPT)" || (echo "ERROR: Predictor not found at $(PRED_CKPT). Run: make collect-wrench && make train-predictor" && exit 1)
	$(TRAIN_CMD) $(COMMON) $(OBS_ANTICIPOSE_CW) \
	  project_name=$(PROJECT) \
	  experiment_name=B5c_anticipose_cw_seed$(SEED) \
	  env.config.anticipose_mode=anticipose \
	  ++env.config.wrench_predictor_ckpt=$(PRED_CKPT) \
	  algo.config.num_learning_iterations=$(NUM_ITERS)

## train-b5c-delta: Train B5c with delta prediction (predicted - current). Graceful degradation to B3.
train-b5c-delta:
	@PRED_V2=$(LOG_DIR)/$(PROJECT)/wrench_predictor_v2_seed$(SEED).pt && \
	test -f "$$PRED_V2" || (echo "ERROR: v2 predictor not found at $$PRED_V2. Run: make train-predictor-v2" && exit 1) && \
	$(TRAIN_CMD) $(COMMON) $(OBS_ANTICIPOSE_CW_DELTA) \
	  project_name=$(PROJECT) \
	  experiment_name=B5c_delta_seed$(SEED) \
	  env.config.anticipose_mode=anticipose \
	  env.config.use_enhanced_predictor_obs=true \
	  ++env.config.wrench_predictor_ckpt=$$PRED_V2 \
	  algo.config.num_learning_iterations=$(NUM_ITERS)

## train-b5c-h1: Train B5c with H=1 only (next-step prediction, 6D). Less noise.
train-b5c-h1:
	@PRED_V2=$(LOG_DIR)/$(PROJECT)/wrench_predictor_v2_seed$(SEED).pt && \
	test -f "$$PRED_V2" || (echo "ERROR: v2 predictor not found at $$PRED_V2. Run: make train-predictor-v2" && exit 1) && \
	$(TRAIN_CMD) $(COMMON) $(OBS_ANTICIPOSE_CW_H1) \
	  project_name=$(PROJECT) \
	  experiment_name=B5c_h1_seed$(SEED) \
	  env.config.anticipose_mode=anticipose \
	  env.config.use_enhanced_predictor_obs=true \
	  ++env.config.wrench_predictor_ckpt=$$PRED_V2 \
	  algo.config.num_learning_iterations=$(NUM_ITERS)

## train-b5c-h1-delta: Train B5c with H=1 delta (both fixes combined). Best candidate.
train-b5c-h1-delta:
	@PRED_V2=$(LOG_DIR)/$(PROJECT)/wrench_predictor_v2_seed$(SEED).pt && \
	test -f "$$PRED_V2" || (echo "ERROR: v2 predictor not found at $$PRED_V2. Run: make train-predictor-v2" && exit 1) && \
	$(TRAIN_CMD) $(COMMON) $(OBS_ANTICIPOSE_CW_H1D) \
	  project_name=$(PROJECT) \
	  experiment_name=B5c_h1_delta_seed$(SEED) \
	  env.config.anticipose_mode=anticipose \
	  env.config.use_enhanced_predictor_obs=true \
	  ++env.config.wrench_predictor_ckpt=$$PRED_V2 \
	  algo.config.num_learning_iterations=$(NUM_ITERS)

## train-b6: Train B6 CVAE latent (requires CVAE checkpoint)
train-b6:
	@test -f "$(CVAE_CKPT)" || (echo "ERROR: CVAE not found at $(CVAE_CKPT). Run: make train-cvae" && exit 1)
	$(TRAIN_CMD) $(COMMON) $(OBS_CVAE) \
	  project_name=$(PROJECT) \
	  experiment_name=B6_cvae_seed$(SEED) \
	  env.config.anticipose_mode=cvae \
	  ++env.config.cvae_ckpt=$(CVAE_CKPT) \
	  algo.config.num_learning_iterations=$(NUM_ITERS)

# ============================================================================
# DATA COLLECTION & PREDICTOR
# ============================================================================

## collect-wrench: Collect wrench supervision data from trained B1 checkpoint
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

## train-predictor: Train wrench predictor MLP from collected data
##   Pass PRED_PLAN_DERIV=1 to enable plan derivative features (vel+acc)
train-predictor:
	@test -f "$(WRENCH_DATA)" || (echo "ERROR: Wrench data not found at $(WRENCH_DATA). Run: make collect-wrench" && exit 1)
	python scripts/train_wrench_predictor.py \
	  --data_path $(WRENCH_DATA) \
	  --save_path $(PRED_CKPT) \
	  --epochs $(PRED_EPOCHS) \
	  --batch_size $(PRED_BATCH) \
	  --patience $(PRED_PATIENCE) \
	  --device cuda \
	  --wandb_entity $(WANDB_ENTITY) \
	  --wandb_project $(WANDB_PROJECT) \
	  --wandb_run_name wrench_pred_seed$(SEED)_ep$(PRED_EPOCHS) \
	  $(if $(PRED_PLAN_DERIV),--use_plan_derivatives,)

## eval-predictor: Evaluate a trained wrench predictor checkpoint (prints R², RMSE, per-component table)
eval-predictor:
	@test -f "$(PRED_CKPT)" || (echo "ERROR: Predictor not found at $(PRED_CKPT). Run: make train-predictor" && exit 1)
	python scripts/train_wrench_predictor.py \
	  --eval_only \
	  --checkpoint $(PRED_CKPT) \
	  --data_path $(WRENCH_DATA) \
	  --batch_size $(PRED_BATCH) \
	  --device cuda \
	  --no_wandb

## train-cvae: Train CVAE arm plan encoder from collected data
train-cvae:
	@test -f "$(WRENCH_DATA)" || (echo "ERROR: Wrench data not found at $(WRENCH_DATA). Run: make collect-wrench" && exit 1)
	python scripts/train_arm_plan_cvae.py \
	  --data_path $(WRENCH_DATA) \
	  --save_path $(CVAE_CKPT) \
	  --epochs $(PRED_EPOCHS) \
	  --batch_size $(PRED_BATCH) \
	  --patience $(PRED_PATIENCE) \
	  --device cuda \
	  --wandb_entity $(WANDB_ENTITY) \
	  --wandb_project $(WANDB_PROJECT) \
	  --wandb_run_name cvae_seed$(SEED)_ep$(PRED_EPOCHS)

# ============================================================================
# ENHANCED PREDICTOR (v2: body-side context, 123D obs)
# ============================================================================

## collect-wrench-v2: Collect wrench data with body-side context (123D obs)
collect-wrench-v2:
	@B1_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B1_reactive_seed$(SEED)* | head -1) && \
	echo "Using B1 checkpoint: $${B1_DIR}/model_$(NUM_ITERS).pt" && \
	python scripts/collect_wrench_data.py \
	  +exp=anticipose $(COMMON) $(OBS_BASE) \
	  project_name=$(PROJECT) \
	  experiment_name=collect_wrench_v2_seed$(SEED) \
	  env.config.anticipose_mode=reactive \
	  env.config.collect_wrench_data=true \
	  env.config.use_enhanced_predictor_obs=true \
	  env.config.collect_buffer_size=$(COLLECT_SAMPLES) \
	  checkpoint=$${B1_DIR}/model_$(NUM_ITERS).pt \
	  +output_path=$(LOG_DIR)/$(PROJECT)/wrench_data_v2_seed$(SEED).pt \
	  +num_samples=$(COLLECT_SAMPLES) \
	  headless=true

## train-predictor-v2: Train wrench predictor on enhanced 123D obs data (500 epochs, patience 20)
train-predictor-v2:
	@test -f "$(LOG_DIR)/$(PROJECT)/wrench_data_v2_seed$(SEED).pt" || \
	  (echo "ERROR: Enhanced wrench data not found. Run: make collect-wrench-v2" && exit 1)
	python scripts/train_wrench_predictor.py \
	  --data_path $(LOG_DIR)/$(PROJECT)/wrench_data_v2_seed$(SEED).pt \
	  --save_path $(LOG_DIR)/$(PROJECT)/wrench_predictor_v2_seed$(SEED).pt \
	  --epochs $(PRED_V2_EPOCHS) \
	  --batch_size $(PRED_BATCH) \
	  --patience $(PRED_V2_PATIENCE) \
	  --device cuda \
	  --wandb_entity $(WANDB_ENTITY) \
	  --wandb_project $(WANDB_PROJECT) \
	  --wandb_run_name wrench_pred_v2_seed$(SEED)_ep$(PRED_V2_EPOCHS) \
	  $(if $(PRED_PLAN_DERIV),--use_plan_derivatives,)

## eval-predictor-v2: Evaluate v2 wrench predictor (123D obs) on matching v2 data
eval-predictor-v2:
	@test -f "$(LOG_DIR)/$(PROJECT)/wrench_predictor_v2_seed$(SEED).pt" || \
	  (echo "ERROR: v2 predictor not found. Run: make train-predictor-v2" && exit 1)
	python scripts/train_wrench_predictor.py \
	  --eval_only \
	  --checkpoint $(LOG_DIR)/$(PROJECT)/wrench_predictor_v2_seed$(SEED).pt \
	  --data_path $(LOG_DIR)/$(PROJECT)/wrench_data_v2_seed$(SEED).pt \
	  --batch_size $(PRED_BATCH) \
	  --device cuda \
	  --no_wandb

## train-b5c-v2: Train B5c with improved predictor (enhanced 123D obs)
train-b5c-v2:
	@PRED_V2=$(LOG_DIR)/$(PROJECT)/wrench_predictor_v2_seed$(SEED).pt && \
	test -f "$$PRED_V2" || (echo "ERROR: v2 predictor not found at $$PRED_V2. Run: make train-predictor-v2" && exit 1) && \
	$(TRAIN_CMD) $(COMMON) $(OBS_ANTICIPOSE_CW) \
	  project_name=$(PROJECT) \
	  experiment_name=B5c_v2_anticipose_cw_seed$(SEED) \
	  env.config.anticipose_mode=anticipose \
	  env.config.use_enhanced_predictor_obs=true \
	  ++env.config.wrench_predictor_ckpt=$$PRED_V2 \
	  algo.config.num_learning_iterations=$(NUM_ITERS)

# ============================================================================
# B5 QUICK ITERATION (skip B1 training, re-use existing wrench data if present)
# ============================================================================

## retrain-b5: Re-collect data, retrain predictor, retrain B5, and eval (~4-7h vs 24h full pipeline)
retrain-b5:
	@echo "=== RETRAIN-B5 (seed=$(SEED), pred_epochs=$(PRED_EPOCHS), collect_samples=$(COLLECT_SAMPLES)) ==="
	$(MAKE) --no-print-directory collect-wrench SEED=$(SEED)
	$(MAKE) --no-print-directory train-predictor SEED=$(SEED)
	$(MAKE) --no-print-directory train-b5 SEED=$(SEED)
	$(MAKE) --no-print-directory eval-b5 SEED=$(SEED)
	@echo ""
	@echo "========================================"
	@echo "  RETRAIN-B5 COMPLETE (seed=$(SEED))"
	@echo "========================================"
	@$(MAKE) --no-print-directory results SEED=$(SEED)

# ============================================================================
# FULL SEQUENTIAL PIPELINE (1 GPU, overnight)
# ============================================================================

PIPELINE_TOTAL = 11

# Shell helper: maps step number to "target label"
# Used by train-pipeline, pipeline-resume, and pipeline-status
define PIPELINE_STEP_FUNC
step_target() { \
	case $$1 in \
		1)  echo "train-b1";; \
		2)  echo "collect-wrench";; \
		3)  echo "train-predictor";; \
		4)  echo "train-cvae";; \
		5)  echo "train-b2";; \
		6)  echo "train-b3";; \
		7)  echo "train-b4a";; \
		8)  echo "train-b4b";; \
		9)  echo "train-b5";; \
		10) echo "train-b6";; \
		11) echo "eval-all";; \
	esac; \
}; \
step_label() { \
	case $$1 in \
		1)  echo "B1 reactive";; \
		2)  echo "Collect wrench data";; \
		3)  echo "Wrench predictor";; \
		4)  echo "CVAE encoder";; \
		5)  echo "B2 extended history";; \
		6)  echo "B3 current wrench";; \
		7)  echo "B4a direct plan";; \
		8)  echo "B4b direct plan critic";; \
		9)  echo "B5 anticipose";; \
		10) echo "B6 CVAE latent";; \
		11) echo "Evaluate all";; \
	esac; \
}; \
step_wandb_name() { \
	case $$1 in \
		1)  echo "B1_reactive";; \
		5)  echo "B2_extended_history";; \
		6)  echo "B3_current_wrench";; \
		7)  echo "B4a_direct_plan";; \
		8)  echo "B4b_direct_plan_critic";; \
		9)  echo "B5_anticipose";; \
		10) echo "B6_cvae";; \
		*)  echo "";; \
	esac; \
}
endef

## train-pipeline: Run full pipeline sequentially (B1 -> collect -> predictor/cvae -> B2-B6 -> eval)
train-pipeline:
	@$(PIPELINE_STEP_FUNC); \
	echo "=== FULL PIPELINE (seed=$(SEED), $(NUM_ITERS) iters, $(NUM_ENVS) envs) ==="; \
	mkdir -p $$(dirname $(PIPELINE_PROGRESS)); \
	echo 0 > $(PIPELINE_PROGRESS); \
	STEP=1; \
	while [ "$$STEP" -le $(PIPELINE_TOTAL) ]; do \
		TARGET=$$(step_target $$STEP); \
		LABEL=$$(step_label $$STEP); \
		echo ""; \
		echo "[$$STEP/$(PIPELINE_TOTAL)] $$LABEL..."; \
		$(MAKE) --no-print-directory $$TARGET SEED=$(SEED) || exit 1; \
		WANDB_NAME=$$(step_wandb_name $$STEP); \
		if [ -n "$$WANDB_NAME" ]; then \
			RUN_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*$${WANDB_NAME}_seed$(SEED)* 2>/dev/null | head -1); \
			if [ -n "$$RUN_DIR" ]; then \
				echo "[wandb] Re-syncing $$WANDB_NAME from $$RUN_DIR"; \
				python scripts/sync_wandb.py "$$RUN_DIR" "$${WANDB_NAME}_seed$(SEED)" || echo "[wandb] Sync failed (non-fatal)"; \
			fi; \
		fi; \
		echo $$STEP > $(PIPELINE_PROGRESS); \
		STEP=$$((STEP + 1)); \
	done; \
	echo ""; \
	echo "========================================"; \
	echo "  ALL DONE (seed=$(SEED))"; \
	echo "========================================"; \
	rm -f $(PIPELINE_PROGRESS)

## pipeline-resume: Resume pipeline from last completed step
pipeline-resume:
	@$(PIPELINE_STEP_FUNC); \
	if [ -f "$(PIPELINE_PROGRESS)" ]; then \
		DONE=$$(cat $(PIPELINE_PROGRESS)); \
		echo "=== RESUMING PIPELINE (seed=$(SEED), last completed: step $$DONE/$(PIPELINE_TOTAL)) ==="; \
	else \
		DONE=0; \
		echo "=== No progress file found — starting full pipeline (seed=$(SEED)) ==="; \
	fi; \
	mkdir -p $$(dirname $(PIPELINE_PROGRESS)); \
	if [ ! -f "$(PIPELINE_PROGRESS)" ]; then echo 0 > $(PIPELINE_PROGRESS); fi; \
	STEP=1; \
	while [ "$$STEP" -le $(PIPELINE_TOTAL) ]; do \
		TARGET=$$(step_target $$STEP); \
		LABEL=$$(step_label $$STEP); \
		if [ "$$STEP" -le "$$DONE" ]; then \
			echo "[$$STEP/$(PIPELINE_TOTAL)] $$LABEL — skipped (already done)"; \
			STEP=$$((STEP + 1)); \
			continue; \
		fi; \
		echo ""; \
		echo "[$$STEP/$(PIPELINE_TOTAL)] $$LABEL..."; \
		$(MAKE) --no-print-directory $$TARGET SEED=$(SEED) || exit 1; \
		WANDB_NAME=$$(step_wandb_name $$STEP); \
		if [ -n "$$WANDB_NAME" ]; then \
			RUN_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*$${WANDB_NAME}_seed$(SEED)* 2>/dev/null | head -1); \
			if [ -n "$$RUN_DIR" ]; then \
				echo "[wandb] Re-syncing $$WANDB_NAME from $$RUN_DIR"; \
				python scripts/sync_wandb.py "$$RUN_DIR" "$${WANDB_NAME}_seed$(SEED)" || echo "[wandb] Sync failed (non-fatal)"; \
			fi; \
		fi; \
		echo $$STEP > $(PIPELINE_PROGRESS); \
		STEP=$$((STEP + 1)); \
	done; \
	echo ""; \
	echo "========================================"; \
	echo "  ALL DONE (seed=$(SEED))"; \
	echo "========================================"; \
	rm -f $(PIPELINE_PROGRESS)

## pipeline-status: Show pipeline progress for a seed
pipeline-status:
	@$(PIPELINE_STEP_FUNC); \
	if [ ! -f "$(PIPELINE_PROGRESS)" ]; then \
		echo "No pipeline in progress for seed $(SEED)."; \
		exit 0; \
	fi; \
	DONE=$$(cat $(PIPELINE_PROGRESS)); \
	if [ "$$DONE" -ge $(PIPELINE_TOTAL) ]; then \
		echo "Pipeline complete for seed $(SEED) (all $(PIPELINE_TOTAL) steps done)."; \
		exit 0; \
	fi; \
	DONE_LABEL=$$(step_label $$DONE); \
	NEXT=$$((DONE + 1)); \
	NEXT_LABEL=$$(step_label $$NEXT); \
	echo "Pipeline status (seed=$(SEED)):"; \
	echo "  Last completed: [$$DONE/$(PIPELINE_TOTAL)] $$DONE_LABEL"; \
	echo "  Next step:      [$$NEXT/$(PIPELINE_TOTAL)] $$NEXT_LABEL"; \
	echo ""; \
	echo "Resume with: make pipeline-resume SEED=$(SEED)"

## train-pipeline-tmux: Same as train-pipeline but in a detached tmux session
train-pipeline-tmux:
	@tmux kill-session -t pipeline_s$(SEED) 2>/dev/null || true
	@tmux new-session -d -s pipeline_s$(SEED) \
		'source $(VENV) && make train-pipeline SEED=$(SEED) NUM_ITERS=$(NUM_ITERS) NUM_ENVS=$(NUM_ENVS) 2>&1 | tee logs/pipeline_seed$(SEED).log ; exec bash'
	@echo "==> tmux session: pipeline_s$(SEED)"
	@echo "==> Attach:  tmux attach -t pipeline_s$(SEED)"
	@echo "==> Log:     tail -f logs/pipeline_seed$(SEED).log"

# ============================================================================
# EVALUATION TARGETS
# ============================================================================

## eval-all: Evaluate all baselines on train + held-out tasks (B5c included if trained)
eval-all: eval-b1 eval-b2 eval-b3 eval-b4a eval-b4b eval-b5 eval-b5c eval-b6
	@echo ""
	@echo "========================================"
	@echo "  ALL EVALUATIONS COMPLETE (seed=$(SEED))"
	@echo "========================================"
	@$(MAKE) --no-print-directory results SEED=$(SEED)

## eval-b1: Evaluate B1 reactive
eval-b1:
	@B1_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B1_reactive_seed$(SEED)* 2>/dev/null | head -1) && \
	if [ -z "$$B1_DIR" ]; then echo "SKIP: B1 not found for seed $(SEED)"; exit 0; fi && \
	$(EVAL_CMD) \
	  --checkpoint $${B1_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B1_reactive_train_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task random \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS) && \
	$(EVAL_CMD) \
	  --checkpoint $${B1_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B1_reactive_heldout_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task lateral_slam_down \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS)

## eval-b2: Evaluate B2 extended history
eval-b2:
	@B2_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B2_extended_history_seed$(SEED)* 2>/dev/null | head -1) && \
	if [ -z "$$B2_DIR" ]; then echo "SKIP: B2 not found for seed $(SEED)"; exit 0; fi && \
	$(EVAL_CMD) \
	  --checkpoint $${B2_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B2_extended_history_train_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task random \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS) && \
	$(EVAL_CMD) \
	  --checkpoint $${B2_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B2_extended_history_heldout_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task lateral_slam_down \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS)

## eval-b3: Evaluate B3 current wrench
eval-b3:
	@B3_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B3_current_wrench_seed$(SEED)* 2>/dev/null | head -1) && \
	if [ -z "$$B3_DIR" ]; then echo "SKIP: B3 not found for seed $(SEED)"; exit 0; fi && \
	$(EVAL_CMD) \
	  --checkpoint $${B3_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B3_current_wrench_train_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task random \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS) && \
	$(EVAL_CMD) \
	  --checkpoint $${B3_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B3_current_wrench_heldout_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task lateral_slam_down \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS)

## eval-b4a: Evaluate B4a direct plan (actor)
eval-b4a:
	@B4A_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B4a_direct_plan_seed$(SEED)* 2>/dev/null | head -1) && \
	if [ -z "$$B4A_DIR" ]; then echo "SKIP: B4a not found for seed $(SEED)"; exit 0; fi && \
	$(EVAL_CMD) \
	  --checkpoint $${B4A_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B4a_direct_plan_train_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task random \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS) && \
	$(EVAL_CMD) \
	  --checkpoint $${B4A_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B4a_direct_plan_heldout_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task lateral_slam_down \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS)

## eval-b4b: Evaluate B4b direct plan (critic only)
eval-b4b:
	@B4B_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B4b_direct_plan_critic_seed$(SEED)* 2>/dev/null | head -1) && \
	if [ -z "$$B4B_DIR" ]; then echo "SKIP: B4b not found for seed $(SEED)"; exit 0; fi && \
	$(EVAL_CMD) \
	  --checkpoint $${B4B_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B4b_direct_plan_critic_train_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task random \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS) && \
	$(EVAL_CMD) \
	  --checkpoint $${B4B_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B4b_direct_plan_critic_heldout_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task lateral_slam_down \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS)

## eval-b5: Evaluate B5 anticipose
eval-b5:
	@B5_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B5_anticipose_seed$(SEED)* 2>/dev/null | head -1) && \
	if [ -z "$$B5_DIR" ]; then echo "SKIP: B5 not found for seed $(SEED)"; exit 0; fi && \
	$(EVAL_CMD) \
	  --checkpoint $${B5_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B5_anticipose_train_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task random \
	  --wrench_predictor_ckpt $(PRED_CKPT) \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS) && \
	$(EVAL_CMD) \
	  --checkpoint $${B5_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B5_anticipose_heldout_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task lateral_slam_down \
	  --wrench_predictor_ckpt $(PRED_CKPT) \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS)

## eval-b5c: Evaluate B5c anticipose + current wrench anchor
eval-b5c:
	@B5C_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B5c_anticipose_cw_seed$(SEED)* 2>/dev/null | head -1) && \
	if [ -z "$$B5C_DIR" ]; then echo "SKIP: B5c not found for seed $(SEED)"; exit 0; fi && \
	$(EVAL_CMD) \
	  --checkpoint $${B5C_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B5c_anticipose_cw_train_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task random \
	  --wrench_predictor_ckpt $(PRED_CKPT) \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS) && \
	$(EVAL_CMD) \
	  --checkpoint $${B5C_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B5c_anticipose_cw_heldout_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task lateral_slam_down \
	  --wrench_predictor_ckpt $(PRED_CKPT) \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS)

## eval-b5c-v2: Evaluate B5c with improved predictor (v2, enhanced 123D obs)
eval-b5c-v2:
	@B5C_V2_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B5c_v2_anticipose_cw_seed$(SEED)* 2>/dev/null | head -1) && \
	PRED_V2=$(LOG_DIR)/$(PROJECT)/wrench_predictor_v2_seed$(SEED).pt && \
	if [ -z "$$B5C_V2_DIR" ]; then echo "SKIP: B5c-v2 not found for seed $(SEED)"; exit 0; fi && \
	$(EVAL_CMD) \
	  --checkpoint $${B5C_V2_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B5c_v2_anticipose_cw_train_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task random \
	  --wrench_predictor_ckpt $$PRED_V2 \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS) && \
	$(EVAL_CMD) \
	  --checkpoint $${B5C_V2_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B5c_v2_anticipose_cw_heldout_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task lateral_slam_down \
	  --wrench_predictor_ckpt $$PRED_V2 \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS)

## eval-b5c-delta: Evaluate B5c delta variant
eval-b5c-delta:
	@B5CD_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B5c_delta_seed$(SEED)* 2>/dev/null | head -1) && \
	PRED_V2=$(LOG_DIR)/$(PROJECT)/wrench_predictor_v2_seed$(SEED).pt && \
	if [ -z "$$B5CD_DIR" ]; then echo "SKIP: B5c-delta not found for seed $(SEED)"; exit 0; fi && \
	$(EVAL_CMD) \
	  --checkpoint $${B5CD_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B5c_delta_train_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task random \
	  --wrench_predictor_ckpt $$PRED_V2 \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS) && \
	$(EVAL_CMD) \
	  --checkpoint $${B5CD_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B5c_delta_heldout_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task lateral_slam_down \
	  --wrench_predictor_ckpt $$PRED_V2 \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS)

## eval-b5c-h1: Evaluate B5c H=1 variant
eval-b5c-h1:
	@B5CH1_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B5c_h1_seed$(SEED)* 2>/dev/null | head -1) && \
	PRED_V2=$(LOG_DIR)/$(PROJECT)/wrench_predictor_v2_seed$(SEED).pt && \
	if [ -z "$$B5CH1_DIR" ]; then echo "SKIP: B5c-h1 not found for seed $(SEED)"; exit 0; fi && \
	$(EVAL_CMD) \
	  --checkpoint $${B5CH1_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B5c_h1_train_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task random \
	  --wrench_predictor_ckpt $$PRED_V2 \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS) && \
	$(EVAL_CMD) \
	  --checkpoint $${B5CH1_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B5c_h1_heldout_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task lateral_slam_down \
	  --wrench_predictor_ckpt $$PRED_V2 \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS)

## eval-b5c-h1-delta: Evaluate B5c H=1 delta variant (both fixes combined)
eval-b5c-h1-delta:
	@B5CH1D_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B5c_h1_delta_seed$(SEED)* 2>/dev/null | head -1) && \
	PRED_V2=$(LOG_DIR)/$(PROJECT)/wrench_predictor_v2_seed$(SEED).pt && \
	if [ -z "$$B5CH1D_DIR" ]; then echo "SKIP: B5c-h1-delta not found for seed $(SEED)"; exit 0; fi && \
	$(EVAL_CMD) \
	  --checkpoint $${B5CH1D_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B5c_h1_delta_train_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task random \
	  --wrench_predictor_ckpt $$PRED_V2 \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS) && \
	$(EVAL_CMD) \
	  --checkpoint $${B5CH1D_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B5c_h1_delta_heldout_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task lateral_slam_down \
	  --wrench_predictor_ckpt $$PRED_V2 \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS)

## eval-b6: Evaluate B6 CVAE latent
eval-b6:
	@B6_DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*B6_cvae_seed$(SEED)* 2>/dev/null | head -1) && \
	if [ -z "$$B6_DIR" ]; then echo "SKIP: B6 not found for seed $(SEED)"; exit 0; fi && \
	$(EVAL_CMD) \
	  --checkpoint $${B6_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B6_cvae_train_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task random \
	  --cvae_ckpt $(CVAE_CKPT) \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS) && \
	$(EVAL_CMD) \
	  --checkpoint $${B6_DIR}/model_$(NUM_ITERS).pt \
	  --eval_name eval_B6_cvae_heldout_s$(SEED) \
	  --num_episodes $(NUM_EPISODES) --num_envs $(EVAL_NUM_ENVS) \
	  --max_episode_length_s $(MAX_EP_LEN_S) \
	  --arm_trajectory_task lateral_slam_down \
	  --cvae_ckpt $(CVAE_CKPT) \
	  --output_dir $(OUTPUT_DIR) $(EVAL_EXTRA_ARGS)

# ============================================================================
# UTILITIES
# ============================================================================

## smoke-test: Run full pipeline with minimal settings to catch config/import errors
smoke-test:
	$(MAKE) train-pipeline SEED=$(SEED) \
	  NUM_ITERS=2 NUM_ENVS=4 NUM_EPISODES=2 \
	  EVAL_NUM_ENVS=4 COLLECT_SAMPLES=100 \
	  PRED_EPOCHS=2 PRED_BATCH=32 PRED_PATIENCE=1 \
	  PROJECT=anticipose_smoke_test \
	  EVAL_EXTRA_ARGS="--walking_speeds 0.0 0.6"

## sync-wandb: Re-log TensorBoard data to WandB for all baselines
sync-wandb:
	@for mode in B1_reactive B2_extended_history B3_current_wrench B4a_direct_plan B4b_direct_plan_critic B5_anticipose B6_cvae; do \
	  DIR=$$(ls -td $(LOG_DIR)/$(PROJECT)/*$${mode}_seed$(SEED)* 2>/dev/null | head -1) ; \
	  if [ -n "$$DIR" ]; then \
	    python scripts/sync_wandb.py "$$DIR" "$${mode}_seed$(SEED)" ; \
	  fi ; \
	done

## results: Print eval results table for a seed
results:
	@echo ""
	@echo "=== Eval Results (seed=$(SEED)) ==="
	@echo ""
	@printf "%-45s %12s %12s %12s\n" "Eval Name" "Mean Reward" "Mean EpLen" "Survival%"
	@printf "%-45s %12s %12s %12s\n" "---------------------------------------------" "------------" "------------" "------------"
	@for f in $(OUTPUT_DIR)/eval_*_s$(SEED)*/results.json; do \
	  if [ -f "$$f" ]; then \
	    python -c "import json, sys; d = json.load(open('$$f')); print(f\"{d['eval_name']:<45s} {d['mean_reward']:>12.2f} {d['mean_episode_length']:>12.1f} {d['survival_rate']*100:>11.1f}%\")" ; \
	  fi ; \
	done
	@echo ""

## help: Show available targets and usage
help:
	@echo "AnticiPose Makefile"
	@echo ""
	@echo "Usage: make <target> [SEED=N] [NUM_ITERS=N] [NUM_ENVS=N] [NUM_EPISODES=N]"
	@echo ""
	@echo "Variables (with defaults):"
	@echo "  SEED=$(SEED)  NUM_ENVS=$(NUM_ENVS)  NUM_ITERS=$(NUM_ITERS)"
	@echo "  NUM_EPISODES=$(NUM_EPISODES)  MAX_EP_LEN_S=$(MAX_EP_LEN_S)  EVAL_NUM_ENVS=$(EVAL_NUM_ENVS)"
	@echo "  PRED_EPOCHS=$(PRED_EPOCHS)  PRED_BATCH=$(PRED_BATCH)  PRED_PATIENCE=$(PRED_PATIENCE)"
	@echo "  COLLECT_SAMPLES=$(COLLECT_SAMPLES)  WANDB_ENTITY=$(WANDB_ENTITY)  WANDB_PROJECT=$(WANDB_PROJECT)"
	@echo ""
	@echo "Baselines:"
	@echo "  B1   Reactive (FALCON)         B2   Extended History (10-step)"
	@echo "  B3   Current Wrench            B4a  Direct Plan (Actor)"
	@echo "  B4b  Direct Plan (Critic)      B5   AnticiPose (ours)"
	@echo "  B6   CVAE Latent"
	@echo ""
	@echo "Quick iteration (retrain predictor + B5 only, ~4-7h):"
	@echo "  make retrain-b5 SEED=42 COLLECT_SAMPLES=2000000 PRED_EPOCHS=500 PRED_PATIENCE=20"
	@echo ""
	@echo "Quick verification:"
	@echo "  make smoke-test SEED=42            # Full pipeline, tiny settings (~minutes)"
	@echo ""
	@echo "Pipeline checkpoint/resume:"
	@echo "  make pipeline-status SEED=42    # Check where pipeline left off"
	@echo "  make pipeline-resume SEED=42    # Resume from last completed step"
	@echo ""
	@grep -E '^##' Makefile | sed 's/^## /  /'
