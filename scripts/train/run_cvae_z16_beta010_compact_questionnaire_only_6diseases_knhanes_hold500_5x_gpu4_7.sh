#!/usr/bin/env bash
set -euo pipefail

SWEEP_ID="${SWEEP_ID:-cvae_z16_beta010_compact_questionnaire_only_6diseases_hold500_1x_gpu4_7_$(date +%Y%m%d_%H%M%S)}"

BETAS="${BETAS:-0.1}" \
DISEASES="${DISEASES:-diabetes hypertension dyslipidemia liver_disease kidney_disease anemia}" \
REPEATS="${REPEATS:-1}" \
GPUS="${GPUS:-4 5 6 7}" \
CONFIG="${CONFIG:-configs/train/train_conditional_vae_eddi_z16_beta010_compact_questionnaire_only_beta_hold500.yaml}" \
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}" \
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}" \
BATCH_SIZE="${BATCH_SIZE:-1024}" \
STEPS="${STEPS:-2000}" \
LEARNING_RATE="${LEARNING_RATE:-0.001}" \
BETA_WARMUP_START_STEP="${BETA_WARMUP_START_STEP:-500}" \
BETA_WARMUP_STEPS="${BETA_WARMUP_STEPS:-1000}" \
DISABLE_EARLY_STOPPING="${DISABLE_EARLY_STOPPING:-1}" \
SWEEP_ID="${SWEEP_ID}" \
bash scripts/train/repeat_cvae_z16_beta_selection_6diseases.sh
