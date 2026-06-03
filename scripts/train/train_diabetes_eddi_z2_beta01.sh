#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-eddi_z2_beta01_diabetes_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/conditional_vae_eddi_z2_beta01/${RUN_NAME}}"
CONFIG="${CONFIG:-configs/train/train_conditional_vae_eddi_z2_beta01_diabetes.yaml}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
DISEASE="${DISEASE:-diabetes}"
GPU="${GPU:-1}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
STEPS="${STEPS:-2000}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
BETA="${BETA:-0.01}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
VALIDATE_EVERY="${VALIDATE_EVERY:-10}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/${DISEASE}"
LOG_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/logs"
LOG_PATH="${LOG_DIR}/${DISEASE}_eddi_z2_beta01.log"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export PYTHONUNBUFFERED

mkdir -p "${LOG_DIR}"

echo "==> Training diabetes EDDI CVAE z2 beta01"
echo "==> run=${RUN_NAME}"
echo "==> dataset=${DATASET_NAME}; disease=${DISEASE}"
echo "==> GPU=${GPU}"
echo "==> batch_size=${BATCH_SIZE}"
echo "==> lr=${LEARNING_RATE}; beta=${BETA}"
echo "==> output=${OUTPUT_DIR}"
echo "==> log=${LOG_PATH}"

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/train/train_conditional_vae.py \
  --config "${CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --target-group "${DISEASE}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --beta "${BETA}" \
  --num-workers "${NUM_WORKERS}" \
  --log-every "${LOG_EVERY}" \
  --validate-every "${VALIDATE_EVERY}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  > "${LOG_PATH}" 2>&1

echo "==> Done. Output: ${OUTPUT_DIR}"
