#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-eddi_toward_dobule_decent_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/conditional_vae_eddi_toward_dobule_decent/${RUN_NAME}}"
CONFIG="${CONFIG:-configs/train/train_conditional_vae_eddi_toward_dobule_decent.yaml}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
DISEASE="${DISEASE:-diabetes}"
GPU="${1:-${GPU:-0}}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/${DISEASE}"
LOG_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/logs"
LOG_PATH="${LOG_DIR}/${DISEASE}.log"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export PYTHONUNBUFFERED

nvidia-smi
mkdir -p "${LOG_DIR}"

echo "==> Training diabetes EDDI toward dobule decent"
echo "==> run=${RUN_NAME}"
echo "==> GPU=${GPU}"
echo "==> output=${OUTPUT_DIR}"
echo "==> log=${LOG_PATH}"

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/train/train_conditional_vae.py \
  --config "${CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --target-group "${DISEASE}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  > "${LOG_PATH}" 2>&1

echo "==> Done. Output: ${OUTPUT_DIR}"
