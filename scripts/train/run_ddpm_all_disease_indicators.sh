#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/train/train_ddpm_mlp.yaml}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
TARGET_GROUP="${TARGET_GROUP:-all_disease_indicators}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ddpm_mlp_all_disease_indicators/${DATASET_NAME}}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-4}"
STEPS="${STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-4096}"

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/train/train_ddpm_mlp.py \
  --config "${CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --target-group "${TARGET_GROUP}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}"
