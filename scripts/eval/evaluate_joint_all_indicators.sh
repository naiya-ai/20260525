#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-cvae}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to checkpoint_best.pt or checkpoint_latest.pt}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for evaluation outputs}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
TARGET_GROUP="${TARGET_GROUP:-all_disease_indicators}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DISEASES=(${DISEASES:-diabetes hypertension dyslipidemia liver_disease hepatitis_b hepatitis_c kidney_disease anemia})

if [[ "${MODEL}" == "cvae" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/eval/evaluate_joint_cvae_diseases.py \
    --checkpoint "${CHECKPOINT}" \
    --target-group "${TARGET_GROUP}" \
    --variant "cvae_all_indicators" \
    --output-dir "${OUTPUT_DIR}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-name "${DATASET_NAME}" \
    --diseases "${DISEASES[@]}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}"
elif [[ "${MODEL}" == "ddpm" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/eval/evaluate_joint_ddpm_diseases.py \
    --checkpoint "${CHECKPOINT}" \
    --target-group "${TARGET_GROUP}" \
    --variant "ddpm_all_indicators" \
    --output-dir "${OUTPUT_DIR}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-name "${DATASET_NAME}" \
    --diseases "${DISEASES[@]}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}"
else
  echo "MODEL must be cvae or ddpm, got: ${MODEL}" >&2
  exit 1
fi
