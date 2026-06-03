#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/preprocess/preprocess_variable_groups.yaml}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024_plus_nhanes_1988_2023}"
HARMONIZED_DATASET="${HARMONIZED_DATASET:-datasets/harmonized/harmonized_knhanes_nhanes.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-datasets/preprocessed/gaussian_quantile}"
TARGET_GROUP_FILE="${TARGET_GROUP_FILE:-all_disease_indicators.txt}"

uv run python scripts/dataset/preprocess_variable_groups.py \
  --config "${CONFIG}" \
  --harmonized-dataset "${HARMONIZED_DATASET}" \
  --dataset-name "${DATASET_NAME}" \
  --survey knhanes nhanes \
  --year-min 1988 \
  --year-max 2024 \
  --output-dir "${OUTPUT_DIR}" \
  --variable-groups "${TARGET_GROUP_FILE}"

echo "==> wrote ${OUTPUT_DIR}/${DATASET_NAME}/all_disease_indicators"
