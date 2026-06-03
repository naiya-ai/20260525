#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/preprocess/preprocess_variable_groups.yaml}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
HARMONIZED_DATASET="${HARMONIZED_DATASET:-datasets/harmonized/harmonized_knhanes_nhanes.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-datasets/preprocessed/gaussian_quantile}"
TARGET_GROUP_FILE="${TARGET_GROUP_FILE:-all_disease_indicators.txt}"
SURVEYS=(${SURVEYS:-knhanes})
YEAR_MIN="${YEAR_MIN:-1998}"
YEAR_MAX="${YEAR_MAX:-2024}"

uv run python scripts/dataset/preprocess_variable_groups.py \
  --config "${CONFIG}" \
  --harmonized-dataset "${HARMONIZED_DATASET}" \
  --dataset-name "${DATASET_NAME}" \
  --survey "${SURVEYS[@]}" \
  --year-min "${YEAR_MIN}" \
  --year-max "${YEAR_MAX}" \
  --output-dir "${OUTPUT_DIR}" \
  --variable-groups "${TARGET_GROUP_FILE}"

echo "==> wrote ${OUTPUT_DIR}/${DATASET_NAME}/all_disease_indicators"
