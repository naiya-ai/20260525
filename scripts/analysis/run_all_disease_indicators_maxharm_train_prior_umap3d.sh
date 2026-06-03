#!/usr/bin/env bash
set -euo pipefail

SWEEP_ROOT="${SWEEP_ROOT:?Set SWEEP_ROOT to the completed all-disease-indicators CVAE sweep root.}"
TARGET_GROUP="${TARGET_GROUP:-all_disease_indicators}"
BETA="${BETA:-0.1}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_latest.pt}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024_plus_nhanes_1988_2023}"
HARMONIZED_CSV="${HARMONIZED_CSV:-datasets/harmonized/harmonized_knhanes_nhanes.csv}"
SPLIT="${SPLIT:-train}"
MAX_POINTS="${MAX_POINTS:-30000}"
SAMPLE_MODE="${SAMPLE_MODE:-balanced}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
MIN_KL_PER_INSTANCE="${MIN_KL_PER_INSTANCE:-0.1}"
N_NEIGHBORS="${N_NEIGHBORS:-30}"
MIN_DIST="${MIN_DIST:-0.05}"
METRIC="${METRIC:-euclidean}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/analysis/$(basename "${SWEEP_ROOT}")/train_prior_umap3d_by_source}"
FIGURES_DIR="${FIGURES_DIR:-figures/$(basename "${SWEEP_ROOT}")/train_prior_umap3d_by_source}"

uv run python scripts/analysis/plot_cvae_train_prior_umap3d_by_source.py \
  --sweep-root "${SWEEP_ROOT}" \
  --checkpoint-name "${CHECKPOINT_NAME}" \
  --target-group "${TARGET_GROUP}" \
  --beta "${BETA}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --harmonized-csv "${HARMONIZED_CSV}" \
  --split "${SPLIT}" \
  --max-points "${MAX_POINTS}" \
  --sample-mode "${SAMPLE_MODE}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --min-kl-per-instance "${MIN_KL_PER_INSTANCE}" \
  --n-neighbors "${N_NEIGHBORS}" \
  --min-dist "${MIN_DIST}" \
  --metric "${METRIC}" \
  --output-dir "${OUTPUT_DIR}" \
  --figures-dir "${FIGURES_DIR}"

echo "==> output_dir=${OUTPUT_DIR}"
echo "==> figures_dir=${FIGURES_DIR}"
