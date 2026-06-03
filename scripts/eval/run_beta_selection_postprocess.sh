#!/usr/bin/env bash
set -euo pipefail

SWEEP_ROOT="${SWEEP_ROOT:?Set SWEEP_ROOT to the completed beta-selection sweep root.}"
STATUS_FILE="${STATUS_FILE:-${SWEEP_ROOT}/train_status.tsv}"
TRAIN_PID="${TRAIN_PID:-}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_latest.pt}"
DIAGNOSTICS_SAMPLES="${DIAGNOSTICS_SAMPLES:-1000}"
COVERAGE_SAMPLES="${COVERAGE_SAMPLES:-1000}"
DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-${SWEEP_ROOT}/diagnostics_s${DIAGNOSTICS_SAMPLES}_${CHECKPOINT_NAME%.pt}}"
COVERAGE_DIR="${COVERAGE_DIR:-${SWEEP_ROOT}/coverage_s${COVERAGE_SAMPLES}_${CHECKPOINT_NAME%.pt}}"
REPORT_DIR="${REPORT_DIR:-${SWEEP_ROOT}/beta_selection_report_${CHECKPOINT_NAME%.pt}}"
FIGURES_DIR="${FIGURES_DIR:-figures/$(basename "${SWEEP_ROOT}")}"
PRIOR_SAMPLE_CACHE_DIR="${PRIOR_SAMPLE_CACHE_DIR:-${SWEEP_ROOT}/prior_sample_cache_s${DIAGNOSTICS_SAMPLES}_${CHECKPOINT_NAME%.pt}}"
COVERAGE_PRIOR_SAMPLE_CACHE_DIR=""
if [ "${DIAGNOSTICS_SAMPLES}" = "${COVERAGE_SAMPLES}" ]; then
  COVERAGE_PRIOR_SAMPLE_CACHE_DIR="${PRIOR_SAMPLE_CACHE_DIR}"
fi

if [ -n "${TRAIN_PID}" ]; then
  echo "==> waiting for training pid=${TRAIN_PID}"
  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    sleep 60
  done
fi

echo "==> sweep_root=${SWEEP_ROOT}"
echo "==> status_file=${STATUS_FILE}"
echo "==> checkpoint_name=${CHECKPOINT_NAME}"

SWEEP_ROOT="${SWEEP_ROOT}" \
STATUS_FILE="${STATUS_FILE}" \
OUT_DIR="${DIAGNOSTICS_DIR}" \
DEVICE="${DEVICE}" \
GPUS="${GPUS}" \
NUM_SAMPLES="${DIAGNOSTICS_SAMPLES}" \
CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
PRIOR_SAMPLE_CACHE_DIR="${PRIOR_SAMPLE_CACHE_DIR}" \
PRIOR_SAMPLE_CACHE_SPLITS="test" \
./scripts/eval/evaluate_beta_sweep_diagnostics.sh

SWEEP_ROOT="${SWEEP_ROOT}" \
STATUS_FILE="${STATUS_FILE}" \
OUT_DIR="${COVERAGE_DIR}" \
DEVICE="${DEVICE}" \
GPUS="${GPUS}" \
NUM_SAMPLES="${COVERAGE_SAMPLES}" \
CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
MIXED_PRECISION="${MIXED_PRECISION:-1}" \
PRIOR_SAMPLE_CACHE_DIR="${COVERAGE_PRIOR_SAMPLE_CACHE_DIR}" \
./scripts/eval/evaluate_beta_sweep_coverage.sh

uv run python scripts/eval/plot_beta_selection_onepager.py \
  --status-file "${STATUS_FILE}" \
  --diagnostics-summary "${DIAGNOSTICS_DIR}/diagnostics_summary.csv" \
  --coverage-summary "${COVERAGE_DIR}/coverage_summary.csv" \
  --output-dir "${REPORT_DIR}"

echo "==> report=${REPORT_DIR}/beta_selection_onepager.png"

mkdir -p "${FIGURES_DIR}"
find "${REPORT_DIR}" -maxdepth 1 -type f \( -name '*.png' -o -name '*.svg' \) -exec cp -f {} "${FIGURES_DIR}/" \;
echo "==> figures_dir=${FIGURES_DIR}"
