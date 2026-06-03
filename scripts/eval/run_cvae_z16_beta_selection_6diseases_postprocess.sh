#!/usr/bin/env bash
set -euo pipefail

SWEEP_ROOT="${SWEEP_ROOT:?Set SWEEP_ROOT to the completed z16 sweep root.}"
STATUS_FILE="${STATUS_FILE:-${SWEEP_ROOT}/train_status.tsv}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_latest.pt}"
DIAGNOSTICS_SAMPLES="${DIAGNOSTICS_SAMPLES:-1000}"
COVERAGE_SAMPLES="${COVERAGE_SAMPLES:-1000}"
DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-${SWEEP_ROOT}/diagnostics_s${DIAGNOSTICS_SAMPLES}_${CHECKPOINT_NAME%.pt}}"
COVERAGE_DIR="${COVERAGE_DIR:-${SWEEP_ROOT}/coverage_s${COVERAGE_SAMPLES}_${CHECKPOINT_NAME%.pt}}"
REPORT_DIR="${REPORT_DIR:-${SWEEP_ROOT}/beta_selection_report_${CHECKPOINT_NAME%.pt}}"
FIGURES_DIR="${FIGURES_DIR:-figures/$(basename "${SWEEP_ROOT}")}"
DISEASES=(${DISEASES:-diabetes hypertension dyslipidemia liver_disease kidney_disease anemia})
CATBOOST_SUMMARY="${CATBOOST_SUMMARY:-outputs/catboost_weighted_range_compare/20260525_071919/weighted_catboost_range_summary.csv}"
CATBOOST_DATASET_NAME="${CATBOOST_DATASET_NAME:-harmonized_knhanes_1998_2024}"

SWEEP_ROOT="${SWEEP_ROOT}" \
STATUS_FILE="${STATUS_FILE}" \
GPUS="${GPUS}" \
DEVICE="${DEVICE}" \
CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
DIAGNOSTICS_SAMPLES="${DIAGNOSTICS_SAMPLES}" \
COVERAGE_SAMPLES="${COVERAGE_SAMPLES}" \
DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR}" \
COVERAGE_DIR="${COVERAGE_DIR}" \
REPORT_DIR="${REPORT_DIR}" \
FIGURES_DIR="${FIGURES_DIR}" \
./scripts/eval/run_beta_selection_postprocess.sh

uv run python scripts/eval/plot_disease_beta_metric_tables.py \
  --diagnostics-summary "${DIAGNOSTICS_DIR}/diagnostics_summary.csv" \
  --coverage-summary "${COVERAGE_DIR}/coverage_summary.csv" \
  --diseases "${DISEASES[@]}" \
  --output-prefix disease_beta_metric_table_no_hepatitis \
  --output-dir "${REPORT_DIR}"

if [ -f "${CATBOOST_SUMMARY}" ]; then
  if [ -f "${REPORT_DIR}/selected_betas.csv" ]; then
    uv run python scripts/eval/plot_selected_cvae_noncollapsed_vs_catboost_auroc.py \
      --diagnostics-summary "${DIAGNOSTICS_DIR}/diagnostics_summary.csv" \
      --selected-betas "${REPORT_DIR}/selected_betas.csv" \
      --catboost-summary "${CATBOOST_SUMMARY}" \
      --dataset-name "${CATBOOST_DATASET_NAME}" \
      --output-prefix selected_cvae_noncollapsed_no_hepatitis_vs_catboost_auroc \
      --output-dir "${REPORT_DIR}"
  fi

  uv run python scripts/eval/plot_selected_cvae_noncollapsed_vs_catboost_auroc.py \
    --diagnostics-summary "${DIAGNOSTICS_DIR}/diagnostics_summary.csv" \
    --beta 0.1 \
    --catboost-summary "${CATBOOST_SUMMARY}" \
    --dataset-name "${CATBOOST_DATASET_NAME}" \
    --output-prefix beta010_cvae_noncollapsed_no_hepatitis_vs_catboost_auroc \
    --output-dir "${REPORT_DIR}"
else
  echo "skip CatBoost comparison: missing ${CATBOOST_SUMMARY}" >&2
fi

mkdir -p "${FIGURES_DIR}"
find "${REPORT_DIR}" -maxdepth 1 -type f \( -name '*.png' -o -name '*.svg' \) -exec cp -f {} "${FIGURES_DIR}/" \;
echo "==> report_dir=${REPORT_DIR}"
echo "==> figures_dir=${FIGURES_DIR}"
