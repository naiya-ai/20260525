#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-outputs/ddpm_mlp_6diseases_knhanes}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MODEL_ROOT}/eval}"
DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-512}"

DISEASES=(
  diabetes
  hypertension
  dyslipidemia
  liver_disease
  kidney_disease
  anemia
)

for disease in "${DISEASES[@]}"; do
  checkpoint="${MODEL_ROOT}/${disease}/checkpoint_best.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    checkpoint="${MODEL_ROOT}/${disease}/checkpoint_latest.pt"
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    printf 'missing checkpoint for disease=%s under %s\n' "${disease}" "${MODEL_ROOT}" >&2
    exit 1
  fi
  output_dir="${OUTPUT_ROOT}/${disease}"
  printf 'evaluate ddpm disease=%s checkpoint=%s\n' "${disease}" "${checkpoint}"
  CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/eval/evaluate_ddpm_prior_probability.py \
    --checkpoint "${checkpoint}" \
    --output-dir "${output_dir}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}"
done
