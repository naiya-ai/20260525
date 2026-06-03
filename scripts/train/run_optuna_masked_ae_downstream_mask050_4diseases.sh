#!/usr/bin/env bash
set -euo pipefail

MAE_CHECKPOINT="${MAE_CHECKPOINT:-outputs/masked_autoencoder_source/masked_ae_source_4variants_20260527_235405/mask050/checkpoint_best.pt}"
MAE_VARIANT="${MAE_VARIANT:-mask050}"
TOTAL_TRIALS="${TOTAL_TRIALS:-50}"
STEPS="${STEPS:-1000}"
VALIDATE_EVERY="${VALIDATE_EVERY:-1}"
GPUS=(${GPUS:-4 5 6 7})
DISEASES=(${DISEASES:-diabetes hypertension dyslipidemia liver_disease})
SWEEP_ID="${SWEEP_ID:-optuna_masked_ae_downstream_${MAE_VARIANT}_${TOTAL_TRIALS}t_4d_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/optuna/${SWEEP_ID}}"

if [[ ! -f "${MAE_CHECKPOINT}" ]]; then
  echo "missing MAE checkpoint: ${MAE_CHECKPOINT}" >&2
  exit 1
fi

MAE_CHECKPOINT="${MAE_CHECKPOINT}" \
MAE_VARIANT="${MAE_VARIANT}" \
TRIALS_PER_DISEASE="${TOTAL_TRIALS}" \
STEPS="${STEPS}" \
GPUS="${GPUS[*]}" \
DISEASES="${DISEASES[*]}" \
SWEEP_ID="${SWEEP_ID}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
bash scripts/train/run_optuna_masked_ae_downstream_4diseases.sh

echo
echo "Optuna mask050 downstream launched."
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "tail example:"
echo "tail -f logs/optuna/${SWEEP_ID}_${DISEASES[0]}_gpu${GPUS[0]}.log"
