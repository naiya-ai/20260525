#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${MAE_CHECKPOINT:-}" ]]; then
  echo "Set MAE_CHECKPOINT to the selected masked-AE checkpoint_best.pt." >&2
  exit 1
fi

MAE_VARIANT="${MAE_VARIANT:-selected}"
TOTAL_TRIALS="${TOTAL_TRIALS:-50}"
GPUS=(${GPUS:-4 5 6 7})
DISEASES=(${DISEASES:-diabetes hypertension dyslipidemia liver_disease})
TRIALS_PER_DISEASE="${TRIALS_PER_DISEASE:-${TOTAL_TRIALS}}"
STEPS="${STEPS:-1000}"
VALIDATE_EVERY="${VALIDATE_EVERY:-100}"
SWEEP_ID="${SWEEP_ID:-optuna_masked_ae_downstream_${MAE_VARIANT}_${TRIALS_PER_DISEASE}t_4d_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/optuna/${SWEEP_ID}}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

mkdir -p "${OUTPUT_ROOT}" logs/optuna
{
  echo "SWEEP_ID=${SWEEP_ID}"
  echo "MAE_CHECKPOINT=${MAE_CHECKPOINT}"
  echo "MAE_VARIANT=${MAE_VARIANT}"
  echo "DISEASES=${DISEASES[*]}"
  echo "GPUS=${GPUS[*]}"
  echo "TRIALS_PER_DISEASE=${TRIALS_PER_DISEASE}"
  echo "VALIDATE_EVERY=${VALIDATE_EVERY}"
} | tee "${OUTPUT_ROOT}/run.info"

for i in "${!DISEASES[@]}"; do
  disease="${DISEASES[$i]}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  disease_root="${OUTPUT_ROOT}/${disease}"
  log="logs/optuna/${SWEEP_ID}_${disease}_gpu${gpu}.log"
  storage="sqlite:///${disease_root}/optuna_study.db"
  study_name="${SWEEP_ID}_${disease}"
  mkdir -p "${disease_root}"
  session="opt_mae_${disease}_gpu${gpu}_$(date +%H%M%S)"
  cmd="cd '$PWD' && CUDA_VISIBLE_DEVICES='${gpu}' PYTHONUNBUFFERED=1 '${PYTHON_BIN}' scripts/train/optuna_masked_ae_downstream_classifier.py --config configs/train/train_masked_ae_downstream_diabetes.yaml --output-root '${disease_root}' --study-name '${study_name}' --storage '${storage}' --n-trials '${TRIALS_PER_DISEASE}' --target-disease '${disease}' --pretrained-autoencoder-path '${MAE_CHECKPOINT}' --variant-name '${MAE_VARIANT}' --device cuda --steps '${STEPS}' --validate-every '${VALIDATE_EVERY}' 2>&1 | tee '${log}'"
  tmux new-session -d -s "${session}" "${cmd}"
  printf '%s\n' "${session}" | tee "${disease_root}/worker.session"
  printf '%s\t%s\t%s\t%s\t%s\n' "${disease}" "${gpu}" "${session}" "${disease_root}" "${log}" \
    | tee -a "${OUTPUT_ROOT}/workers.tsv"
done

echo "${OUTPUT_ROOT}" | tee outputs/optuna/latest_masked_ae_downstream_4diseases.txt
tmux ls | rg 'opt_mae_' || true
echo "tail -f logs/optuna/${SWEEP_ID}_${DISEASES[0]}_gpu${GPUS[0]}.log"
