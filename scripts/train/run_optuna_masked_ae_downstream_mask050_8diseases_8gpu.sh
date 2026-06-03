#!/usr/bin/env bash
set -euo pipefail

MAE_CHECKPOINT="${MAE_CHECKPOINT:-outputs/masked_autoencoder_source/masked_ae_source_4variants_20260527_235405/mask050/checkpoint_best.pt}"
MAE_VARIANT="${MAE_VARIANT:-mask050}"
TOTAL_TRIALS="${TOTAL_TRIALS:-20}"
STEPS="${STEPS:-1000}"
VALIDATE_EVERY="${VALIDATE_EVERY:-1}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
DISEASES=(${DISEASES:-diabetes hypertension dyslipidemia liver_disease hepatitis_b hepatitis_c kidney_disease anemia})
SWEEP_ID="${SWEEP_ID:-optuna_masked_ae_downstream_${MAE_VARIANT}_${TOTAL_TRIALS}t_8d_8gpu_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/optuna/${SWEEP_ID}}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

if [[ "${#GPUS[@]}" -ne "${#DISEASES[@]}" ]]; then
  echo "This runner maps one disease to one GPU. Got ${#DISEASES[@]} diseases and ${#GPUS[@]} GPUs." >&2
  exit 1
fi
if [[ ! -f "${MAE_CHECKPOINT}" ]]; then
  echo "missing MAE checkpoint: ${MAE_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" logs/optuna
{
  echo "SWEEP_ID=${SWEEP_ID}"
  echo "MAE_CHECKPOINT=${MAE_CHECKPOINT}"
  echo "MAE_VARIANT=${MAE_VARIANT}"
  echo "DISEASES=${DISEASES[*]}"
  echo "GPUS=${GPUS[*]}"
  echo "TOTAL_TRIALS_PER_DISEASE=${TOTAL_TRIALS}"
  echo "STEPS=${STEPS}"
  echo "VALIDATE_EVERY=${VALIDATE_EVERY}"
} | tee "${OUTPUT_ROOT}/run.info"

for i in "${!DISEASES[@]}"; do
  disease="${DISEASES[$i]}"
  gpu="${GPUS[$i]}"
  disease_root="${OUTPUT_ROOT}/${disease}"
  log="logs/optuna/${SWEEP_ID}_${disease}_gpu${gpu}.log"
  session="opt_mae_${MAE_VARIANT}_${disease}_gpu${gpu}_$(date +%H%M%S)"
  storage="sqlite:///${disease_root}/optuna_study.db"
  study="${SWEEP_ID}_${disease}"

  mkdir -p "${disease_root}"
  cmd="cd '$PWD' && set -euo pipefail; echo '[start]' '${disease}' 'gpu=${gpu}' 'root=${disease_root}'; CUDA_VISIBLE_DEVICES='${gpu}' PYTHONUNBUFFERED=1 '${PYTHON_BIN}' scripts/train/optuna_masked_ae_downstream_classifier.py --config configs/train/train_masked_ae_downstream_diabetes.yaml --output-root '${disease_root}' --study-name '${study}' --storage '${storage}' --n-trials '${TOTAL_TRIALS}' --target-disease '${disease}' --pretrained-autoencoder-path '${MAE_CHECKPOINT}' --variant-name '${MAE_VARIANT}' --device cuda --steps '${STEPS}' --validate-every '${VALIDATE_EVERY}'; echo '[done]' '${disease}'"
  tmux new-session -d -s "${session}" "bash -lc \"${cmd} 2>&1 | tee '${log}'\""
  printf '%s\t%s\t%s\t%s\n' "${gpu}" "${disease}" "${session}" "${log}" \
    | tee -a "${OUTPUT_ROOT}/workers.tsv"
done

echo "${OUTPUT_ROOT}" | tee outputs/optuna/latest_masked_ae_downstream_8diseases_8gpu.txt
tmux ls | grep "opt_mae_${MAE_VARIANT}_" || true
echo "tail -f logs/optuna/${SWEEP_ID}_${DISEASES[0]}_gpu${GPUS[0]}.log"
