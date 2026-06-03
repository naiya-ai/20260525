#!/usr/bin/env bash
set -euo pipefail

MAE_CHECKPOINT="${MAE_CHECKPOINT:-outputs/masked_autoencoder_source/masked_ae_source_4variants_20260527_235405/mask050/checkpoint_best.pt}"
MAE_VARIANT="${MAE_VARIANT:-mask050}"
TOTAL_TRIALS="${TOTAL_TRIALS:-50}"
STEPS="${STEPS:-1000}"
VALIDATE_EVERY="${VALIDATE_EVERY:-1}"
GPUS=(${GPUS:-4 5})
DISEASES=(${DISEASES:-diabetes hypertension dyslipidemia liver_disease hepatitis_b hepatitis_c kidney_disease anemia})
SWEEP_ID="${SWEEP_ID:-optuna_masked_ae_downstream_${MAE_VARIANT}_${TOTAL_TRIALS}t_8d_2gpu_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/optuna/${SWEEP_ID}}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

if [[ "${#GPUS[@]}" -ne 2 ]]; then
  echo "This queue runner expects exactly 2 GPUs. Got: ${GPUS[*]}" >&2
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

for worker_idx in 0 1; do
  gpu="${GPUS[$worker_idx]}"
  queue=()
  for i in "${!DISEASES[@]}"; do
    if (( i % 2 == worker_idx )); then
      queue+=("${DISEASES[$i]}")
    fi
  done
  queue_file="${OUTPUT_ROOT}/gpu${gpu}_queue.txt"
  printf '%s\n' "${queue[@]}" > "${queue_file}"
  log="logs/optuna/${SWEEP_ID}_gpu${gpu}.log"
  session="opt_mae_${MAE_VARIANT}_gpu${gpu}_$(date +%H%M%S)"
  cmd="cd '$PWD' && set -euo pipefail; while read -r disease; do [[ -z \"\$disease\" ]] && continue; disease_root='${OUTPUT_ROOT}/'\"\$disease\"; mkdir -p \"\$disease_root\"; storage='sqlite:///'\"\$disease_root\"'/optuna_study.db'; study='${SWEEP_ID}_'\"\$disease\"; echo '[start]' \"\$disease\" 'gpu=${gpu}' 'root='\"\$disease_root\"; CUDA_VISIBLE_DEVICES='${gpu}' PYTHONUNBUFFERED=1 '${PYTHON_BIN}' scripts/train/optuna_masked_ae_downstream_classifier.py --config configs/train/train_masked_ae_downstream_diabetes.yaml --output-root \"\$disease_root\" --study-name \"\$study\" --storage \"\$storage\" --n-trials '${TOTAL_TRIALS}' --target-disease \"\$disease\" --pretrained-autoencoder-path '${MAE_CHECKPOINT}' --variant-name '${MAE_VARIANT}' --device cuda --steps '${STEPS}' --validate-every '${VALIDATE_EVERY}'; echo '[done]' \"\$disease\"; done < '${queue_file}' 2>&1 | tee '${log}'"
  tmux new-session -d -s "${session}" "${cmd}"
  printf '%s\t%s\t%s\t%s\t%s\n' "${gpu}" "${session}" "${queue[*]}" "${queue_file}" "${log}" \
    | tee -a "${OUTPUT_ROOT}/workers.tsv"
done

echo "${OUTPUT_ROOT}" | tee outputs/optuna/latest_masked_ae_downstream_8diseases_2gpu.txt
tmux ls | rg "opt_mae_${MAE_VARIANT}_gpu" || true
echo "tail -f logs/optuna/${SWEEP_ID}_gpu${GPUS[0]}.log"
