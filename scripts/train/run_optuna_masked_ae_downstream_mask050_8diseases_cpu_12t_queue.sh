#!/usr/bin/env bash
set -euo pipefail

MAE_CHECKPOINT="${MAE_CHECKPOINT:-outputs/masked_autoencoder_source/masked_ae_source_4variants_20260527_235405/mask050/checkpoint_best.pt}"
MAE_VARIANT="${MAE_VARIANT:-mask050}"
TOTAL_TRIALS="${TOTAL_TRIALS:-12}"
STEPS="${STEPS:-1000}"
VALIDATE_EVERY="${VALIDATE_EVERY:-1}"
CPU_WORKERS="${CPU_WORKERS:-2}"
DISEASES=(${DISEASES:-diabetes hypertension dyslipidemia liver_disease hepatitis_b hepatitis_c kidney_disease anemia})
SWEEP_ID="${SWEEP_ID:-optuna_masked_ae_downstream_${MAE_VARIANT}_${TOTAL_TRIALS}t_8d_cpu_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/optuna/${SWEEP_ID}}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

if (( CPU_WORKERS < 1 )); then
  echo "CPU_WORKERS must be >= 1. Got: ${CPU_WORKERS}" >&2
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
  echo "CPU_WORKERS=${CPU_WORKERS}"
  echo "TOTAL_TRIALS_PER_DISEASE=${TOTAL_TRIALS}"
  echo "STEPS=${STEPS}"
  echo "VALIDATE_EVERY=${VALIDATE_EVERY}"
  echo "DEVICE=cpu"
  echo "CUDA_VISIBLE_DEVICES="
} | tee "${OUTPUT_ROOT}/run.info"

for (( worker_idx = 0; worker_idx < CPU_WORKERS; worker_idx++ )); do
  queue=()
  for i in "${!DISEASES[@]}"; do
    if (( i % CPU_WORKERS == worker_idx )); then
      queue+=("${DISEASES[$i]}")
    fi
  done
  queue_file="${OUTPUT_ROOT}/cpu${worker_idx}_queue.txt"
  printf '%s\n' "${queue[@]}" > "${queue_file}"
  log="logs/optuna/${SWEEP_ID}_cpu${worker_idx}.log"
  session="opt_mae_${MAE_VARIANT}_cpu${worker_idx}_$(date +%H%M%S)"
  cmd="cd '$PWD' && set -euo pipefail; export CUDA_VISIBLE_DEVICES=''; export PYTHONUNBUFFERED=1; while read -r disease; do [[ -z \"\$disease\" ]] && continue; disease_root='${OUTPUT_ROOT}/'\"\$disease\"; mkdir -p \"\$disease_root\"; storage='sqlite:///'\"\$disease_root\"'/optuna_study.db'; study='${SWEEP_ID}_'\"\$disease\"; echo '[start]' \"\$disease\" 'cpu_worker=${worker_idx}' 'root='\"\$disease_root\"; '${PYTHON_BIN}' scripts/train/optuna_masked_ae_downstream_classifier.py --config configs/train/train_masked_ae_downstream_diabetes.yaml --output-root \"\$disease_root\" --study-name \"\$study\" --storage \"\$storage\" --n-trials '${TOTAL_TRIALS}' --target-disease \"\$disease\" --pretrained-autoencoder-path '${MAE_CHECKPOINT}' --variant-name '${MAE_VARIANT}' --device cpu --steps '${STEPS}' --validate-every '${VALIDATE_EVERY}'; echo '[done]' \"\$disease\"; done < '${queue_file}' 2>&1 | tee '${log}'"
  tmux new-session -d -s "${session}" "${cmd}"
  printf '%s\t%s\t%s\t%s\n' "${worker_idx}" "${session}" "${queue[*]}" "${log}" \
    | tee -a "${OUTPUT_ROOT}/workers.tsv"
done

echo "${OUTPUT_ROOT}" | tee outputs/optuna/latest_masked_ae_downstream_8diseases_cpu.txt
tmux ls | rg "opt_mae_${MAE_VARIANT}_cpu" || true
echo "tail -f logs/optuna/${SWEEP_ID}_cpu0.log"
