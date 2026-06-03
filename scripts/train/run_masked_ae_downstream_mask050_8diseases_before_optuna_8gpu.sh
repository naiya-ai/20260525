#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-masked_ae_downstream_mask050_8diseases_before_optuna_$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${BASE_OUT:-outputs/masked_ae_downstream/${RUN_ID}}"
MAE_CHECKPOINT="${MAE_CHECKPOINT:-outputs/masked_autoencoder_source/masked_ae_source_4variants_20260527_235405/mask050/checkpoint_best.pt}"
MAE_VARIANT="${MAE_VARIANT:-mask050}"
CONFIG="${CONFIG:-configs/train/train_masked_ae_downstream_diabetes.yaml}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
DISEASES=(${DISEASES:-diabetes hypertension dyslipidemia liver_disease hepatitis_b hepatitis_c kidney_disease anemia})
STEPS="${STEPS:-1000}"
VALIDATE_EVERY="${VALIDATE_EVERY:-1}"

if [[ "${#GPUS[@]}" -ne "${#DISEASES[@]}" ]]; then
  echo "This runner maps one disease to one GPU. Got ${#DISEASES[@]} diseases and ${#GPUS[@]} GPUs." >&2
  exit 1
fi
if [[ ! -f "${MAE_CHECKPOINT}" ]]; then
  echo "missing MAE checkpoint: ${MAE_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${BASE_OUT}" logs/masked_ae_downstream
{
  echo "RUN_ID=${RUN_ID}"
  echo "BASE_OUT=${BASE_OUT}"
  echo "MAE_CHECKPOINT=${MAE_CHECKPOINT}"
  echo "MAE_VARIANT=${MAE_VARIANT}"
  echo "CONFIG=${CONFIG}"
  echo "DISEASES=${DISEASES[*]}"
  echo "GPUS=${GPUS[*]}"
  echo "STEPS=${STEPS}"
  echo "VALIDATE_EVERY=${VALIDATE_EVERY}"
  echo "NOTE=fixed config before Optuna"
} | tee "${BASE_OUT}/run.info"

for i in "${!DISEASES[@]}"; do
  disease="${DISEASES[$i]}"
  gpu="${GPUS[$i]}"
  out_dir="${BASE_OUT}/${disease}"
  log="logs/masked_ae_downstream/${RUN_ID}_${disease}_gpu${gpu}.log"
  session="mae_before_${MAE_VARIANT}_${disease}_gpu${gpu}_$(date +%H%M%S)"

  mkdir -p "${out_dir}"
  cmd="cd '$PWD' && set -euo pipefail; echo '[start]' '${disease}' 'gpu=${gpu}' 'out=${out_dir}'; CUDA_VISIBLE_DEVICES='${gpu}' PYTHONUNBUFFERED=1 '${PYTHON_BIN}' scripts/train/train_masked_ae_downstream_classifier.py --config '${CONFIG}' --output-dir '${out_dir}' --pretrained-autoencoder-path '${MAE_CHECKPOINT}' --variant-name '${MAE_VARIANT}' --target-disease '${disease}' --device cuda --steps '${STEPS}' --validate-every '${VALIDATE_EVERY}'; echo '[done]' '${disease}'"
  tmux new-session -d -s "${session}" "bash -lc \"${cmd}\""
  tmux pipe-pane -o -t "${session}" "cat >> '${PWD}/${log}'"
  printf '%s\t%s\t%s\t%s\t%s\n' "${gpu}" "${disease}" "${session}" "${out_dir}" "${log}" \
    | tee -a "${BASE_OUT}/workers.tsv"
done

echo "${BASE_OUT}" | tee outputs/masked_ae_downstream/latest_mask050_8diseases_before_optuna.txt
tmux ls | grep "mae_before_${MAE_VARIANT}_" || true
echo "tail -f logs/masked_ae_downstream/${RUN_ID}_${DISEASES[0]}_gpu${GPUS[0]}.log"
