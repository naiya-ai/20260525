#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-masked_ae_downstream_4mae_diabetes_$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${BASE_OUT:-outputs/masked_ae_downstream/${RUN_ID}}"
MAE_ROOT="${MAE_ROOT:-outputs/masked_autoencoder_source/masked_ae_source_4variants_20260527_235405}"
CONFIG="${CONFIG:-configs/train/train_masked_ae_downstream_diabetes.yaml}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
GPUS=(${GPUS:-4 5 6 7})
VARIANTS=(default mask050 wide deep)

mkdir -p "${BASE_OUT}" logs/masked_ae_downstream

for i in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$i]}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  checkpoint="${MAE_ROOT}/${variant}/checkpoint_best.pt"
  out_dir="${BASE_OUT}/${variant}"
  log="logs/masked_ae_downstream/${RUN_ID}_${variant}_gpu${gpu}.log"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  session="mae_down_${variant}_gpu${gpu}_$(date +%H%M%S)"
  cmd="cd '$PWD' && CUDA_VISIBLE_DEVICES='${gpu}' PYTHONUNBUFFERED=1 '${PYTHON_BIN}' scripts/train/train_masked_ae_downstream_classifier.py --config '${CONFIG}' --output-dir '${out_dir}' --pretrained-autoencoder-path '${checkpoint}' --variant-name '${variant}' --target-disease diabetes --device cuda 2>&1 | tee '${log}'"
  tmux new-session -d -s "${session}" "${cmd}"
  printf '%s\n' "${session}" | tee "${BASE_OUT}/${variant}.session"
  printf '%s\t%s\t%s\t%s\t%s\n' "${variant}" "${gpu}" "${session}" "${out_dir}" "${log}" \
    | tee -a "${BASE_OUT}/workers.tsv"
done

echo "${BASE_OUT}" | tee outputs/masked_ae_downstream/latest_4mae_diabetes.txt
tmux ls | rg 'mae_down_' || true
echo "tail -f logs/masked_ae_downstream/${RUN_ID}_default_gpu${GPUS[0]}.log"
