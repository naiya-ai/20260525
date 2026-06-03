#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-masked_ae_source_4variants_resume3000_$(date +%Y%m%d_%H%M%S)}"
MAE_ROOT="${MAE_ROOT:-outputs/masked_autoencoder_source/masked_ae_source_4variants_20260527_235405}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
ADDITIONAL_STEPS="${ADDITIONAL_STEPS:-3000}"
GPUS=(${GPUS:-4 5 6 7})
VARIANTS=(default mask050 wide deep)

mkdir -p logs/masked_autoencoder

for i in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$i]}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  out_dir="${MAE_ROOT}/${variant}"
  config="$(find "${out_dir}" -maxdepth 1 -name 'train_masked_autoencoder_source*.yaml' | head -n 1)"
  checkpoint="${out_dir}/checkpoint_latest.pt"
  log="logs/masked_autoencoder/${RUN_ID}_${variant}_gpu${gpu}.log"
  session="mae_resume_${variant}_gpu${gpu}_$(date +%H%M%S)"
  if [[ ! -f "${config}" ]]; then
    echo "missing config under ${out_dir}" >&2
    exit 1
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  cmd="cd '$PWD' && CUDA_VISIBLE_DEVICES='${gpu}' PYTHONUNBUFFERED=1 '${PYTHON_BIN}' scripts/train/train_masked_autoencoder.py --config '${config}' --output-dir '${out_dir}' --device cuda --resume-checkpoint '${checkpoint}' --additional-steps '${ADDITIONAL_STEPS}' 2>&1 | tee '${log}'"
  tmux new-session -d -s "${session}" "${cmd}"
  printf '%s\t%s\t%s\t%s\t%s\n' "${variant}" "${gpu}" "${session}" "${out_dir}" "${log}" | tee -a "${MAE_ROOT}/${RUN_ID}.workers.tsv"
done

echo "${RUN_ID}" | tee "${MAE_ROOT}/latest_resume_run.txt"
tmux ls | rg 'mae_resume_' || true
echo "tail -f logs/masked_autoencoder/${RUN_ID}_default_gpu${GPUS[0]}.log"
