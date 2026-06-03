#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-masked_ae_source_4variants_$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${BASE_OUT:-outputs/masked_autoencoder_source/${RUN_ID}}"
mkdir -p "${BASE_OUT}" logs/masked_autoencoder

VARIANTS=(default mask050 wide deep)
CONFIGS=(
  configs/train/train_masked_autoencoder_source_default.yaml
  configs/train/train_masked_autoencoder_source_mask050.yaml
  configs/train/train_masked_autoencoder_source_wide.yaml
  configs/train/train_masked_autoencoder_source_deep.yaml
)
GPUS=(${GPUS:-4 5 6 7})

for i in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$i]}"
  config="${CONFIGS[$i]}"
  gpu="${GPUS[$i]}"
  out_dir="${BASE_OUT}/${variant}"
  log="logs/masked_autoencoder/${RUN_ID}_${variant}_gpu${gpu}.log"
  session="mae_${variant}_gpu${gpu}_$(date +%H%M%S)"
  cmd="cd '$PWD' && CUDA_VISIBLE_DEVICES='$gpu' .venv/bin/python scripts/train/train_masked_autoencoder.py --config '$config' --output-dir '$out_dir' --device cuda 2>&1 | tee '$log'"
  tmux new-session -d -s "$session" "$cmd"
  printf '%s\t%s\t%s\t%s\t%s\n' "$variant" "$gpu" "$session" "$out_dir" "$log" | tee -a "${BASE_OUT}/workers.tsv"
done

echo "${BASE_OUT}" | tee outputs/masked_autoencoder_source/latest_4variants.txt
tmux ls | rg 'mae_.*_gpu' || true
cat "${BASE_OUT}/workers.tsv"
