#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-7}"
ROOT="${ROOT:-model_experiment}"
BASE_CONFIG="${BASE_CONFIG:-configs/train/train_conditional_vae_raw_mlp_diabetes.yaml}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DISEASE="${DISEASE:-diabetes}"
WIDTHS=(${WIDTHS:-128 256 512 1024 2048})
REPS="${REPS:-5}"
STEPS="${STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-}"
BATCH_NORM="${BATCH_NORM:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4096}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
SKIP_TRAIN_IF_DONE="${SKIP_TRAIN_IF_DONE:-1}"

MODEL_ROOT="${ROOT}/models/mlp"
CONFIG_ROOT="${ROOT}/configs/mlp"
FIGURE_ROOT="${ROOT}/figures"
LOSS_ROOT="${FIGURE_ROOT}/loss/mlp"
SUMMARY_CSV="${ROOT}/summary.csv"

mkdir -p "${MODEL_ROOT}" "${CONFIG_ROOT}" "${LOSS_ROOT}" "${FIGURE_ROOT}"

for width in "${WIDTHS[@]}"; do
  for rep in $(seq 1 "${REPS}"); do
    rep_tag="$(printf '%02d' "${rep}")"
    run_dir="${MODEL_ROOT}/width_${width}/rep_${rep_tag}"
    config_path="${CONFIG_ROOT}/train_conditional_vae_raw_mlp_width${width}_rep${rep_tag}.yaml"
    eval_dir="${run_dir}/eval"
    variant="mlp_w${width}_rep${rep_tag}"
    checkpoint="${run_dir}/checkpoint_best.pt"
    metrics_json="${eval_dir}/${variant}_${DISEASE}_metrics.json"
    seed=$((50000 + width * 10 + rep))

    mkdir -p "${run_dir}" "${eval_dir}"
    uv run python - "${BASE_CONFIG}" "${config_path}" "${run_dir}" "${width}" "${seed}" "${DATASET_ROOT}" "${DATASET_NAME}" "${DISEASE}" "${STEPS}" "${BATCH_SIZE}" "${LEARNING_RATE}" "${LR_WARMUP_STEPS}" "${GRADIENT_CLIP_NORM}" "${BATCH_NORM}" <<'PY'
import sys
from pathlib import Path

import yaml

(
    base_path,
    out_path,
    run_dir,
    width,
    seed,
    dataset_root,
    dataset_name,
    disease,
    steps,
    batch_size,
    learning_rate,
    lr_warmup_steps,
    gradient_clip_norm,
    batch_norm,
) = sys.argv[1:]
with Path(base_path).open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)
width = int(width)
config["seed"] = int(seed)
config["output_dir"] = run_dir
config["data"]["dataset_root"] = dataset_root
config["data"]["dataset_name"] = dataset_name
config["data"]["target_group"] = disease
config["model"]["encoder_hidden_layers"] = [width, width]
config["model"]["prior_hidden_layers"] = [width, width]
config["model"]["decoder_hidden_layers"] = [width, width]
if steps:
    config["train"]["steps"] = int(steps)
if batch_size:
    config["train"]["batch_size"] = int(batch_size)
if learning_rate:
    config["train"]["learning_rate"] = float(learning_rate)
if lr_warmup_steps:
    config["train"]["lr_warmup_steps"] = int(lr_warmup_steps)
if gradient_clip_norm:
    config["train"]["gradient_clip_norm"] = float(gradient_clip_norm)
if batch_norm:
    config["model"]["batch_norm"] = batch_norm.strip().lower() in {"1", "true", "yes", "y"}
Path(out_path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

    if [[ "${SKIP_TRAIN_IF_DONE}" == "1" && -f "${checkpoint}" ]]; then
      echo "==> skip training existing ${checkpoint}"
    else
      echo "==> train ${variant} on GPU ${GPU}"
      CUDA_VISIBLE_DEVICES="${GPU}" uv run python src/train/train_conditional_vae.py \
        --config "${config_path}" \
        --device cuda
    fi

    if [[ ! -f "${checkpoint}" ]]; then
      echo "Missing checkpoint: ${checkpoint}" >&2
      exit 1
    fi

    echo "==> evaluate ${variant}"
    CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/eval/evaluate_cvae_prior_probability.py \
      --checkpoint "${checkpoint}" \
      --target-group "${DISEASE}" \
      --variant "${variant}" \
      --output-dir "${eval_dir}" \
      --dataset-root "${DATASET_ROOT}" \
      --dataset-name "${DATASET_NAME}" \
      --device cuda \
      --batch-size "${EVAL_BATCH_SIZE}" \
      --num-workers 0 \
      --num-samples "${NUM_SAMPLES}" \
      --save-probabilities

    uv run python scripts/model_experiment/update_experiment_summary.py \
      --model mlp \
      --width "${width}" \
      --rep "${rep}" \
      --config "${config_path}" \
      --run-dir "${run_dir}" \
      --checkpoint "${checkpoint}" \
      --metrics-json "${metrics_json}" \
      --summary-csv "${SUMMARY_CSV}" \
      --figures-dir "${FIGURE_ROOT}" \
      --loss-dir "${LOSS_ROOT}"
  done
done
