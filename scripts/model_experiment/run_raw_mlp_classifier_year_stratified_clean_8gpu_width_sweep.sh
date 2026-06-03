#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/train/train_raw_mlp_disease_classifier_diabetes_year_stratified.yaml}"
ROOT="${ROOT:-model_experiment/classification_year_stratified_clean_8gpu_lr3e-4_100k}"
PYTHON_BIN="${PYTHON_BIN:-uv run python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-uv run python -m torch.distributed.run}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
read -r -a TORCHRUN_CMD <<< "${TORCHRUN_BIN}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"
WIDTHS=(${WIDTHS:-4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384})
STEPS="${STEPS:-100000}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-64}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
DROPOUT="${DROPOUT:-0}"
NORMALIZATION="${NORMALIZATION:-none}"
VALIDATE_EVERY="${VALIDATE_EVERY:-100}"
LOG_EVERY="${LOG_EVERY:-50}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29700}"

mkdir -p "${ROOT}/configs/mlp" "${ROOT}/logs" "${ROOT}/models/mlp" "${ROOT}/figures/loss/mlp" "${ROOT}/figures/loss"

SUMMARY_CSV="${ROOT}/summary.csv"
FIGURES_DIR="${ROOT}/figures"
LOSS_DIR="${ROOT}/figures/loss/mlp"
COMBINED_LOSS_DIR="${ROOT}/figures/loss"

make_config() {
  local width="$1"
  local config_path="$2"
  local out_dir="$3"
  local seed="$4"
  "${PYTHON_CMD[@]}" - "${CONFIG}" "${config_path}" "${out_dir}" "${width}" "${seed}" "${STEPS}" "${PER_GPU_BATCH_SIZE}" "${LR}" "${WEIGHT_DECAY}" "${DROPOUT}" "${NORMALIZATION}" "${VALIDATE_EVERY}" "${LOG_EVERY}" <<'PY'
from pathlib import Path
import sys
import yaml

(
    base_path,
    out_path,
    out_dir,
    width,
    seed,
    steps,
    batch_size,
    lr,
    weight_decay,
    dropout,
    normalization,
    validate_every,
    log_every,
) = sys.argv[1:]

config = yaml.safe_load(Path(base_path).read_text())
config["output_dir"] = out_dir
config["seed"] = int(seed)
config["device"] = "cuda"
config.setdefault("model", {})["hidden_layers"] = [int(width), int(width)]
config.setdefault("model", {})["batch_norm"] = False
config.setdefault("model", {})["normalization"] = normalization
config.setdefault("model", {})["dropout"] = float(dropout)
config.setdefault("train", {})["steps"] = int(steps)
config.setdefault("train", {})["batch_size"] = int(batch_size)
config.setdefault("train", {})["learning_rate"] = float(lr)
config.setdefault("train", {})["weight_decay"] = float(weight_decay)
config.setdefault("train", {}).setdefault("early_stopping", {})["enabled"] = False
config.setdefault("logging", {})["validate_every"] = int(validate_every)
config.setdefault("logging", {})["log_every"] = int(log_every)
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
Path(out_path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
}

job_index=0
for width in "${WIDTHS[@]}"; do
  out_dir="${ROOT}/models/mlp/width_${width}/rep_01"
  config_path="${ROOT}/configs/mlp/train_raw_mlp_classifier_width${width}_rep01.yaml"
  norm_label="${NORMALIZATION//_/-}"
  log_path="${ROOT}/logs/mlp_width${width}_rep01_ddp_${norm_label}.log"
  seed="$((2000 + width))"
  port="$((MASTER_PORT_BASE + job_index))"
  make_config "${width}" "${config_path}" "${out_dir}" "${seed}"

  echo "==> train width=${width} DDP GPUs=${GPUS} per_gpu_batch=${PER_GPU_BATCH_SIZE} effective_batch=$((PER_GPU_BATCH_SIZE * NPROC)) lr=${LR} normalization=${NORMALIZATION} dropout=${DROPOUT} weight_decay=${WEIGHT_DECAY}"
  CUDA_VISIBLE_DEVICES="${GPUS}" PYTHONUNBUFFERED=1 "${TORCHRUN_CMD[@]}" \
    --nnodes 1 \
    --nproc-per-node "${NPROC}" \
    --master-addr 127.0.0.1 \
    --master-port "${port}" \
    scripts/model_experiment/train_raw_mlp_disease_classifier.py \
      --config "${config_path}" \
      --device cuda \
      --summary-csv "${SUMMARY_CSV}" \
      --figures-dir "${FIGURES_DIR}" \
      --loss-dir "${LOSS_DIR}" 2>&1 | tee "${log_path}"

  "${PYTHON_CMD[@]}" scripts/model_experiment/plot_raw_mlp_loss_by_width.py \
    --root "${ROOT}" \
    --output-dir "${COMBINED_LOSS_DIR}" \
    --prefix train_valid_loss_by_width
  job_index=$((job_index + 1))
done

echo "done: ${ROOT}"
echo "summary: ${SUMMARY_CSV}"
echo "scatter plots: ${FIGURES_DIR}"
echo "combined loss: ${COMBINED_LOSS_DIR}/train_valid_loss_by_width.png"
