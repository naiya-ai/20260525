#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/train/train_raw_mlp_disease_classifier_diabetes.yaml}"
ROOT="${ROOT:-model_experiment/classification_ddp_plan}"
PYTHON_BIN="${PYTHON_BIN:-uv run python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-uv run python -m torch.distributed.run}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
read -r -a TORCHRUN_CMD <<< "${TORCHRUN_BIN}"

STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
VALIDATE_EVERY="${VALIDATE_EVERY:-100}"
LOG_EVERY="${LOG_EVERY:-50}"

mkdir -p "${ROOT}/configs/mlp" "${ROOT}/logs" "${ROOT}/models/mlp" "${ROOT}/figures/loss/mlp"

SUMMARY_CSV="${ROOT}/summary.csv"
FIGURES_DIR="${ROOT}/figures"
LOSS_DIR="${ROOT}/figures/loss/mlp"

make_config() {
  local width="$1"
  local config_path="$2"
  local out_dir="$3"
  local seed="$4"
  "${PYTHON_CMD[@]}" - "${CONFIG}" "${config_path}" "${out_dir}" "${width}" "${seed}" "${STEPS}" "${BATCH_SIZE}" "${LR}" "${WEIGHT_DECAY}" "${VALIDATE_EVERY}" "${LOG_EVERY}" <<'PY'
from pathlib import Path
import sys
import yaml

base_path, out_path, out_dir, width, seed, steps, batch_size, lr, weight_decay, validate_every, log_every = sys.argv[1:]
config = yaml.safe_load(Path(base_path).read_text())
config["output_dir"] = out_dir
config["seed"] = int(seed)
config["device"] = "cuda"
config.setdefault("model", {})["hidden_layers"] = [int(width), int(width)]
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

run_single() {
  local gpu="$1"
  local width="$2"
  local seed="$3"
  local out_dir="${ROOT}/models/mlp/width_${width}/rep_01"
  local config_path="${ROOT}/configs/mlp/train_raw_mlp_classifier_width${width}_rep01.yaml"
  local log_path="${ROOT}/logs/mlp_width${width}_rep01_gpu${gpu}.log"
  make_config "${width}" "${config_path}" "${out_dir}" "${seed}"
  echo "==> train width=${width} rep=01 on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PYTHON_CMD[@]}" scripts/model_experiment/train_raw_mlp_disease_classifier.py \
    --config "${config_path}" \
    --device cuda \
    --summary-csv "${SUMMARY_CSV}" \
    --figures-dir "${FIGURES_DIR}" \
    --loss-dir "${LOSS_DIR}" 2>&1 | tee "${log_path}"
}

run_ddp() {
  local gpus="$1"
  local nproc="$2"
  local width="$3"
  local seed="$4"
  local port="$5"
  local gpu_label="${gpus//,/}"
  local out_dir="${ROOT}/models/mlp/width_${width}/rep_01"
  local config_path="${ROOT}/configs/mlp/train_raw_mlp_classifier_width${width}_rep01.yaml"
  local log_path="${ROOT}/logs/mlp_width${width}_rep01_gpu${gpu_label}_ddp.log"
  make_config "${width}" "${config_path}" "${out_dir}" "${seed}"
  echo "==> train width=${width} rep=01 DDP on GPUs ${gpus}"
  CUDA_VISIBLE_DEVICES="${gpus}" PYTHONUNBUFFERED=1 "${TORCHRUN_CMD[@]}" \
    --nnodes 1 \
    --nproc-per-node "${nproc}" \
    --master-addr 127.0.0.1 \
    --master-port "${port}" \
    scripts/model_experiment/train_raw_mlp_disease_classifier.py \
      --config "${config_path}" \
      --device cuda \
      --summary-csv "${SUMMARY_CSV}" \
      --figures-dir "${FIGURES_DIR}" \
      --loss-dir "${LOSS_DIR}" 2>&1 | tee "${log_path}"
}

pids=()

(
  run_single 0 128 1128
  run_single 0 256 1256
  run_single 0 512 1512
  run_single 0 1024 2024
) &
pids+=("$!")

(
  run_single 1 2048 3048
) &
pids+=("$!")

(
  run_ddp "2,3" 2 4096 5096 29541
) &
pids+=("$!")

(
  run_ddp "4,5,6,7" 4 8192 9192 29542
) &
pids+=("$!")

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "done: ${ROOT}"
echo "summary: ${SUMMARY_CSV}"
echo "figures: ${FIGURES_DIR}"
