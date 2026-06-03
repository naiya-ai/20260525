#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/train/train_raw_mlp_disease_classifier_diabetes.yaml}"
ROOT="${ROOT:-model_experiment/classification}"
PYTHON_BIN="${PYTHON_BIN:-uv run python}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
WIDTHS=(${WIDTHS:-256 512 1024 2048 4096 8192})
REPS="${REPS:-5}"
STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
VALIDATE_EVERY="${VALIDATE_EVERY:-100}"
LOG_EVERY="${LOG_EVERY:-50}"

mkdir -p "${ROOT}/configs/mlp" "${ROOT}/logs" "${ROOT}/models/mlp" "${ROOT}/figures/loss/mlp"

SUMMARY_CSV="${ROOT}/summary.csv"
FIGURES_DIR="${ROOT}/figures"
LOSS_DIR="${ROOT}/figures/loss/mlp"

pids=()

wait_for_slot() {
  local max_jobs="$1"
  while (( ${#pids[@]} >= max_jobs )); do
    local next=()
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        next+=("${pid}")
      else
        wait "${pid}"
      fi
    done
    pids=("${next[@]}")
    if (( ${#pids[@]} >= max_jobs )); then
      sleep 5
    fi
  done
}

job_index=0
for width in "${WIDTHS[@]}"; do
  for rep in $(seq 1 "${REPS}"); do
    gpu="${GPUS[$((job_index % ${#GPUS[@]}))]}"
    rep_label="$(printf '%02d' "${rep}")"
    out_dir="${ROOT}/models/mlp/width_${width}/rep_${rep_label}"
    config_path="${ROOT}/configs/mlp/train_raw_mlp_classifier_width${width}_rep${rep_label}.yaml"
    log_path="${ROOT}/logs/mlp_width${width}_rep${rep_label}_gpu${gpu}.log"
    seed="$((1000 + width + rep))"

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
config.setdefault("logging", {})["validate_every"] = int(validate_every)
config.setdefault("logging", {})["log_every"] = int(log_every)
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
Path(out_path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

    echo "==> train raw_mlp_classifier width=${width} rep=${rep_label} on GPU ${gpu}"
    wait_for_slot "${#GPUS[@]}"
    (
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PYTHON_CMD[@]}" scripts/model_experiment/train_raw_mlp_disease_classifier.py \
        --config "${config_path}" \
        --device cuda \
        --summary-csv "${SUMMARY_CSV}" \
        --figures-dir "${FIGURES_DIR}" \
        --loss-dir "${LOSS_DIR}"
    ) 2>&1 | tee "${log_path}" &
    pids+=("$!")
    job_index=$((job_index + 1))
  done
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "done: ${ROOT}"
echo "summary: ${SUMMARY_CSV}"
