#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-model_experiment/classification_ddp_no_norm_lr3e-4_100k}"
CONFIG="${CONFIG:-configs/train/train_raw_mlp_disease_classifier_diabetes.yaml}"
PYTHON_BIN="${PYTHON_BIN:-uv run python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-uv run python -m torch.distributed.run}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
read -r -a TORCHRUN_CMD <<< "${TORCHRUN_BIN}"

STEPS="${STEPS:-100000}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
NORMALIZATION="${NORMALIZATION:-none}"
VALIDATE_EVERY="${VALIDATE_EVERY:-100}"
LOG_EVERY="${LOG_EVERY:-50}"

SUMMARY_CSV="${ROOT}/summary.csv"
FIGURES_DIR="${ROOT}/figures"
LOSS_DIR="${ROOT}/figures/loss/mlp"
COMBINED_LOSS_DIR="${ROOT}/figures/loss"

mkdir -p "${ROOT}/logs" "${ROOT}/models/mlp" "${LOSS_DIR}" "${COMBINED_LOSS_DIR}"

run_single() {
  local gpu="$1"
  local width="$2"
  local batch_size="$3"
  local out_dir="${ROOT}/models/mlp/width_${width}/rep_01"
  local log_path="${ROOT}/logs/mlp_width${width}_rep01_gpu${gpu}_single_none.log"
  echo "==> train width=${width} single GPU=${gpu} batch=${batch_size} effective_batch=${batch_size}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PYTHON_CMD[@]}" scripts/model_experiment/train_raw_mlp_disease_classifier.py \
    --config "${CONFIG}" \
    --device cuda \
    --output-dir "${out_dir}" \
    --hidden-layers "${width}" "${width}" \
    --normalization "${NORMALIZATION}" \
    --batch-size "${batch_size}" \
    --learning-rate "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --steps "${STEPS}" \
    --validate-every "${VALIDATE_EVERY}" \
    --log-every "${LOG_EVERY}" \
    --summary-csv "${SUMMARY_CSV}" \
    --figures-dir "${FIGURES_DIR}" \
    --loss-dir "${LOSS_DIR}" 2>&1 | tee "${log_path}"
}

run_ddp() {
  local gpus="$1"
  local nproc="$2"
  local width="$3"
  local batch_size="$4"
  local port="$5"
  local gpu_label="${gpus//,/}"
  local out_dir="${ROOT}/models/mlp/width_${width}/rep_01"
  local log_path="${ROOT}/logs/mlp_width${width}_rep01_gpu${gpu_label}_ddp_none.log"
  echo "==> train width=${width} DDP GPUs=${gpus} per_gpu_batch=${batch_size} effective_batch=$((batch_size * nproc))"
  CUDA_VISIBLE_DEVICES="${gpus}" PYTHONUNBUFFERED=1 "${TORCHRUN_CMD[@]}" \
    --nnodes 1 \
    --nproc-per-node "${nproc}" \
    --master-addr 127.0.0.1 \
    --master-port "${port}" \
    scripts/model_experiment/train_raw_mlp_disease_classifier.py \
      --config "${CONFIG}" \
      --device cuda \
      --output-dir "${out_dir}" \
      --hidden-layers "${width}" "${width}" \
      --normalization "${NORMALIZATION}" \
      --batch-size "${batch_size}" \
      --learning-rate "${LR}" \
      --weight-decay "${WEIGHT_DECAY}" \
      --steps "${STEPS}" \
      --validate-every "${VALIDATE_EVERY}" \
      --log-every "${LOG_EVERY}" \
      --summary-csv "${SUMMARY_CSV}" \
      --figures-dir "${FIGURES_DIR}" \
      --loss-dir "${LOSS_DIR}" 2>&1 | tee "${log_path}"
}

pids=()

run_single 0 2048 512 &
pids+=("$!")

run_single 1 4096 512 &
pids+=("$!")

run_ddp "2,3" 2 8192 256 29720 &
pids+=("$!")

run_ddp "4,5,6,7" 4 16384 128 29721 &
pids+=("$!")

for pid in "${pids[@]}"; do
  wait "${pid}"
done

"${PYTHON_CMD[@]}" scripts/model_experiment/plot_raw_mlp_loss_by_width.py \
  --root "${ROOT}" \
  --output-dir "${COMBINED_LOSS_DIR}" \
  --prefix train_valid_loss_by_width

echo "done: ${ROOT}"
echo "summary: ${SUMMARY_CSV}"
echo "combined loss: ${COMBINED_LOSS_DIR}/train_valid_loss_by_width.png"
