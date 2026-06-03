#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/train/train_raw_mlp_disease_classifier_diabetes_2024_train25_valid25_test50.yaml}"
ROOT="${ROOT:-model_experiment/classification_2024_train25_valid25_test50_lr1e-3_3k}"
PYTHON_BIN="${PYTHON_BIN:-uv run python}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

WIDTHS=(${WIDTHS:-4 8 16 32 64 128 256 512})
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
DROPOUT="${DROPOUT:-0}"
NORMALIZATION="${NORMALIZATION:-none}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-0}"
CLASS_WEIGHT="${CLASS_WEIGHT:-balanced}"
VALIDATE_EVERY="${VALIDATE_EVERY:-100}"
LOG_EVERY="${LOG_EVERY:-50}"

if [[ "${#WIDTHS[@]}" -ne "${#GPUS[@]}" ]]; then
  echo "WIDTHS and GPUS must have the same length." >&2
  exit 1
fi

mkdir -p "${ROOT}/logs" "${ROOT}/models/mlp" "${ROOT}/figures/loss/mlp" "${ROOT}/figures/loss"

SUMMARY_CSV="${ROOT}/summary.csv"
FIGURES_DIR="${ROOT}/figures"
LOSS_DIR="${ROOT}/figures/loss/mlp"
PIDS=()

for index in "${!WIDTHS[@]}"; do
  width="${WIDTHS[$index]}"
  gpu="${GPUS[$index]}"
  out_dir="${ROOT}/models/mlp/width_${width}/rep_01"
  log_path="${ROOT}/logs/mlp_width${width}_gpu${gpu}.log"
  seed="$((3000 + width))"

  echo "==> launch width=${width} on GPU ${gpu}, batch=${BATCH_SIZE}, lr=${LR}, steps=${STEPS}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PYTHON_CMD[@]}" \
      scripts/model_experiment/train_raw_mlp_disease_classifier.py \
        --config "${CONFIG}" \
        --hidden-layers "${width}" "${width}" \
        --normalization "${NORMALIZATION}" \
        --dropout "${DROPOUT}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --learning-rate "${LR}" \
        --steps "${STEPS}" \
        --batch-size "${BATCH_SIZE}" \
        --gradient-clip-norm "${GRADIENT_CLIP_NORM}" \
        --class-weight "${CLASS_WEIGHT}" \
        --log-every "${LOG_EVERY}" \
        --validate-every "${VALIDATE_EVERY}" \
        --checkpoint-every "${STEPS}" \
        --seed "${seed}" \
        --device cuda \
        --output-dir "${out_dir}" \
        --summary-csv "${SUMMARY_CSV}" \
        --figures-dir "${FIGURES_DIR}" \
        --loss-dir "${LOSS_DIR}" 2>&1 | tee "${log_path}"
  ) &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

"${PYTHON_CMD[@]}" scripts/model_experiment/plot_raw_mlp_loss_by_width.py \
  --root "${ROOT}" \
  --output-dir "${ROOT}/figures/loss" \
  --prefix train_valid_loss_by_width

echo "done: ${ROOT}"
echo "summary: ${SUMMARY_CSV}"
echo "loss figures: ${ROOT}/figures/loss"
