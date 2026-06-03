#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/train/train_conditional_vae_transformer.yaml}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/conditional_vae_transformer}"
DEVICE="${DEVICE:-cuda}"
GPUS="${GPUS:-0 1 2 3}"
PARALLEL="${PARALLEL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/${DATASET_NAME}/logs}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${ACCUMULATION_STEPS:-}}"

DEFAULT_DISEASES=(
  diabetes
  hypertension
  dyslipidemia
  liver_disease
  hepatitis_b
  hepatitis_c
  kidney_disease
  anemia
)

if [ "$#" -gt 0 ]; then
  DISEASES=("$@")
else
  DISEASES=("${DEFAULT_DISEASES[@]}")
fi

mkdir -p "${LOG_DIR}"

PENDING_DISEASES=()
for disease in "${DISEASES[@]}"; do
  output_dir="${OUTPUT_ROOT}/${DATASET_NAME}/${disease}"
  if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${output_dir}/checkpoint_best.pt" ]; then
    echo "==> Skipping transformer CVAE: dataset=${DATASET_NAME} disease=${disease} checkpoint exists"
  else
    PENDING_DISEASES+=("${disease}")
  fi
done

train_one() {
  local disease="$1"
  local gpu="$2"
  output_dir="${OUTPUT_ROOT}/${DATASET_NAME}/${disease}"

  echo "==> Training transformer CVAE: dataset=${DATASET_NAME} disease=${disease} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python scripts/train/train_conditional_vae.py \
    --config "${CONFIG}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-name "${DATASET_NAME}" \
    --target-group "${disease}" \
    --output-dir "${output_dir}" \
    --device "${DEVICE}" \
    ${STEPS:+--steps "${STEPS}"} \
    ${BATCH_SIZE:+--batch-size "${BATCH_SIZE}"} \
    ${LEARNING_RATE:+--learning-rate "${LEARNING_RATE}"} \
    ${LR_WARMUP_STEPS:+--lr-warmup-steps "${LR_WARMUP_STEPS}"} \
    ${GRADIENT_ACCUMULATION_STEPS:+--gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"} \
    ${BETA:+--beta "${BETA}"} \
    ${NUM_WORKERS:+--num-workers "${NUM_WORKERS}"} \
    ${LOG_EVERY:+--log-every "${LOG_EVERY}"} \
    ${VALIDATE_EVERY:+--validate-every "${VALIDATE_EVERY}"} \
    ${CHECKPOINT_EVERY:+--checkpoint-every "${CHECKPOINT_EVERY}"} \
    ${MAX_ROWS_PER_SPLIT:+--max-rows-per-split "${MAX_ROWS_PER_SPLIT}"}
}

if [ "${PARALLEL}" = "1" ]; then
  read -r -a GPU_LIST <<< "${GPUS}"
  if [ "${#GPU_LIST[@]}" -eq 0 ]; then
    echo "GPUS must contain at least one GPU id." >&2
    exit 1
  fi

  pids=()
  for i in "${!PENDING_DISEASES[@]}"; do
    disease="${PENDING_DISEASES[$i]}"
    gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
    train_one "${disease}" "${gpu}" > "${LOG_DIR}/${disease}.log" 2>&1 &
    pids+=("$!")

    if [ "${#pids[@]}" -eq "${#GPU_LIST[@]}" ]; then
      wait "${pids[@]}"
      pids=()
    fi
  done

  if [ "${#pids[@]}" -gt 0 ]; then
    wait "${pids[@]}"
  fi
else
  first_gpu="${GPUS%% *}"
  for disease in "${PENDING_DISEASES[@]}"; do
    train_one "${disease}" "${first_gpu}"
  done
fi
