#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-transformer_d512_l6_z2_basecvae_except_lr1e5_diabetes_allgpu_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/conditional_vae_transformer_d512/${RUN_NAME}}"
CONFIG="${CONFIG:-configs/train/train_conditional_vae_transformer_d512_diabetes.yaml}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
DISEASE="${DISEASE:-diabetes}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"
DEVICE="${DEVICE:-cuda}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-64}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
STEPS="${STEPS:-2000}"
LEARNING_RATE="${LEARNING_RATE:-0.00001}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
LR_SCHEDULE="${LR_SCHEDULE:-constant}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-0.0}"
BETA="${BETA:-0.01}"
SOURCE_MASK_DROPOUT="${SOURCE_MASK_DROPOUT:-0.0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
VALIDATE_EVERY="${VALIDATE_EVERY:-10}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/${DISEASE}"
LOG_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/logs"
LOG_PATH="${LOG_DIR}/${DISEASE}_transformer_d512_ddp.log"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export PYTHONUNBUFFERED

mkdir -p "${LOG_DIR}"

echo "==> Training diabetes transformer d512"
echo "==> run=${RUN_NAME}"
echo "==> dataset=${DATASET_NAME}; disease=${DISEASE}"
echo "==> GPUs=${GPUS}; nproc=${NPROC}"
echo "==> per_gpu_batch=${PER_GPU_BATCH_SIZE}; accumulation=${GRADIENT_ACCUMULATION_STEPS}"
echo "==> effective_batch=$((PER_GPU_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC))"
echo "==> lr=${LEARNING_RATE}; warmup=${LR_WARMUP_STEPS}; schedule=${LR_SCHEDULE}; min_lr=${MIN_LEARNING_RATE}"
echo "==> beta=${BETA}; source_mask_dropout=${SOURCE_MASK_DROPOUT}"
echo "==> output=${OUTPUT_DIR}"
echo "==> log=${LOG_PATH}"

CUDA_VISIBLE_DEVICES="${GPUS}" uv run python -m torch.distributed.run \
  --standalone \
  --nproc_per_node "${NPROC}" \
  scripts/train/train_conditional_vae.py \
    --config "${CONFIG}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-name "${DATASET_NAME}" \
    --target-group "${DISEASE}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --steps "${STEPS}" \
    --batch-size "${PER_GPU_BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning-rate "${LEARNING_RATE}" \
    --lr-warmup-steps "${LR_WARMUP_STEPS}" \
    --lr-schedule "${LR_SCHEDULE}" \
    --min-learning-rate "${MIN_LEARNING_RATE}" \
    --source-mask-dropout "${SOURCE_MASK_DROPOUT}" \
    --beta "${BETA}" \
    --num-workers "${NUM_WORKERS}" \
    --log-every "${LOG_EVERY}" \
    --validate-every "${VALIDATE_EVERY}" \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    > "${LOG_PATH}" 2>&1

echo "==> Done. Output: ${OUTPUT_DIR}"
