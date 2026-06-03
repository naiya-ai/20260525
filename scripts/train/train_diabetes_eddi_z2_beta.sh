#!/usr/bin/env bash
set -euo pipefail

BETA="${BETA:-0.003}"
BETA_TAG="${BETA_TAG:-$(uv run python - "${BETA}" <<'PY'
import sys
value = float(sys.argv[1])
print(f"{int(round(value * 1000)):03d}")
PY
)}"
RUN_NAME="${RUN_NAME:-eddi_z2_beta${BETA_TAG}_diabetes_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/conditional_vae_eddi_z2_beta${BETA_TAG}/${RUN_NAME}}"
CONFIG="${CONFIG:-configs/train/train_conditional_vae_eddi_z2_beta003_diabetes.yaml}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
DISEASE="${DISEASE:-diabetes}"
GPU="${GPU:-1}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
STEPS="${STEPS:-2000}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
BETA_WARMUP_STEPS="${BETA_WARMUP_STEPS:-0}"
BETA_WARMUP_START_STEP="${BETA_WARMUP_START_STEP:-0}"
SEED="${SEED:-}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
VALIDATE_EVERY="${VALIDATE_EVERY:-10}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
DISABLE_EARLY_STOPPING="${DISABLE_EARLY_STOPPING:-0}"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/${DISEASE}"
LOG_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/logs"
LOG_PATH="${LOG_DIR}/${DISEASE}_eddi_z2_beta${BETA_TAG}.log"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export PYTHONUNBUFFERED

mkdir -p "${LOG_DIR}"

echo "==> Training EDDI CVAE z2 beta${BETA_TAG}"
echo "==> run=${RUN_NAME}"
echo "==> dataset=${DATASET_NAME}; disease=${DISEASE}"
echo "==> GPU=${GPU}"
echo "==> batch_size=${BATCH_SIZE}"
echo "==> lr=${LEARNING_RATE}; beta=${BETA}; beta_warmup_steps=${BETA_WARMUP_STEPS}; beta_warmup_start_step=${BETA_WARMUP_START_STEP}"
echo "==> disable_early_stopping=${DISABLE_EARLY_STOPPING}"
echo "==> seed=${SEED:-config-default}"
echo "==> output=${OUTPUT_DIR}"
echo "==> log=${LOG_PATH}"

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/train/train_conditional_vae.py \
  --config "${CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --target-group "${DISEASE}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --beta "${BETA}" \
  --beta-warmup-steps "${BETA_WARMUP_STEPS}" \
  --beta-warmup-start-step "${BETA_WARMUP_START_STEP}" \
  --num-workers "${NUM_WORKERS}" \
  --log-every "${LOG_EVERY}" \
  --validate-every "${VALIDATE_EVERY}" \
  --checkpoint-every "${CHECKPOINT_EVERY}" \
  $(if [ "${DISABLE_EARLY_STOPPING}" = "1" ]; then echo "--disable-early-stopping"; fi) \
  ${SEED:+--seed "${SEED}"} \
  > "${LOG_PATH}" 2>&1

echo "==> Done. Output: ${OUTPUT_DIR}"
