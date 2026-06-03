#!/usr/bin/env bash
set -euo pipefail

BETAS=(${BETAS:-0.001 0.01 0.1 1.0})
DISEASES=(${DISEASES:-diabetes hypertension dyslipidemia liver_disease hepatitis_b hepatitis_c kidney_disease anemia})
REPEATS="${REPEATS:-5}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
CONFIG="${CONFIG:-configs/train/train_conditional_vae_eddi_z2_beta003_diabetes.yaml}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
BASE_SEED="${BASE_SEED:-20260525}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
STEPS="${STEPS:-2000}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
BETA_WARMUP_STEPS="${BETA_WARMUP_STEPS:-1000}"
BETA_WARMUP_START_STEP="${BETA_WARMUP_START_STEP:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
VALIDATE_EVERY="${VALIDATE_EVERY:-10}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
DISABLE_EARLY_STOPPING="${DISABLE_EARLY_STOPPING:-1}"
SWEEP_ID="${SWEEP_ID:-cvae_beta_selection_all_diseases_$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-outputs/repeats/${SWEEP_ID}}"

mkdir -p "${ROOT}/logs"

echo "==> CVAE beta-selection all-disease train sweep"
echo "==> betas=${BETAS[*]}"
echo "==> diseases=${DISEASES[*]}"
echo "==> repeats=${REPEATS}; total_runs=$((${#BETAS[@]} * ${#DISEASES[@]} * REPEATS))"
echo "==> gpus=${GPUS[*]}"
echo "==> config=${CONFIG}"
echo "==> dataset=${DATASET_NAME}"
echo "==> batch_size=${BATCH_SIZE}; steps=${STEPS}; lr=${LEARNING_RATE}; beta_warmup_steps=${BETA_WARMUP_STEPS}; beta_warmup_start_step=${BETA_WARMUP_START_STEP}; disable_early_stopping=${DISABLE_EARLY_STOPPING}"
echo "==> root=${ROOT}"

tasks=()
for disease_idx in "${!DISEASES[@]}"; do
  disease="${DISEASES[$disease_idx]}"
  for beta in "${BETAS[@]}"; do
    beta_tag="$(awk -v beta="${beta}" 'BEGIN { printf "%03d", int(beta * 1000 + 0.5) }')"
    for rep in $(seq 1 "${REPEATS}"); do
      seed=$((BASE_SEED + disease_idx * 1000000 + 10#${beta_tag} * 100 + rep))
      tasks+=("${disease}|${beta}|${beta_tag}|${rep}|${seed}")
    done
  done
done

status_file="${ROOT}/train_status.tsv"
printf "beta\tbeta_tag\trep\tseed\tgpu\texit_code\trun_dir\n" > "${status_file}"

run_task() {
  local task="$1"
  local gpu="$2"
  IFS='|' read -r disease beta beta_tag rep seed <<< "${task}"
  local run_name="${disease}_beta${beta_tag}_rep${rep}_seed${seed}_${SWEEP_ID}"
  local output_root="outputs/conditional_vae_beta_selection/beta${beta_tag}/${run_name}"
  local run_dir="${output_root}/${DATASET_NAME}/${disease}"
  local log_path="${ROOT}/logs/${disease}_beta${beta_tag}_rep${rep}_gpu${gpu}.log"

  {
    echo "==> disease=${disease} beta=${beta} beta_tag=${beta_tag} rep=${rep} seed=${seed} gpu=${gpu}"
    echo "==> run_dir=${run_dir}"
    set +e
    BETA="${beta}" \
    BETA_TAG="${beta_tag}" \
    RUN_NAME="${run_name}" \
    OUTPUT_ROOT="${output_root}" \
    CONFIG="${CONFIG}" \
    DATASET_NAME="${DATASET_NAME}" \
    DATASET_ROOT="${DATASET_ROOT}" \
    DISEASE="${disease}" \
    GPU="${gpu}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    STEPS="${STEPS}" \
    LEARNING_RATE="${LEARNING_RATE}" \
    BETA_WARMUP_STEPS="${BETA_WARMUP_STEPS}" \
    BETA_WARMUP_START_STEP="${BETA_WARMUP_START_STEP}" \
    SEED="${seed}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    LOG_EVERY="${LOG_EVERY}" \
    VALIDATE_EVERY="${VALIDATE_EVERY}" \
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
    DISABLE_EARLY_STOPPING="${DISABLE_EARLY_STOPPING}" \
    ./scripts/train/train_diabetes_eddi_z2_beta.sh
    code=$?
    set -e
  } > "${log_path}" 2>&1
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${beta}" "${beta_tag}" "${rep}" "${seed}" "${gpu}" "${code}" "${run_dir}" \
    >> "${status_file}"
  return "${code}"
}

failed=0
for offset in $(seq 0 "${#GPUS[@]}" "$((${#tasks[@]} - 1))"); do
  pids=()
  for idx in "${!GPUS[@]}"; do
    task_index=$((offset + idx))
    if [ "${task_index}" -ge "${#tasks[@]}" ]; then
      break
    fi
    task="${tasks[$task_index]}"
    gpu="${GPUS[$idx]}"
    echo "==> launch ${task} on gpu=${gpu}"
    run_task "${task}" "${gpu}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
done

echo "==> status: ${status_file}"
cat "${status_file}"
echo "==> root: ${ROOT}"

if [ "${failed}" -ne 0 ]; then
  exit 1
fi
