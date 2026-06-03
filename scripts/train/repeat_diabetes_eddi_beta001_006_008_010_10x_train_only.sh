#!/usr/bin/env bash
set -euo pipefail

BETAS=(${BETAS:-0.001 0.002 0.003 0.004 0.005 0.006 0.008 0.010})
REPEATS="${REPEATS:-10}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
CONFIG="${CONFIG:-configs/train/train_conditional_vae_eddi_z2_beta003_diabetes.yaml}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
DISEASE="${DISEASE:-diabetes}"
BASE_SEED="${BASE_SEED:-20260524}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
STEPS="${STEPS:-2000}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
BETA_WARMUP_STEPS="${BETA_WARMUP_STEPS:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
VALIDATE_EVERY="${VALIDATE_EVERY:-10}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
SWEEP_ID="${SWEEP_ID:-eddi_beta001_006_008_010_10x_train_$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-outputs/repeats/${SWEEP_ID}}"

if [ "${DISEASE}" != "diabetes" ]; then
  echo "This beta sweep is diabetes-only. Got DISEASE=${DISEASE}" >&2
  exit 1
fi

mkdir -p "${ROOT}/logs"

echo "==> Diabetes EDDI CVAE beta train-only sweep"
echo "==> betas=${BETAS[*]}"
echo "==> repeats=${REPEATS}; total_runs=$((${#BETAS[@]} * REPEATS))"
echo "==> gpus=${GPUS[*]}"
echo "==> config=${CONFIG}"
echo "==> dataset=${DATASET_NAME}; disease=${DISEASE}"
echo "==> batch_size=${BATCH_SIZE}; steps=${STEPS}; lr=${LEARNING_RATE}; beta_warmup_steps=${BETA_WARMUP_STEPS}"
echo "==> root=${ROOT}"

tasks=()
for beta in "${BETAS[@]}"; do
  beta_tag="$(uv run python - "${beta}" <<'PY'
import sys
print(f"{int(round(float(sys.argv[1]) * 1000)):03d}")
PY
)"
  for rep in $(seq 1 "${REPEATS}"); do
    seed="$(uv run python - "${BASE_SEED}" "${beta_tag}" "${rep}" <<'PY'
import sys
base = int(sys.argv[1])
tag = int(sys.argv[2])
rep = int(sys.argv[3])
print(base + tag * 100 + rep)
PY
)"
    tasks+=("${beta}|${beta_tag}|${rep}|${seed}")
  done
done

status_file="${ROOT}/train_status.tsv"
printf "beta\tbeta_tag\trep\tseed\tgpu\texit_code\trun_dir\n" > "${status_file}"

run_task() {
  local task="$1"
  local gpu="$2"
  IFS='|' read -r beta beta_tag rep seed <<< "${task}"
  local run_name="eddi_beta${beta_tag}_rep${rep}_seed${seed}_${SWEEP_ID}"
  local output_root="outputs/conditional_vae_eddi_beta${beta_tag}/${run_name}"
  local run_dir="${output_root}/${DATASET_NAME}/${DISEASE}"
  local log_path="${ROOT}/logs/beta${beta_tag}_rep${rep}_gpu${gpu}.log"

  {
    echo "==> beta=${beta} beta_tag=${beta_tag} rep=${rep} seed=${seed} gpu=${gpu}"
    echo "==> run_dir=${run_dir}"
    BETA="${beta}" \
    BETA_TAG="${beta_tag}" \
    RUN_NAME="${run_name}" \
    OUTPUT_ROOT="${output_root}" \
    CONFIG="${CONFIG}" \
    DATASET_NAME="${DATASET_NAME}" \
    DATASET_ROOT="${DATASET_ROOT}" \
    DISEASE="${DISEASE}" \
    GPU="${gpu}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    STEPS="${STEPS}" \
    LEARNING_RATE="${LEARNING_RATE}" \
    BETA_WARMUP_STEPS="${BETA_WARMUP_STEPS}" \
    SEED="${seed}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    LOG_EVERY="${LOG_EVERY}" \
    VALIDATE_EVERY="${VALIDATE_EVERY}" \
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
    ./scripts/train/train_diabetes_eddi_z2_beta.sh
  } > "${log_path}" 2>&1
  local code=$?
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
