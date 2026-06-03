#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/train/train_ddpm_mlp.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ddpm_mlp_6diseases_knhanes}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
GPUS_CSV="${GPUS:-4}"
STEPS="${STEPS:-}"

DISEASES=(
  diabetes
  hypertension
  dyslipidemia
  liver_disease
  kidney_disease
  anemia
)

IFS=',' read -r -a GPUS <<<"${GPUS_CSV}"
mkdir -p "${OUTPUT_ROOT}/logs"

pids=()
cleanup() {
  if [[ "${#pids[@]}" -gt 0 ]]; then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for index in "${!DISEASES[@]}"; do
  disease="${DISEASES[$index]}"
  gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
  output_dir="${OUTPUT_ROOT}/${disease}"
  log_path="${OUTPUT_ROOT}/logs/${disease}.log"
  cmd=(
    env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1
    uv run python -u scripts/train/train_ddpm_mlp.py
    --config "${CONFIG}"
    --dataset-name "${DATASET_NAME}"
    --target-group "${disease}"
    --output-dir "${output_dir}"
  )
  if [[ -n "${STEPS}" ]]; then
    cmd+=(--steps "${STEPS}")
  fi
  printf 'launch ddpm disease=%s gpu=%s log=%s\n' "${disease}" "${gpu}" "${log_path}"
  (
    set -o pipefail
    "${cmd[@]}" 2>&1 | tee "${log_path}"
  ) &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  disease="${DISEASES[$index]}"
  if wait "${pids[$index]}"; then
    printf 'done ddpm disease=%s\n' "${disease}"
  else
    status="$?"
    printf 'failed ddpm disease=%s status=%s log=%s\n' "${disease}" "${status}" "${OUTPUT_ROOT}/logs/${disease}.log" >&2
    failed=1
  fi
done

exit "${failed}"
