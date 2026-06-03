#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Train disease-specific DDPM models for six non-hepatitis diseases.

Environment overrides:
  CONFIG           Default: configs/train/train_ddpm_mlp.yaml
  DATASET_NAME     Default: harmonized_knhanes_1998_2024
  OUTPUT_ROOT      Default: outputs/ddpm_disease_specific_6diseases_${DATASET_NAME}_${RUN_ID}
  RUN_ID           Default: current timestamp
  GPU              Default: 4
  STEPS            Optional train steps override
  BATCH_SIZE       Optional batch size override
  NUM_TIMESTEPS    Optional diffusion timestep override

Example:
  GPU=4 ./scripts/train/run_ddpm_disease_specific_6diseases.sh
EOF
  exit 0
fi

CONFIG="${CONFIG:-configs/train/train_ddpm_mlp.yaml}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ddpm_disease_specific_6diseases_${DATASET_NAME}_${RUN_ID}}"
GPU="${GPU:-4}"
STEPS="${STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-}"

DISEASES=(
  diabetes
  hypertension
  dyslipidemia
  liver_disease
  kidney_disease
  anemia
)

mkdir -p "${OUTPUT_ROOT}/logs"

cleanup() {
  if [[ -n "${current_pid:-}" ]]; then
    kill "${current_pid}" 2>/dev/null || true
    wait "${current_pid}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

echo "DDPM disease-specific 6-disease run"
echo "  config=${CONFIG}"
echo "  dataset=${DATASET_NAME}"
echo "  output_root=${OUTPUT_ROOT}"
echo "  gpu=${GPU}"
echo "  diseases=${DISEASES[*]}"

failed=0
for disease in "${DISEASES[@]}"; do
  output_dir="${OUTPUT_ROOT}/${disease}"
  log_path="${OUTPUT_ROOT}/logs/${disease}.log"
  cmd=(
    env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1
    uv run python -u scripts/train/train_ddpm_mlp.py
    --config "${CONFIG}"
    --dataset-name "${DATASET_NAME}"
    --target-group "${disease}"
    --output-dir "${output_dir}"
  )
  if [[ -n "${STEPS}" ]]; then
    cmd+=(--steps "${STEPS}")
  fi
  if [[ -n "${BATCH_SIZE}" ]]; then
    cmd+=(--batch-size "${BATCH_SIZE}")
  fi
  if [[ -n "${NUM_TIMESTEPS}" ]]; then
    cmd+=(--num-timesteps "${NUM_TIMESTEPS}")
  fi
  printf 'launch disease=%s gpu=%s output=%s log=%s\n' "${disease}" "${GPU}" "${output_dir}" "${log_path}"
  (
    set -o pipefail
    "${cmd[@]}" 2>&1 | tee "${log_path}"
  ) &
  current_pid="$!"
  if wait "${current_pid}"; then
    printf 'done disease=%s\n' "${disease}"
  else
    status="$?"
    printf 'failed disease=%s status=%s log=%s\n' "${disease}" "${status}" "${OUTPUT_ROOT}/logs/${disease}.log" >&2
    failed=1
  fi
  current_pid=""
done

echo "${OUTPUT_ROOT}" > ddpm_disease_specific_6diseases_latest_output.txt
exit "${failed}"
