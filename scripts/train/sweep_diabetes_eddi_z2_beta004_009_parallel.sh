#!/usr/bin/env bash
set -euo pipefail

BETAS=(${BETAS:-0.004 0.005 0.006 0.007 0.008 0.009})
GPUS=(${GPUS:-1 2 4 5 6 7})
BATCH_SIZE="${BATCH_SIZE:-1024}"
STEPS="${STEPS:-2000}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
DISEASE="${DISEASE:-diabetes}"
SWEEP_ID="${SWEEP_ID:-eddi_z2_beta004_009_$(date +%Y%m%d_%H%M%S)}"

if [ "${#BETAS[@]}" -gt "${#GPUS[@]}" ]; then
  echo "Need at least as many GPUs as betas: betas=${#BETAS[@]} gpus=${#GPUS[@]}" >&2
  exit 1
fi

nvidia-smi

mkdir -p "outputs/sweeps/${SWEEP_ID}"
status_file="outputs/sweeps/${SWEEP_ID}/status.tsv"
: > "${status_file}"

pids=()
for idx in "${!BETAS[@]}"; do
  beta="${BETAS[$idx]}"
  gpu="${GPUS[$idx]}"
  beta_tag="$(uv run python - "${beta}" <<'PY'
import sys
print(f"{int(round(float(sys.argv[1]) * 1000)):03d}")
PY
)"
  run_name="eddi_z2_beta${beta_tag}_diabetes_${SWEEP_ID}"
  output_root="outputs/conditional_vae_eddi_z2_beta${beta_tag}/${run_name}"
  echo "==> launch beta=${beta} tag=${beta_tag} gpu=${gpu} output=${output_root}"
  (
    set +e
    BETA="${beta}" \
    BETA_TAG="${beta_tag}" \
    RUN_NAME="${run_name}" \
    OUTPUT_ROOT="${output_root}" \
    DATASET_NAME="${DATASET_NAME}" \
    DATASET_ROOT="${DATASET_ROOT}" \
    DISEASE="${DISEASE}" \
    GPU="${gpu}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    STEPS="${STEPS}" \
    LEARNING_RATE="${LEARNING_RATE}" \
    ./scripts/train/train_diabetes_eddi_z2_beta.sh
    code=$?
    printf "%s\t%s\t%s\t%s\n" "${beta}" "${gpu}" "${code}" "${output_root}" >> "${status_file}"
    exit "${code}"
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

echo "==> sweep status: ${status_file}"
cat "${status_file}"

uv run python scripts/analysis/plot_eddi_z2_beta_rc_loss.py \
  --output "outputs/plots/${SWEEP_ID}/diabetes_eddi_z2_beta003_010_reconstruction_loss.png" \
  --summary-output "outputs/plots/${SWEEP_ID}/diabetes_eddi_z2_beta003_010_reconstruction_loss_summary.csv"

if [ "${failed}" -ne 0 ]; then
  exit 1
fi
