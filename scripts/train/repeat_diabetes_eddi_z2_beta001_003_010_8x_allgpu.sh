#!/usr/bin/env bash
set -euo pipefail

BETAS=(${BETAS:-0.001 0.003 0.010})
REPEATS="${REPEATS:-8}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
BATCH_SIZE="${BATCH_SIZE:-1024}"
STEPS="${STEPS:-2000}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
DISEASE="${DISEASE:-diabetes}"
BASE_SEED="${BASE_SEED:-20260523}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4096}"
SWEEP_ID="${SWEEP_ID:-eddi_z2_beta001_003_010_8x_$(date +%Y%m%d_%H%M%S)}"
ROOT="outputs/repeats/${SWEEP_ID}"

nvidia-smi

mkdir -p "${ROOT}/logs"

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

run_task() {
  local task="$1"
  local gpu="$2"
  IFS='|' read -r beta beta_tag rep seed <<< "${task}"
  local run_name="eddi_z2_beta${beta_tag}_rep${rep}_seed${seed}_${SWEEP_ID}"
  local output_root="outputs/conditional_vae_eddi_z2_beta${beta_tag}/${run_name}"
  local run_dir="${output_root}/${DATASET_NAME}/${DISEASE}"
  local eval_root="${ROOT}/beta${beta_tag}/rep${rep}"
  local probabilities="${eval_root}/eddi_z2_beta${beta_tag}_rep${rep}_${DISEASE}_probabilities.csv"
  local auroc_json="${eval_root}/auroc.json"
  local log_path="${ROOT}/logs/beta${beta_tag}_rep${rep}_gpu${gpu}.log"

  mkdir -p "${eval_root}"
  {
    echo "==> beta=${beta} rep=${rep} seed=${seed} gpu=${gpu}"
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
    SEED="${seed}" \
    ./scripts/train/train_diabetes_eddi_z2_beta.sh

    local checkpoint="${run_dir}/checkpoint_best.pt"
    CUDA_VISIBLE_DEVICES="${gpu}" uv run python scripts/eval/evaluate_cvae_prior_probability.py \
      --checkpoint "${checkpoint}" \
      --target-group "${DISEASE}" \
      --variant "eddi_z2_beta${beta_tag}_rep${rep}" \
      --output-dir "${eval_root}" \
      --dataset-root "${DATASET_ROOT}" \
      --dataset-name "${DATASET_NAME}" \
      --device cuda \
      --batch-size "${EVAL_BATCH_SIZE}" \
      --num-workers 0 \
      --num-samples "${NUM_SAMPLES}" \
      --save-probabilities

    uv run python - "${probabilities}" "${auroc_json}" "${beta}" "${rep}" "${seed}" "${run_dir}" <<'PY'
import csv
import json
import sys
from pathlib import Path

import numpy as np


def parse_bool(value):
    value = str(value).strip().lower()
    if value in {"true", "1", "yes"}:
        return 1
    if value in {"false", "0", "no"}:
        return 0
    if value in {"", "nan"}:
        return None
    return int(float(value))


def auroc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


prob_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
beta = float(sys.argv[3])
rep = int(sys.argv[4])
seed = int(sys.argv[5])
run_dir = sys.argv[6]
labels = []
scores = []
with prob_path.open(newline="") as file:
    reader = csv.DictReader(file)
    for row in reader:
        label = parse_bool(row.get("label", ""))
        score = row.get("disease_probability", "")
        if label is None or score == "":
            continue
        labels.append(label)
        scores.append(float(score))
result = {
    "beta": beta,
    "rep": rep,
    "seed": seed,
    "run_dir": run_dir,
    "probabilities": str(prob_path),
    "n_evaluable": len(labels),
    "n_positive": int(sum(labels)),
    "auroc": auroc(labels, scores),
}
out_path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
  } > "${log_path}" 2>&1
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

uv run python scripts/analysis/summarize_eddi_z2_beta_repeat_auroc.py \
  --root "${ROOT}" \
  --summary-output "${ROOT}/auroc_summary.csv" \
  --detail-output "${ROOT}/auroc_detail.csv"

echo "==> repeat root: ${ROOT}"
echo "==> summary: ${ROOT}/auroc_summary.csv"
echo "==> detail: ${ROOT}/auroc_detail.csv"

if [ "${failed}" -ne 0 ]; then
  exit 1
fi
