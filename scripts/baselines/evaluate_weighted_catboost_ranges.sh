#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/catboost_weighted_range_compare/$(date +%Y%m%d_%H%M%S)}"
TASK_TYPE="${TASK_TYPE:-GPU}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
THREAD_COUNT="${THREAD_COUNT:-8}"
SEED="${SEED:-43}"
ITERATIONS="${ITERATIONS:-2000}"
LEARNING_RATE="${LEARNING_RATE:-0.03}"
DEPTH="${DEPTH:-6}"
L2_LEAF_REG="${L2_LEAF_REG:-3.0}"
RANDOM_STRENGTH="${RANDOM_STRENGTH:-1.0}"
BAGGING_TEMPERATURE="${BAGGING_TEMPERATURE:-1.0}"
EARLY_STOPPING_ROUNDS="${EARLY_STOPPING_ROUNDS:-100}"
VERBOSE_EVAL="${VERBOSE_EVAL:-200}"
CLASS_WEIGHT="${CLASS_WEIGHT:-inverse_prevalence}"

DATASETS=(${DATASETS:-harmonized_knhanes_2024 harmonized_knhanes_2013_2024 harmonized_knhanes_1998_2024 harmonized_knhanes_1998_2024_plus_nhanes_1988_2023})
DISEASES=(${DISEASES:-diabetes hypertension dyslipidemia liver_disease hepatitis_b hepatitis_c kidney_disease anemia})

mkdir -p "${OUTPUT_ROOT}/logs"

echo "==> Weighted CatBoost range comparison"
echo "==> output=${OUTPUT_ROOT}"
echo "==> datasets=${DATASETS[*]}"
echo "==> diseases=${DISEASES[*]}"
echo "==> task_type=${TASK_TYPE}; gpus=${GPUS[*]}; seed=${SEED}"

tasks=()
for dataset_name in "${DATASETS[@]}"; do
  for disease in "${DISEASES[@]}"; do
    tasks+=("${dataset_name}|${disease}")
  done
done

run_task() {
  local task="$1"
  local gpu="$2"
  IFS='|' read -r dataset_name disease <<< "${task}"
  local output_dir="${OUTPUT_ROOT}/${dataset_name}/weighted_catboost"
  local variant="weighted_catboost"
  local metrics="${output_dir}/${variant}_${disease}_metrics.json"
  local log="${OUTPUT_ROOT}/logs/${dataset_name}_${disease}_gpu${gpu}.log"

  if [ -f "${metrics}" ]; then
    echo "skip dataset=${dataset_name} disease=${disease}: already exists"
    return 0
  fi

  mkdir -p "${output_dir}"
  echo "run dataset=${dataset_name} disease=${disease} gpu=${gpu}"
  args=(
    uv run python scripts/baselines/train_eval_catboost_disease.py
    --dataset-root "${DATASET_ROOT}"
    --dataset-name "${dataset_name}"
    --target-group "${disease}"
    --variant "${variant}"
    --output-dir "${output_dir}"
    --class-weight "${CLASS_WEIGHT}"
    --iterations "${ITERATIONS}"
    --learning-rate "${LEARNING_RATE}"
    --depth "${DEPTH}"
    --l2-leaf-reg "${L2_LEAF_REG}"
    --random-strength "${RANDOM_STRENGTH}"
    --bagging-temperature "${BAGGING_TEMPERATURE}"
    --early-stopping-rounds "${EARLY_STOPPING_ROUNDS}"
    --task-type "${TASK_TYPE}"
    --thread-count "${THREAD_COUNT}"
    --seed "${SEED}"
    --verbose-eval "${VERBOSE_EVAL}"
  )
  if [ "${TASK_TYPE}" = "GPU" ]; then
    args+=(--devices "${gpu}")
  fi
  "${args[@]}" > "${log}" 2>&1
}

parallelism=1
if [ "${TASK_TYPE}" = "GPU" ]; then
  parallelism="${#GPUS[@]}"
fi

failed=0
for offset in $(seq 0 "${parallelism}" "$((${#tasks[@]} - 1))"); do
  pids=()
  for idx in $(seq 0 "$((parallelism - 1))"); do
    task_index=$((offset + idx))
    if [ "${task_index}" -ge "${#tasks[@]}" ]; then
      break
    fi
    gpu="0"
    if [ "${TASK_TYPE}" = "GPU" ]; then
      gpu="${GPUS[$idx]}"
    fi
    run_task "${tasks[$task_index]}" "${gpu}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
done

if [ "${failed}" -ne 0 ]; then
  echo "One or more CatBoost runs failed. See ${OUTPUT_ROOT}/logs/*.log" >&2
  exit 1
fi

uv run python - "${OUTPUT_ROOT}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/weighted_catboost/*_metrics.json")):
    data = json.loads(path.read_text())
    rows.append({
        "dataset_name": data.get("dataset_name"),
        "target_group": data.get("target_group"),
        "variant": data.get("variant"),
        "auroc": data.get("auroc"),
        "sensitivity": data.get("sensitivity"),
        "specificity": data.get("specificity"),
        "n_evaluable": data.get("n_evaluable"),
        "tp": data.get("tp"),
        "tn": data.get("tn"),
        "fp": data.get("fp"),
        "fn": data.get("fn"),
        "missing_label": data.get("missing_label"),
        "positive_class_weight": data.get("positive_class_weight"),
        "metrics_path": str(path),
    })

summary = root / "weighted_catboost_range_summary.csv"
fieldnames = [
    "dataset_name",
    "target_group",
    "variant",
    "auroc",
    "sensitivity",
    "specificity",
    "n_evaluable",
    "tp",
    "tn",
    "fp",
    "fn",
    "missing_label",
    "positive_class_weight",
    "metrics_path",
]
with summary.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"wrote {summary}")
PY

echo "==> done: ${OUTPUT_ROOT}"
