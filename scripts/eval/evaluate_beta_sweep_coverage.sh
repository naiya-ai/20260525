#!/usr/bin/env bash
set -euo pipefail

SWEEP_ROOT="${SWEEP_ROOT:-outputs/repeats/eddi_beta001_006_008_010_10x_train_20260524_191745}"
STATUS_FILE="${STATUS_FILE:-${SWEEP_ROOT}/train_status.tsv}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
DEVICE="${DEVICE:-cuda}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_ROWS="${MAX_ROWS:-}"
SPLIT="${SPLIT:-test}"
INTERVAL_LEVELS=(${INTERVAL_LEVELS:-0.90 0.95 0.99})
MIXED_PRECISION="${MIXED_PRECISION:-0}"
OUT_DIR="${OUT_DIR:-${SWEEP_ROOT}/coverage_s1000}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_best.pt}"
PRIOR_SAMPLE_CACHE_DIR="${PRIOR_SAMPLE_CACHE_DIR:-}"

if [ ! -f "${STATUS_FILE}" ]; then
  echo "Missing status file: ${STATUS_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "==> Evaluating CVAE raw-scale numerical coverage"
echo "==> status=${STATUS_FILE}"
echo "==> out=${OUT_DIR}"
echo "==> checkpoint_name=${CHECKPOINT_NAME}"
echo "==> split=${SPLIT}; num_samples=${NUM_SAMPLES}; intervals=${INTERVAL_LEVELS[*]}"
echo "==> device=${DEVICE}; batch_size=${BATCH_SIZE}"
if [ -n "${PRIOR_SAMPLE_CACHE_DIR}" ]; then
  echo "==> prior_sample_cache_dir=${PRIOR_SAMPLE_CACHE_DIR}"
fi
if [ "${DEVICE}" = "cuda" ]; then
  echo "==> gpus=${GPUS[*]}"
fi

tasks=()
while IFS=$'\t' read -r beta beta_tag rep seed gpu exit_code run_dir; do
  if [ -z "${run_dir:-}" ] || [ "${exit_code}" != "0" ]; then
    continue
  fi
  checkpoint="${run_dir}/${CHECKPOINT_NAME}"
  if [ ! -f "${checkpoint}" ]; then
    echo "skip beta=${beta} rep=${rep}: missing ${checkpoint}"
    continue
  fi
  target_group="$(basename "${run_dir}")"
  eval_dir="${OUT_DIR}/beta${beta_tag}_rep${rep}_${target_group}"
  result="${eval_dir}/beta${beta_tag}_rep${rep}_${target_group}_${SPLIT}_coverage.json"
  if [ -f "${result}" ]; then
    echo "skip beta=${beta} rep=${rep} target=${target_group}: already evaluated"
    continue
  fi
  tasks+=("${beta}|${beta_tag}|${rep}|${seed}|${target_group}|${checkpoint}|${eval_dir}")
done < <(tail -n +2 "${STATUS_FILE}")

run_eval() {
  local task="$1"
  local assigned_gpu="$2"
  IFS='|' read -r beta beta_tag rep seed target_group checkpoint eval_dir <<< "${task}"
  mkdir -p "${eval_dir}"
  echo "eval beta=${beta} rep=${rep} target=${target_group} gpu=${assigned_gpu}"
  args=(
    uv run python scripts/eval/evaluate_cvae_coverage.py
    --checkpoint "${checkpoint}"
    --target-group "${target_group}"
    --variant "beta${beta_tag}_rep${rep}"
    --output-dir "${eval_dir}"
    --dataset-name "${DATASET_NAME}"
    --dataset-root "${DATASET_ROOT}"
    --split "${SPLIT}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --num-samples "${NUM_SAMPLES}"
    --interval-levels "${INTERVAL_LEVELS[@]}"
    --seed "${seed}"
    --save-csv
  )
  if [ -n "${PRIOR_SAMPLE_CACHE_DIR}" ]; then
    args+=(
      --prior-sample-cache
      "${PRIOR_SAMPLE_CACHE_DIR}/beta${beta_tag}_rep${rep}_${target_group}_${SPLIT}_prior_samples.npz"
    )
  fi
  if [ -n "${MAX_ROWS}" ]; then
    args+=(--max-rows "${MAX_ROWS}")
  fi
  if [ "${MIXED_PRECISION}" = "1" ]; then
    args+=(--mixed-precision)
  fi
  if [ "${DEVICE}" = "cuda" ]; then
    CUDA_VISIBLE_DEVICES="${assigned_gpu}" "${args[@]}" > "${eval_dir}/eval.log" 2>&1
  else
    "${args[@]}" > "${eval_dir}/eval.log" 2>&1
  fi
}

if [ "${#tasks[@]}" -eq 0 ]; then
  echo "==> no pending completed runs to evaluate"
else
  echo "==> pending_evaluations=${#tasks[@]}"
fi

parallelism=1
if [ "${DEVICE}" = "cuda" ]; then
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
    assigned_gpu="0"
    if [ "${DEVICE}" = "cuda" ]; then
      assigned_gpu="${GPUS[$idx]}"
    fi
    run_eval "${tasks[$task_index]}" "${assigned_gpu}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
done

if [ "${failed}" -ne 0 ]; then
  echo "One or more coverage evaluations failed. See ${OUT_DIR}/*/eval.log" >&2
  exit 1
fi

uv run python - "${STATUS_FILE}" "${OUT_DIR}" "${SPLIT}" "${CHECKPOINT_NAME}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

status_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
split = sys.argv[3]
checkpoint_name = sys.argv[4]

rows = []
with status_path.open(newline="") as file:
    for row in csv.DictReader(file, delimiter="\t"):
        if row["exit_code"] != "0":
            continue
        target_group = Path(row["run_dir"]).name
        result_path = out_dir / f"beta{row['beta_tag']}_rep{row['rep']}_{target_group}" / (
            f"beta{row['beta_tag']}_rep{row['rep']}_{target_group}_{split}_coverage.json"
        )
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        for feature in result["features"]:
            for interval in feature["intervals"]:
                rows.append({
                    "beta": row["beta"],
                    "beta_tag": row["beta_tag"],
                    "rep": row["rep"],
                    "seed": row["seed"],
                    "target_group": result["target_group"],
                    "feature": feature["feature"],
                    "interval_level": interval["interval_level"],
                    "n_observed": interval["n_observed"],
                    "coverage": interval["coverage"],
                    "coverage_error": interval["coverage_error"],
                    "abs_coverage_error": None
                    if interval["coverage_error"] is None
                    else abs(interval["coverage_error"]),
                    "mean_interval_width": interval["mean_interval_width"],
                    "median_interval_width": interval["median_interval_width"],
                    "mean_generated": interval["mean_generated"],
                    "mean_observed": interval["mean_observed"],
                    "mean_bias": interval["mean_bias"],
                    "abs_mean_bias": None
                    if interval["mean_bias"] is None
                    else abs(interval["mean_bias"]),
                    "checkpoint": str(Path(row["run_dir"]) / checkpoint_name),
                })

summary_path = out_dir / "coverage_summary.csv"
fields = list(rows[0].keys()) if rows else [
    "beta", "beta_tag", "rep", "seed", "target_group", "feature", "interval_level",
    "n_observed", "coverage", "coverage_error", "abs_coverage_error",
    "mean_interval_width", "median_interval_width", "mean_generated",
    "mean_observed", "mean_bias", "abs_mean_bias", "checkpoint",
]
with summary_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

groups = defaultdict(list)
for row in rows:
    groups[(row["beta"], row["target_group"], row["feature"], row["interval_level"])].append(row)

aggregate_rows = []
for key, group_rows in sorted(groups.items(), key=lambda item: (float(item[0][0]), item[0][1], item[0][2], float(item[0][3]))):
    beta, target_group, feature, interval_level = key
    aggregate = {
        "beta": beta,
        "target_group": target_group,
        "feature": feature,
        "interval_level": interval_level,
        "n_reps": len(group_rows),
    }
    for field in [
        "coverage",
        "coverage_error",
        "abs_coverage_error",
        "mean_interval_width",
        "median_interval_width",
        "mean_bias",
        "abs_mean_bias",
    ]:
        values = [float(row[field]) for row in group_rows if row[field] not in (None, "")]
        aggregate[f"{field}_mean"] = None if not values else sum(values) / len(values)
    aggregate_rows.append(aggregate)

aggregate_path = out_dir / "coverage_beta_feature_summary.csv"
aggregate_fields = list(aggregate_rows[0].keys()) if aggregate_rows else [
    "beta", "target_group", "feature", "interval_level", "n_reps",
]
with aggregate_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=aggregate_fields)
    writer.writeheader()
    writer.writerows(aggregate_rows)

print(f"wrote {summary_path}")
print(f"wrote {aggregate_path}")
PY
