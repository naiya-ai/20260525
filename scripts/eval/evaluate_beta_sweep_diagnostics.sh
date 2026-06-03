#!/usr/bin/env bash
set -euo pipefail

SWEEP_ROOT="${SWEEP_ROOT:-outputs/repeats/eddi_beta001_006_008_010_10x_train_20260524_191745}"
STATUS_FILE="${STATUS_FILE:-${SWEEP_ROOT}/train_status.tsv}"
DISEASE="${DISEASE:-}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
DEVICE="${DEVICE:-cuda}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
BATCH_SIZE="${BATCH_SIZE:-2048}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_ROWS="${MAX_ROWS:-}"
OUT_DIR="${OUT_DIR:-${SWEEP_ROOT}/diagnostics}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint_best.pt}"
PRIOR_SAMPLE_CACHE_DIR="${PRIOR_SAMPLE_CACHE_DIR:-}"
PRIOR_SAMPLE_CACHE_SPLITS=(${PRIOR_SAMPLE_CACHE_SPLITS:-test})

if [ ! -f "${STATUS_FILE}" ]; then
  echo "Missing status file: ${STATUS_FILE}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

if [ -n "${DISEASE}" ]; then
  echo "==> Evaluating completed ${DISEASE} CVAE beta runs"
else
  echo "==> Evaluating completed CVAE beta runs for all target groups in status file"
fi
echo "==> status=${STATUS_FILE}"
echo "==> out=${OUT_DIR}"
echo "==> checkpoint_name=${CHECKPOINT_NAME}"
echo "==> device=${DEVICE}; num_samples=${NUM_SAMPLES}; batch_size=${BATCH_SIZE}"
if [ -n "${PRIOR_SAMPLE_CACHE_DIR}" ]; then
  echo "==> prior_sample_cache_dir=${PRIOR_SAMPLE_CACHE_DIR}; cache_splits=${PRIOR_SAMPLE_CACHE_SPLITS[*]}"
  mkdir -p "${PRIOR_SAMPLE_CACHE_DIR}"
fi
if [ "${DEVICE}" = "cuda" ]; then
  echo "==> gpus=${GPUS[*]}"
fi

tasks=()
while IFS=$'\t' read -r beta beta_tag rep seed gpu exit_code run_dir; do
  if [ -z "${run_dir:-}" ] || [ "${exit_code}" != "0" ]; then
    continue
  fi
  target_group="$(basename "${run_dir}")"
  if [ -n "${DISEASE}" ] && [ "${target_group}" != "${DISEASE}" ]; then
    continue
  fi
  checkpoint="${run_dir}/${CHECKPOINT_NAME}"
  if [ ! -f "${checkpoint}" ]; then
    echo "skip beta=${beta} rep=${rep}: missing ${checkpoint}"
    continue
  fi
  eval_dir="${OUT_DIR}/beta${beta_tag}_rep${rep}_${target_group}"
  result="${eval_dir}/beta${beta_tag}_rep${rep}_${target_group}_diagnostics.json"
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
  echo "eval beta=${beta} rep=${rep} target=${target_group} gpu=${assigned_gpu} checkpoint=${checkpoint}"
  args=(
    uv run python scripts/eval/evaluate_cvae_beta_diagnostics.py
    --checkpoint "${checkpoint}"
    --target-group "${target_group}"
    --variant "beta${beta_tag}_rep${rep}"
    --output-dir "${eval_dir}"
    --dataset-name "${DATASET_NAME}"
    --dataset-root "${DATASET_ROOT}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --num-samples "${NUM_SAMPLES}"
  )
  if [ -n "${PRIOR_SAMPLE_CACHE_DIR}" ]; then
    args+=(
      --prior-sample-cache-dir "${PRIOR_SAMPLE_CACHE_DIR}"
      --prior-sample-cache-splits "${PRIOR_SAMPLE_CACHE_SPLITS[@]}"
    )
  fi
  if [ -n "${MAX_ROWS}" ]; then
    args+=(--max-rows "${MAX_ROWS}")
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
  echo "One or more evaluations failed. See ${OUT_DIR}/beta*_rep*/eval.log" >&2
  exit 1
fi

uv run python - "${STATUS_FILE}" "${OUT_DIR}" "${DISEASE}" "${CHECKPOINT_NAME}" <<'PY'
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

status_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
disease = sys.argv[3]
checkpoint_name = sys.argv[4]

rows = []
with status_path.open(newline="") as file:
    for row in csv.DictReader(file, delimiter="\t"):
        if row["exit_code"] != "0":
            continue
        target_group = Path(row["run_dir"]).name
        if disease and target_group != disease:
            continue
        result_path = out_dir / f"beta{row['beta_tag']}_rep{row['rep']}_{target_group}" / (
            f"beta{row['beta_tag']}_rep{row['rep']}_{target_group}_diagnostics.json"
        )
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        test = result["splits"].get("test", {})
        cal = test.get("coverage_calibration", {})
        recon = test.get("conditional_prior_reconstruction", {})
        kl = test.get("kl_latent_usage", {})
        rows.append({
            "beta": row["beta"],
            "beta_tag": row["beta_tag"],
            "rep": row["rep"],
            "seed": row["seed"],
            "target_group": target_group,
            "checkpoint": str(Path(row["run_dir"]) / checkpoint_name),
            "test_n": test.get("n_rows"),
            "test_auroc": cal.get("auroc"),
            "test_brier": cal.get("brier_score"),
            "test_log_loss": cal.get("log_loss"),
            "test_ece_10": cal.get("ece_10"),
            "test_prior_reconstruction_estimator": recon.get("estimator", "prior_sample_mean"),
            "test_prior_num_samples": recon.get("num_prior_samples", result.get("num_prior_samples")),
            "test_prior_raw_mae": recon.get("raw_num_mae"),
            "test_prior_raw_rmse": recon.get("raw_num_rmse"),
            "test_prior_transformed_mae": recon.get("transformed_num_mae"),
            "test_prior_transformed_rmse": recon.get("transformed_num_rmse"),
            "test_prior_cat_accuracy": recon.get("cat_accuracy"),
            "test_prior_cat_bce": recon.get("cat_bce", recon.get("cat_nll")),
            "test_kl_mean": kl.get("kl_mean"),
            "test_active_units_kl_gt_0_001": kl.get("active_units_kl_gt_0_001"),
            "test_active_units_kl_gt_0_01": kl.get("active_units_kl_gt_0_01"),
            "test_active_units_mu_var_gt_0_01": kl.get("active_units_posterior_mu_var_gt_0_01"),
        })

summary_path = out_dir / "diagnostics_summary.csv"
fields = list(rows[0].keys()) if rows else [
    "beta", "beta_tag", "rep", "seed", "target_group", "checkpoint", "test_n", "test_auroc",
    "test_brier", "test_log_loss", "test_ece_10", "test_prior_reconstruction_estimator",
    "test_prior_num_samples", "test_prior_raw_mae", "test_prior_raw_rmse",
    "test_prior_transformed_mae", "test_prior_transformed_rmse",
    "test_prior_cat_accuracy", "test_prior_cat_bce",
    "test_kl_mean", "test_active_units_kl_gt_0_001", "test_active_units_kl_gt_0_01",
    "test_active_units_mu_var_gt_0_01",
]
with summary_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

numeric_fields = [field for field in fields if field not in {"beta", "beta_tag", "rep", "seed", "target_group", "checkpoint"}]
groups = defaultdict(list)
for row in rows:
    groups[(row["target_group"], row["beta"])].append(row)

beta_rows = []
for (target_group, beta), beta_rows_raw in sorted(groups.items(), key=lambda item: (item[0][0], float(item[0][1]))):
    aggregate = {"target_group": target_group, "beta": beta, "n_reps": len(beta_rows_raw)}
    for field in numeric_fields:
        values = []
        for row in beta_rows_raw:
            value = row.get(field)
            if value in (None, ""):
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass
        aggregate[f"{field}_mean"] = None if not values else sum(values) / len(values)
    beta_rows.append(aggregate)

beta_summary_path = out_dir / "diagnostics_beta_summary.csv"
beta_fields = list(beta_rows[0].keys()) if beta_rows else ["target_group", "beta", "n_reps"]
with beta_summary_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=beta_fields)
    writer.writeheader()
    writer.writerows(beta_rows)

print(f"wrote {summary_path}")
print(f"wrote {beta_summary_path}")
PY
