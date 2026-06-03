#!/usr/bin/env bash
set -euo pipefail

DISEASE="${DISEASE:-diabetes}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
RUN_NAME="${RUN_NAME:-transformer_oldstruct_size_stack_${DISEASE}_$(date +%Y%m%d_%H%M%S)}"

BASELINE_SUMMARY="${BASELINE_SUMMARY:-outputs/eval_baseline_20260519/${DATASET_NAME}/summary.csv}"
OLD_TRANSFORMER_SUMMARY="${OLD_TRANSFORMER_SUMMARY:-outputs/eval_comparison_transformer/transformer_decoder_masked_prior_500/${DATASET_NAME}/summary.csv}"
COMPARE_ROOT="${COMPARE_ROOT:-outputs/eval_comparison_transformer_oldstruct_size_stack/${RUN_NAME}/${DATASET_NAME}}"

DEVICE="${DEVICE:-cuda}"
TRAIN_GPU="${TRAIN_GPU:-1}"
EVAL_GPU="${EVAL_GPU:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
OVERWRITE="${OVERWRITE:-0}"
STEPS="${STEPS:-500}"

X2_CONFIG="${X2_CONFIG:-configs/train/train_conditional_vae_transformer_oldstruct_x2.yaml}"
XHALF_CONFIG="${XHALF_CONFIG:-configs/train/train_conditional_vae_transformer_oldstruct_xhalf.yaml}"
X2_VARIANT="${X2_VARIANT:-cvae-transformerx2-oldstruct}"
XHALF_VARIANT="${XHALF_VARIANT:-cvae-transformerx0.5-oldstruct}"
X2_OUTPUT_ROOT="${X2_OUTPUT_ROOT:-outputs/conditional_vae_${X2_VARIANT}/${RUN_NAME}}"
XHALF_OUTPUT_ROOT="${XHALF_OUTPUT_ROOT:-outputs/conditional_vae_${XHALF_VARIANT}/${RUN_NAME}}"
X2_EVAL_ROOT="${X2_EVAL_ROOT:-outputs/eval_${X2_VARIANT}/${RUN_NAME}/${DATASET_NAME}}"
XHALF_EVAL_ROOT="${XHALF_EVAL_ROOT:-outputs/eval_${XHALF_VARIANT}/${RUN_NAME}/${DATASET_NAME}}"

echo "==> Old-structure transformer size stack"
echo "==> Disease: ${DISEASE}"
echo "==> Dataset: ${DATASET_NAME}"
echo "==> Run: ${RUN_NAME}"
echo "==> Train GPU: ${TRAIN_GPU}"
echo "==> Eval GPU: ${EVAL_GPU}"
echo "==> Baseline summary: ${BASELINE_SUMMARY}"
echo "==> Old transformer summary: ${OLD_TRANSFORMER_SUMMARY}"
echo "==> Compare root: ${COMPARE_ROOT}"

for required in "${BASELINE_SUMMARY}" "${OLD_TRANSFORMER_SUMMARY}" "${X2_CONFIG}" "${XHALF_CONFIG}"; do
  if [ ! -f "${required}" ]; then
    echo "Missing required file: ${required}" >&2
    exit 1
  fi
done

if [ "${OVERWRITE}" = "1" ]; then
  rm -rf \
    "${X2_OUTPUT_ROOT:?}/${DATASET_NAME}/${DISEASE}" \
    "${XHALF_OUTPUT_ROOT:?}/${DATASET_NAME}/${DISEASE}" \
    "${X2_EVAL_ROOT:?}" \
    "${XHALF_EVAL_ROOT:?}" \
    "${COMPARE_ROOT:?}"
fi

train_one() {
  local config="$1"
  local variant="$2"
  local output_root="$3"
  local output_dir="${output_root}/${DATASET_NAME}/${DISEASE}"
  local log_dir="${output_root}/${DATASET_NAME}/logs"
  mkdir -p "${log_dir}"

  echo "==> Training ${variant}: output=${output_dir}"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" uv run python scripts/train/train_conditional_vae.py \
    --config "${config}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-name "${DATASET_NAME}" \
    --target-group "${DISEASE}" \
    --output-dir "${output_dir}" \
    --device "${DEVICE}" \
    --steps "${STEPS}" \
    ${BATCH_SIZE:+--batch-size "${BATCH_SIZE}"} \
    ${GRADIENT_ACCUMULATION_STEPS:+--gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"} \
    ${LEARNING_RATE:+--learning-rate "${LEARNING_RATE}"} \
    ${BETA:+--beta "${BETA}"} \
    ${NUM_WORKERS:+--num-workers "${NUM_WORKERS}"} \
    ${LOG_EVERY:+--log-every "${LOG_EVERY}"} \
    ${VALIDATE_EVERY:+--validate-every "${VALIDATE_EVERY}"} \
    ${CHECKPOINT_EVERY:+--checkpoint-every "${CHECKPOINT_EVERY}"} \
    ${MAX_ROWS_PER_SPLIT:+--max-rows-per-split "${MAX_ROWS_PER_SPLIT}"} \
    > "${log_dir}/${DISEASE}_${variant}.log" 2>&1
}

eval_one() {
  local variant="$1"
  local output_root="$2"
  local eval_root="$3"
  local checkpoint="${output_root}/${DATASET_NAME}/${DISEASE}/checkpoint_best.pt"
  mkdir -p "${eval_root}/logs"
  if [ ! -f "${checkpoint}" ]; then
    echo "Missing checkpoint for ${variant}: ${checkpoint}" >&2
    exit 1
  fi

  echo "==> Evaluating ${variant}: checkpoint=${checkpoint}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" uv run python scripts/eval/evaluate_cvae_prior_probability.py \
    --checkpoint "${checkpoint}" \
    --target-group "${DISEASE}" \
    --variant "${variant}" \
    --output-dir "${eval_root}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-name "${DATASET_NAME}" \
    --device "${DEVICE}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --num-samples "${NUM_SAMPLES}" \
    --threshold-strategy validation_balanced_accuracy \
    > "${eval_root}/logs/${DISEASE}.log" 2>&1
}

train_one "${X2_CONFIG}" "${X2_VARIANT}" "${X2_OUTPUT_ROOT}"
eval_one "${X2_VARIANT}" "${X2_OUTPUT_ROOT}" "${X2_EVAL_ROOT}"

train_one "${XHALF_CONFIG}" "${XHALF_VARIANT}" "${XHALF_OUTPUT_ROOT}"
eval_one "${XHALF_VARIANT}" "${XHALF_OUTPUT_ROOT}" "${XHALF_EVAL_ROOT}"

DISEASE="${DISEASE}" \
X2_VARIANT="${X2_VARIANT}" \
XHALF_VARIANT="${XHALF_VARIANT}" \
X2_EVAL_ROOT="${X2_EVAL_ROOT}" \
XHALF_EVAL_ROOT="${XHALF_EVAL_ROOT}" \
COMPARE_ROOT="${COMPARE_ROOT}" \
BASELINE_SUMMARY="${BASELINE_SUMMARY}" \
OLD_TRANSFORMER_SUMMARY="${OLD_TRANSFORMER_SUMMARY}" \
uv run python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

disease = os.environ["DISEASE"]
x2_variant = os.environ["X2_VARIANT"]
xhalf_variant = os.environ["XHALF_VARIANT"]
x2_eval_root = Path(os.environ["X2_EVAL_ROOT"])
xhalf_eval_root = Path(os.environ["XHALF_EVAL_ROOT"])
compare_root = Path(os.environ["COMPARE_ROOT"])
baseline_summary = Path(os.environ["BASELINE_SUMMARY"])
old_transformer_summary = Path(os.environ["OLD_TRANSFORMER_SUMMARY"])
compare_root.mkdir(parents=True, exist_ok=True)

def row_from_metrics(eval_root: Path, variant: str) -> pd.DataFrame:
    data = json.loads((eval_root / f"{variant}_{disease}_metrics.json").read_text())
    sensitivity = float(data["sensitivity"])
    specificity = float(data["specificity"])
    return pd.DataFrame([
        {
            "variant": variant,
            "disease": data["target_group"],
            "threshold": data["threshold"],
            "tp": data["tp"],
            "tn": data["tn"],
            "fp": data["fp"],
            "fn": data["fn"],
            "missing_label": data["missing_label"],
            "missing_prediction": data["missing_prediction"],
            "n_evaluable": data["n_evaluable"],
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": 0.5 * (sensitivity + specificity),
        }
    ])

x2 = row_from_metrics(x2_eval_root, x2_variant)
xhalf = row_from_metrics(xhalf_eval_root, xhalf_variant)
x2.to_csv(x2_eval_root / "summary.csv", index=False)
xhalf.to_csv(xhalf_eval_root / "summary.csv", index=False)

baseline = pd.read_csv(baseline_summary)
catboost = baseline[(baseline["disease"] == disease) & (baseline["variant"] == "catboost")].copy()
cvae = baseline[(baseline["disease"] == disease) & (baseline["variant"] == "cvae")].copy()
cvae["variant"] = "cvae-mlp"

old = pd.read_csv(old_transformer_summary)
old_transformer = old[(old["disease"] == disease) & (old["variant"] == "transformer")].copy()
old_transformer["variant"] = "cvae-transformer"
x2["variant"] = "cvae-transformerx2-oldstruct"
xhalf["variant"] = "cvae-transformerx0.5-oldstruct"

combined = pd.concat([catboost, cvae, old_transformer, x2, xhalf], ignore_index=True)
order = [
    "catboost",
    "cvae-mlp",
    "cvae-transformer",
    "cvae-transformerx2-oldstruct",
    "cvae-transformerx0.5-oldstruct",
]
missing = [name for name in order if name not in set(combined["variant"])]
if missing:
    raise RuntimeError(f"Missing comparison rows for variants: {missing}")
combined["variant"] = pd.Categorical(combined["variant"], categories=order, ordered=True)
combined = combined.sort_values("variant").reset_index(drop=True)
combined.to_csv(compare_root / "summary.csv", index=False)

wide = combined.set_index("variant")[["sensitivity", "specificity", "balanced_accuracy"]].T
wide["x2_old_minus_transformer"] = wide["cvae-transformerx2-oldstruct"] - wide["cvae-transformer"]
wide["xhalf_old_minus_transformer"] = wide["cvae-transformerx0.5-oldstruct"] - wide["cvae-transformer"]
wide.to_csv(compare_root / f"{disease}_metrics_wide.csv")

plot_path = compare_root / f"{disease}_metric_comparison.png"
metric_specs = [
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("balanced_accuracy", "Balanced accuracy"),
]
fig, ax = plt.subplots(figsize=(12, 5.6))
x = range(len(metric_specs))
width = 0.82 / len(order)
for idx, name in enumerate(order):
    values = [
        float(combined.loc[combined["variant"] == name, metric].iloc[0])
        for metric, _ in metric_specs
    ]
    offset = (idx - (len(order) - 1) / 2) * width
    ax.bar([pos + offset for pos in x], values, width=width, label=name)
ax.set_title(f"{disease} Prediction Metrics")
ax.set_ylabel("score")
ax.set_xticks(list(x))
ax.set_xticklabels([label for _, label in metric_specs])
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.25)
ax.legend(ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig(plot_path, dpi=180)
plt.close(fig)

print(combined[["variant", "sensitivity", "specificity", "balanced_accuracy"]].to_string(index=False))
print("summary:", compare_root / "summary.csv")
print("wide:", compare_root / f"{disease}_metrics_wide.csv")
print("png:", plot_path)
PY

echo "==> Done."
echo "==> Compare directory: ${COMPARE_ROOT}"
