#!/usr/bin/env bash
set -euo pipefail

DISEASE="${DISEASE:-diabetes}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_2013_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
RUN_NAME="${RUN_NAME:-transformer_large_ddp_diabetes_$(date +%Y%m%d_%H%M%S)}"

LARGE_CONFIG="${LARGE_CONFIG:-configs/train/train_conditional_vae_transformer_large.yaml}"
LARGE_VARIANT="${LARGE_VARIANT:-transformer_large}"
LARGE_OUTPUT_ROOT="${LARGE_OUTPUT_ROOT:-outputs/conditional_vae_${LARGE_VARIANT}/${RUN_NAME}}"
LARGE_EVAL_ROOT="${LARGE_EVAL_ROOT:-outputs/eval_${LARGE_VARIANT}/${RUN_NAME}/${DATASET_NAME}}"

TRANSFORMER_CONFIG="${TRANSFORMER_CONFIG:-configs/train/train_conditional_vae_transformer.yaml}"
TRANSFORMER_VARIANT="${TRANSFORMER_VARIANT:-transformer}"
TRANSFORMER_OUTPUT_ROOT="${TRANSFORMER_OUTPUT_ROOT:-outputs/conditional_vae_${TRANSFORMER_VARIANT}/${RUN_NAME}}"
TRANSFORMER_EVAL_ROOT="${TRANSFORMER_EVAL_ROOT:-outputs/eval_${TRANSFORMER_VARIANT}/${RUN_NAME}/${DATASET_NAME}}"
TRAIN_TRANSFORMER_BASELINE="${TRAIN_TRANSFORMER_BASELINE:-0}"

COMPARE_ROOT="${COMPARE_ROOT:-outputs/eval_comparison_${LARGE_VARIANT}/${RUN_NAME}/${DATASET_NAME}}"
CVAE_BASELINE_SUMMARY="${CVAE_BASELINE_SUMMARY:-outputs/eval_baseline_20260519/${DATASET_NAME}/summary.csv}"
TRANSFORMER_BASELINE_SUMMARY="${TRANSFORMER_BASELINE_SUMMARY:-outputs/eval_comparison_transformer/transformer_decoder_masked_prior_500/${DATASET_NAME}/summary.csv}"

GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
DEVICE="${DEVICE:-cuda}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-256}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
STEPS="${STEPS:-500}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EVAL_GPU="${EVAL_GPU:-0}"
OVERWRITE="${OVERWRITE:-0}"

echo "==> Diabetes transformer-large DDP comparison run: ${RUN_NAME}"
echo "==> Disease: ${DISEASE}"
echo "==> GPUs: ${GPUS}; nproc=${NPROC}"
echo "==> Per-GPU batch: ${PER_GPU_BATCH_SIZE}; accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "==> Effective total batch: $((PER_GPU_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC))"
echo "==> Large config: ${LARGE_CONFIG}"
echo "==> Large prior mode should be separate."
echo "==> CVAE baseline summary: ${CVAE_BASELINE_SUMMARY}"
echo "==> Transformer baseline summary: ${TRANSFORMER_BASELINE_SUMMARY}"
echo "==> Train transformer baseline: ${TRAIN_TRANSFORMER_BASELINE}"
echo "==> Only transformer_large is trained by default; other rows are loaded from summaries."

if [ "${OVERWRITE}" = "1" ]; then
  rm -rf \
    "${LARGE_OUTPUT_ROOT:?}/${DATASET_NAME}/${DISEASE}" \
    "${LARGE_EVAL_ROOT:?}" \
    "${COMPARE_ROOT:?}"
  if [ "${TRAIN_TRANSFORMER_BASELINE}" = "1" ]; then
    rm -rf \
      "${TRANSFORMER_OUTPUT_ROOT:?}/${DATASET_NAME}/${DISEASE}" \
      "${TRANSFORMER_EVAL_ROOT:?}"
  fi
fi

train_ddp() {
  local config="$1"
  local variant="$2"
  local output_root="$3"
  local output_dir="${output_root}/${DATASET_NAME}/${DISEASE}"
  local log_dir="${output_root}/${DATASET_NAME}/logs"
  mkdir -p "${log_dir}"

  echo "==> Training ${variant} with DDP: output=${output_dir}"
  CUDA_VISIBLE_DEVICES="${GPUS}" uv run torchrun \
    --standalone \
    --nproc_per_node "${NPROC}" \
    scripts/train/train_conditional_vae.py \
      --config "${config}" \
      --dataset-root "${DATASET_ROOT}" \
      --dataset-name "${DATASET_NAME}" \
      --target-group "${DISEASE}" \
      --output-dir "${output_dir}" \
      --device "${DEVICE}" \
      --steps "${STEPS}" \
      --batch-size "${PER_GPU_BATCH_SIZE}" \
      --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
      ${LEARNING_RATE:+--learning-rate "${LEARNING_RATE}"} \
      ${LR_WARMUP_STEPS:+--lr-warmup-steps "${LR_WARMUP_STEPS}"} \
      ${LR_SCHEDULE:+--lr-schedule "${LR_SCHEDULE}"} \
      ${MIN_LEARNING_RATE:+--min-learning-rate "${MIN_LEARNING_RATE}"} \
      ${BETA:+--beta "${BETA}"} \
      ${NUM_WORKERS:+--num-workers "${NUM_WORKERS}"} \
      ${LOG_EVERY:+--log-every "${LOG_EVERY}"} \
      ${VALIDATE_EVERY:+--validate-every "${VALIDATE_EVERY}"} \
      ${CHECKPOINT_EVERY:+--checkpoint-every "${CHECKPOINT_EVERY}"} \
      ${MAX_ROWS_PER_SPLIT:+--max-rows-per-split "${MAX_ROWS_PER_SPLIT}"} \
      > "${log_dir}/${DISEASE}_${variant}_ddp.log" 2>&1
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
    --device cuda \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --num-samples "${NUM_SAMPLES}" \
    --threshold-strategy validation_balanced_accuracy \
    > "${eval_root}/logs/${DISEASE}.log" 2>&1

  VARIANT="${variant}" EVAL_ROOT="${eval_root}" uv run python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

variant = os.environ["VARIANT"]
eval_root = Path(os.environ["EVAL_ROOT"])
rows = []
for path in sorted(eval_root.glob(f"{variant}_*_metrics.json")):
    data = json.loads(path.read_text())
    sensitivity = float(data["sensitivity"])
    specificity = float(data["specificity"])
    rows.append(
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
    )
pd.DataFrame(rows).to_csv(eval_root / "summary.csv", index=False)
PY
}

if [ "${TRAIN_TRANSFORMER_BASELINE}" = "1" ]; then
  train_ddp "${TRANSFORMER_CONFIG}" "${TRANSFORMER_VARIANT}" "${TRANSFORMER_OUTPUT_ROOT}"
  eval_one "${TRANSFORMER_VARIANT}" "${TRANSFORMER_OUTPUT_ROOT}" "${TRANSFORMER_EVAL_ROOT}"
  TRANSFORMER_BASELINE_SUMMARY="${TRANSFORMER_EVAL_ROOT}/summary.csv"
fi

train_ddp "${LARGE_CONFIG}" "${LARGE_VARIANT}" "${LARGE_OUTPUT_ROOT}"
eval_one "${LARGE_VARIANT}" "${LARGE_OUTPUT_ROOT}" "${LARGE_EVAL_ROOT}"

mkdir -p "${COMPARE_ROOT}"
DISEASE="${DISEASE}" \
LARGE_VARIANT="${LARGE_VARIANT}" \
TRANSFORMER_VARIANT="${TRANSFORMER_VARIANT}" \
LARGE_EVAL_ROOT="${LARGE_EVAL_ROOT}" \
COMPARE_ROOT="${COMPARE_ROOT}" \
CVAE_BASELINE_SUMMARY="${CVAE_BASELINE_SUMMARY}" \
TRANSFORMER_BASELINE_SUMMARY="${TRANSFORMER_BASELINE_SUMMARY}" \
uv run python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

disease = os.environ["DISEASE"]
large_variant = os.environ["LARGE_VARIANT"]
transformer_variant = os.environ["TRANSFORMER_VARIANT"]
large_eval_root = Path(os.environ["LARGE_EVAL_ROOT"])
compare_root = Path(os.environ["COMPARE_ROOT"])
cvae_summary = Path(os.environ["CVAE_BASELINE_SUMMARY"])
transformer_summary = Path(os.environ["TRANSFORMER_BASELINE_SUMMARY"])

for path in [cvae_summary, transformer_summary, large_eval_root / "summary.csv"]:
    if not path.exists():
        raise FileNotFoundError(path)

baseline = pd.read_csv(cvae_summary)
catboost = baseline[
    (baseline["disease"] == disease) & (baseline["variant"] == "catboost")
]
cvae = baseline[(baseline["disease"] == disease) & (baseline["variant"] == "cvae")]

transformer = pd.read_csv(transformer_summary)
transformer = transformer[
    (transformer["disease"] == disease)
    & (transformer["variant"] == transformer_variant)
]

large = pd.read_csv(large_eval_root / "summary.csv")
large = large[(large["disease"] == disease) & (large["variant"] == large_variant)]

combined = pd.concat([catboost, cvae, transformer, large], ignore_index=True)
expected = ["catboost", "cvae", transformer_variant, large_variant]
missing = [variant for variant in expected if variant not in set(combined["variant"])]
if missing:
    raise RuntimeError(f"Missing comparison rows for variants: {missing}")

combined["variant"] = pd.Categorical(combined["variant"], categories=expected, ordered=True)
combined = combined.sort_values("variant").reset_index(drop=True)
combined.to_csv(compare_root / "summary.csv", index=False)

wide = combined.set_index("variant")[
    ["sensitivity", "specificity", "balanced_accuracy"]
].T
wide[f"{large_variant}_minus_cvae"] = wide[large_variant] - wide["cvae"]
wide[f"{large_variant}_minus_{transformer_variant}"] = (
    wide[large_variant] - wide[transformer_variant]
)
wide[f"{large_variant}_minus_catboost"] = wide[large_variant] - wide["catboost"]
wide.to_csv(compare_root / "diabetes_metrics_wide.csv")

metric_plot = compare_root / "diabetes_metric_comparison.png"
metric_specs = [
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("balanced_accuracy", "Balanced accuracy"),
]
fig, ax = plt.subplots(figsize=(10.5, 5.4))
x = range(len(metric_specs))
width = 0.82 / len(expected)
for idx, variant in enumerate(expected):
    values = [
        float(combined.loc[combined["variant"] == variant, metric].iloc[0])
        for metric, _ in metric_specs
    ]
    offset = (idx - (len(expected) - 1) / 2) * width
    positions = [position + offset for position in x]
    ax.bar(positions, values, width=width, label=variant)
ax.set_title("Diabetes Prediction Metrics")
ax.set_ylabel("score")
ax.set_xticks(list(x))
ax.set_xticklabels([label for _, label in metric_specs])
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.25)
ax.legend(ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig(metric_plot, dpi=180)
plt.close(fig)

print("==> Diabetes comparison")
print(combined[["variant", "sensitivity", "specificity", "balanced_accuracy"]].to_string(index=False))
print("==> Wrote:", compare_root / "summary.csv")
print("==> Wrote:", compare_root / "diabetes_metrics_wide.csv")
print("==> Wrote:", metric_plot)
PY

echo "==> Done."
echo "==> Comparison directory: ${COMPARE_ROOT}"
