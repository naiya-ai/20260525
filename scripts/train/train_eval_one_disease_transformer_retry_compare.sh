#!/usr/bin/env bash
set -euo pipefail

DISEASE="${DISEASE:-diabetes}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
RUN_NAME="${RUN_NAME:-transformer_retry_${DISEASE}_$(date +%Y%m%d_%H%M%S)}"

CONFIG="${CONFIG:-configs/train/train_conditional_vae_transformer.yaml}"
VARIANT="${VARIANT:-transformer_retry}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/conditional_vae_${VARIANT}/${RUN_NAME}}"
EVAL_ROOT="${EVAL_ROOT:-outputs/eval_${VARIANT}/${RUN_NAME}/${DATASET_NAME}}"
COMPARE_ROOT="${COMPARE_ROOT:-outputs/eval_comparison_${VARIANT}/${RUN_NAME}/${DATASET_NAME}}"

# Existing result sources. These are read only; only ${VARIANT} is trained.
BASELINE_SUMMARY="${BASELINE_SUMMARY:-outputs/eval_baseline_20260519/${DATASET_NAME}/summary.csv}"
TRANSFORMER_SUMMARY="${TRANSFORMER_SUMMARY:-outputs/eval_comparison_transformer/transformer_decoder_masked_prior_500/${DATASET_NAME}/summary.csv}"

DEVICE="${DEVICE:-cuda}"
GPUS="${GPUS:-1 2 3}"
EVAL_GPU="${EVAL_GPU:-1}"
PARALLEL="${PARALLEL:-0}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
STEPS="${STEPS:-500}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"

echo "==> One-disease transformer retry comparison"
echo "==> Disease: ${DISEASE}"
echo "==> Dataset: ${DATASET_NAME}"
echo "==> Run: ${RUN_NAME}"
echo "==> Config: ${CONFIG}"
echo "==> Retry variant: ${VARIANT}"
echo "==> Existing baseline summary: ${BASELINE_SUMMARY}"
echo "==> Existing transformer summary: ${TRANSFORMER_SUMMARY}"
echo "==> Output root: ${OUTPUT_ROOT}"
echo "==> Eval root: ${EVAL_ROOT}"
echo "==> Compare root: ${COMPARE_ROOT}"
echo "==> Train GPUs: ${GPUS}"
echo "==> Eval GPU: ${EVAL_GPU}"

if [ ! -f "${BASELINE_SUMMARY}" ]; then
  echo "Missing baseline summary: ${BASELINE_SUMMARY}" >&2
  exit 1
fi
if [ ! -f "${TRANSFORMER_SUMMARY}" ]; then
  echo "Missing transformer summary: ${TRANSFORMER_SUMMARY}" >&2
  exit 1
fi

if [ "${OVERWRITE}" = "1" ]; then
  rm -rf \
    "${OUTPUT_ROOT:?}/${DATASET_NAME}/${DISEASE}" \
    "${EVAL_ROOT:?}" \
    "${COMPARE_ROOT:?}"
fi

export CONFIG DATASET_NAME DATASET_ROOT OUTPUT_ROOT DEVICE GPUS PARALLEL SKIP_EXISTING STEPS
for maybe_override in \
  BATCH_SIZE \
  LEARNING_RATE \
  LR_WARMUP_STEPS \
  GRADIENT_ACCUMULATION_STEPS \
  BETA \
  NUM_WORKERS \
  LOG_EVERY \
  VALIDATE_EVERY \
  CHECKPOINT_EVERY \
  MAX_ROWS_PER_SPLIT
do
  if [ -n "${!maybe_override:-}" ]; then
    export "${maybe_override}"
  fi
done

./scripts/train/train_transformer_cvae_diseases.sh "${DISEASE}"

mkdir -p "${EVAL_ROOT}/logs"
checkpoint="${OUTPUT_ROOT}/${DATASET_NAME}/${DISEASE}/checkpoint_best.pt"
if [ ! -f "${checkpoint}" ]; then
  echo "Missing retry checkpoint: ${checkpoint}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${EVAL_GPU}" uv run python scripts/eval/evaluate_cvae_prior_probability.py \
  --checkpoint "${checkpoint}" \
  --target-group "${DISEASE}" \
  --variant "${VARIANT}" \
  --output-dir "${EVAL_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --device "${DEVICE}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --num-samples "${NUM_SAMPLES}" \
  --threshold-strategy validation_balanced_accuracy \
  > "${EVAL_ROOT}/logs/${DISEASE}.log" 2>&1

DISEASE="${DISEASE}" \
VARIANT="${VARIANT}" \
EVAL_ROOT="${EVAL_ROOT}" \
COMPARE_ROOT="${COMPARE_ROOT}" \
BASELINE_SUMMARY="${BASELINE_SUMMARY}" \
TRANSFORMER_SUMMARY="${TRANSFORMER_SUMMARY}" \
uv run python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

disease = os.environ["DISEASE"]
variant = os.environ["VARIANT"]
eval_root = Path(os.environ["EVAL_ROOT"])
compare_root = Path(os.environ["COMPARE_ROOT"])
baseline_summary = Path(os.environ["BASELINE_SUMMARY"])
transformer_summary = Path(os.environ["TRANSFORMER_SUMMARY"])
compare_root.mkdir(parents=True, exist_ok=True)

metrics_path = eval_root / f"{variant}_{disease}_metrics.json"
data = json.loads(metrics_path.read_text())
sensitivity = float(data["sensitivity"])
specificity = float(data["specificity"])
retry = pd.DataFrame(
    [
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
    ]
)
retry.to_csv(eval_root / "summary.csv", index=False)

baseline = pd.read_csv(baseline_summary)
catboost = baseline[(baseline["disease"] == disease) & (baseline["variant"] == "catboost")]
cvae = baseline[(baseline["disease"] == disease) & (baseline["variant"] == "cvae")]

transformer = pd.read_csv(transformer_summary)
transformer = transformer[
    (transformer["disease"] == disease) & (transformer["variant"] == "transformer")
]

combined = pd.concat([catboost, cvae, transformer, retry], ignore_index=True)
order = ["catboost", "cvae", "transformer", variant]
missing = [name for name in order if name not in set(combined["variant"])]
if missing:
    raise RuntimeError(f"Missing comparison rows for variants: {missing}")

combined["variant"] = pd.Categorical(combined["variant"], categories=order, ordered=True)
combined = combined.sort_values("variant").reset_index(drop=True)
combined.to_csv(compare_root / "summary.csv", index=False)

wide = combined.set_index("variant")[["sensitivity", "specificity", "balanced_accuracy"]].T
wide[f"{variant}_minus_catboost"] = wide[variant] - wide["catboost"]
wide[f"{variant}_minus_cvae"] = wide[variant] - wide["cvae"]
wide[f"{variant}_minus_transformer"] = wide[variant] - wide["transformer"]
wide.to_csv(compare_root / f"{disease}_metrics_wide.csv")

plot_path = compare_root / f"{disease}_metric_comparison.png"
metric_specs = [
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("balanced_accuracy", "Balanced accuracy"),
]
fig, ax = plt.subplots(figsize=(10.5, 5.4))
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
