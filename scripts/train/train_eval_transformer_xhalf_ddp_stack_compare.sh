#!/usr/bin/env bash
set -euo pipefail

DISEASE="${DISEASE:-diabetes}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
RUN_NAME="${RUN_NAME:-transformer_xhalf_ddp_${DISEASE}_$(date +%Y%m%d_%H%M%S)}"

CONFIG="${CONFIG:-configs/train/train_conditional_vae_transformer_separate_dmodel64_ff256.yaml}"
VARIANT="${VARIANT:-cvae_transformerx0.5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/conditional_vae_${VARIANT}/${RUN_NAME}}"
EVAL_ROOT="${EVAL_ROOT:-outputs/eval_${VARIANT}/${RUN_NAME}/${DATASET_NAME}}"
COMPARE_ROOT="${COMPARE_ROOT:-outputs/eval_comparison_transformer_size_stack/${RUN_NAME}/${DATASET_NAME}}"

# Existing result sources. These are read only; only ${VARIANT} is trained.
BASELINE_SUMMARY="${BASELINE_SUMMARY:-outputs/eval_baseline_20260519/${DATASET_NAME}/summary.csv}"
TRANSFORMER_SUMMARY="${TRANSFORMER_SUMMARY:-outputs/eval_comparison_transformer/transformer_decoder_masked_prior_500/${DATASET_NAME}/summary.csv}"
TRANSFORMER_X2_SUMMARY="${TRANSFORMER_X2_SUMMARY:-outputs/eval_comparison_transformer_separate_dmodel256_ff1024/transformer_retry_ddp_diabetes_20260521_101838/${DATASET_NAME}/summary.csv}"

DEVICE="${DEVICE:-cuda}"
GPUS="${GPUS:-1,2,3}"
NPROC="${NPROC:-3}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-342}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
STEPS="${STEPS:-500}"
EVAL_GPU="${EVAL_GPU:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
OVERWRITE="${OVERWRITE:-0}"

echo "==> Transformer x0.5 DDP size-stack comparison"
echo "==> Disease: ${DISEASE}"
echo "==> Dataset: ${DATASET_NAME}"
echo "==> Run: ${RUN_NAME}"
echo "==> Config: ${CONFIG}"
echo "==> Variant: ${VARIANT}"
echo "==> Baseline summary: ${BASELINE_SUMMARY}"
echo "==> Transformer summary: ${TRANSFORMER_SUMMARY}"
echo "==> Transformer x2 summary: ${TRANSFORMER_X2_SUMMARY}"
echo "==> Output root: ${OUTPUT_ROOT}"
echo "==> Eval root: ${EVAL_ROOT}"
echo "==> Compare root: ${COMPARE_ROOT}"
echo "==> Train GPUs: ${GPUS}; nproc=${NPROC}"
echo "==> Per-GPU batch: ${PER_GPU_BATCH_SIZE}; accumulation=${GRADIENT_ACCUMULATION_STEPS}"
echo "==> Effective total batch: $((PER_GPU_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC))"
echo "==> Eval GPU: ${EVAL_GPU}"

for required in "${CONFIG}" "${BASELINE_SUMMARY}" "${TRANSFORMER_SUMMARY}" "${TRANSFORMER_X2_SUMMARY}"; do
  if [ ! -f "${required}" ]; then
    echo "Missing required file: ${required}" >&2
    exit 1
  fi
done

if [ "${OVERWRITE}" = "1" ]; then
  rm -rf \
    "${OUTPUT_ROOT:?}/${DATASET_NAME}/${DISEASE}" \
    "${EVAL_ROOT:?}" \
    "${COMPARE_ROOT:?}"
fi

output_dir="${OUTPUT_ROOT}/${DATASET_NAME}/${DISEASE}"
log_dir="${OUTPUT_ROOT}/${DATASET_NAME}/logs"
mkdir -p "${log_dir}"

echo "==> Training ${VARIANT} with DDP: output=${output_dir}"
CUDA_VISIBLE_DEVICES="${GPUS}" uv run torchrun \
  --standalone \
  --nproc_per_node "${NPROC}" \
  scripts/train/train_conditional_vae.py \
    --config "${CONFIG}" \
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
    > "${log_dir}/${DISEASE}_${VARIANT}_ddp.log" 2>&1

mkdir -p "${EVAL_ROOT}/logs"
checkpoint="${output_dir}/checkpoint_best.pt"
if [ ! -f "${checkpoint}" ]; then
  echo "Missing x0.5 checkpoint: ${checkpoint}" >&2
  exit 1
fi

echo "==> Evaluating ${VARIANT}: checkpoint=${checkpoint}"
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
TRANSFORMER_X2_SUMMARY="${TRANSFORMER_X2_SUMMARY}" \
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
transformer_x2_summary = Path(os.environ["TRANSFORMER_X2_SUMMARY"])
compare_root.mkdir(parents=True, exist_ok=True)

metrics_path = eval_root / f"{variant}_{disease}_metrics.json"
data = json.loads(metrics_path.read_text())
sensitivity = float(data["sensitivity"])
specificity = float(data["specificity"])
xhalf = pd.DataFrame(
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
xhalf.to_csv(eval_root / "summary.csv", index=False)

baseline = pd.read_csv(baseline_summary)
catboost = baseline[(baseline["disease"] == disease) & (baseline["variant"] == "catboost")].copy()
cvae = baseline[(baseline["disease"] == disease) & (baseline["variant"] == "cvae")].copy()
cvae["variant"] = "cvae-mlp"

transformer = pd.read_csv(transformer_summary)
transformer = transformer[
    (transformer["disease"] == disease) & (transformer["variant"] == "transformer")
].copy()
transformer["variant"] = "cvae-transformer"

x2 = pd.read_csv(transformer_x2_summary)
x2 = x2[x2["disease"] == disease].copy()
x2 = x2[~x2["variant"].isin(["catboost", "cvae", "transformer"])]
if x2.empty:
    raise RuntimeError(f"No x2 row found in {transformer_x2_summary}")
x2 = x2.tail(1).copy()
x2["variant"] = "cvae-transformerx2"
xhalf["variant"] = "cvae-transformerx0.5"

combined = pd.concat([catboost, cvae, transformer, x2, xhalf], ignore_index=True)
order = [
    "catboost",
    "cvae-mlp",
    "cvae-transformer",
    "cvae-transformerx2",
    "cvae-transformerx0.5",
]
missing = [name for name in order if name not in set(combined["variant"])]
if missing:
    raise RuntimeError(f"Missing comparison rows for variants: {missing}")

combined["variant"] = pd.Categorical(combined["variant"], categories=order, ordered=True)
combined = combined.sort_values("variant").reset_index(drop=True)
combined.to_csv(compare_root / "summary.csv", index=False)

wide = combined.set_index("variant")[["sensitivity", "specificity", "balanced_accuracy"]].T
wide["x0.5_minus_transformer"] = wide["cvae-transformerx0.5"] - wide["cvae-transformer"]
wide["x0.5_minus_x2"] = wide["cvae-transformerx0.5"] - wide["cvae-transformerx2"]
wide["x2_minus_transformer"] = wide["cvae-transformerx2"] - wide["cvae-transformer"]
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
ax.legend(ncol=3, loc="upper right")
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
