#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/train/train_conditional_vae_eddi_masked_prior_layer_norm.yaml}"
VARIANT="${VARIANT:-cvae_masked_prior_layer_norm}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/conditional_vae_${VARIANT}/${RUN_NAME}}"
EVAL_ROOT="${EVAL_ROOT:-outputs/eval_${VARIANT}/${RUN_NAME}/${DATASET_NAME}}"
COMPARE_ROOT="${COMPARE_ROOT:-outputs/eval_comparison_${VARIANT}/${RUN_NAME}/${DATASET_NAME}}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-outputs/eval_comparison_cvae_masked_prior/cvae_masked_prior_vs_original_2000/${DATASET_NAME}/summary.csv}"
DEVICE="${DEVICE:-cuda}"
EVAL_DEVICE="${EVAL_DEVICE:-${DEVICE}}"
GPUS="${GPUS:-0 1 2 3}"
EVAL_GPUS="${EVAL_GPUS:-${GPUS}}"
EVAL_PARALLEL="${EVAL_PARALLEL:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

DISEASES=(
  diabetes
  hypertension
  dyslipidemia
  liver_disease
  hepatitis_b
  hepatitis_c
  kidney_disease
  anemia
)

echo "==> EDDI CVAE layer-norm masked-prior run: ${RUN_NAME}"
echo "==> Config: ${CONFIG}"
echo "==> Variant: ${VARIANT}"
echo "==> Baseline summary: ${BASELINE_SUMMARY}"
echo "==> Training outputs: ${OUTPUT_ROOT}/${DATASET_NAME}"
echo "==> Eval outputs: ${EVAL_ROOT}"
echo "==> Comparison outputs: ${COMPARE_ROOT}"
echo "==> Train GPUs: ${GPUS}"
echo "==> Eval GPUs: ${EVAL_GPUS}"

if [ ! -f "${BASELINE_SUMMARY}" ]; then
  echo "Missing baseline summary: ${BASELINE_SUMMARY}" >&2
  exit 1
fi

if [ "${OVERWRITE}" = "1" ]; then
  echo "==> Overwriting layer-norm outputs only"
  rm -rf "${OUTPUT_ROOT:?}/${DATASET_NAME}" "${EVAL_ROOT:?}" "${COMPARE_ROOT:?}"
fi

read -r -a GPU_LIST <<< "${GPUS}"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  echo "GPUS must contain at least one GPU id." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/${DATASET_NAME}/logs"

train_one() {
  local disease="$1"
  local gpu="$2"
  local output_dir="${OUTPUT_ROOT}/${DATASET_NAME}/${disease}"

  echo "==> Training ${VARIANT}: disease=${disease} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python scripts/train/train_conditional_vae.py \
    --config "${CONFIG}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-name "${DATASET_NAME}" \
    --target-group "${disease}" \
    --output-dir "${output_dir}" \
    --device "${DEVICE}" \
    ${STEPS:+--steps "${STEPS}"} \
    ${BATCH_SIZE:+--batch-size "${BATCH_SIZE}"} \
    ${LEARNING_RATE:+--learning-rate "${LEARNING_RATE}"} \
    ${LR_WARMUP_STEPS:+--lr-warmup-steps "${LR_WARMUP_STEPS}"} \
    ${GRADIENT_ACCUMULATION_STEPS:+--gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"} \
    ${BETA:+--beta "${BETA}"} \
    ${NUM_WORKERS:+--num-workers "${NUM_WORKERS}"} \
    ${LOG_EVERY:+--log-every "${LOG_EVERY}"} \
    ${VALIDATE_EVERY:+--validate-every "${VALIDATE_EVERY}"} \
    ${CHECKPOINT_EVERY:+--checkpoint-every "${CHECKPOINT_EVERY}"} \
    ${MAX_ROWS_PER_SPLIT:+--max-rows-per-split "${MAX_ROWS_PER_SPLIT}"}
}

pending_diseases=()
for disease in "${DISEASES[@]}"; do
  output_dir="${OUTPUT_ROOT}/${DATASET_NAME}/${disease}"
  if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${output_dir}/checkpoint_best.pt" ]; then
    echo "==> Skipping ${VARIANT}: disease=${disease} checkpoint exists"
  else
    pending_diseases+=("${disease}")
  fi
done

train_pids=()
for i in "${!pending_diseases[@]}"; do
  disease="${pending_diseases[$i]}"
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  train_one "${disease}" "${gpu}" \
    > "${OUTPUT_ROOT}/${DATASET_NAME}/logs/${disease}.log" 2>&1 &
  train_pids+=("$!")

  if [ "${#train_pids[@]}" -eq "${#GPU_LIST[@]}" ]; then
    wait "${train_pids[@]}"
    train_pids=()
  fi
done

if [ "${#train_pids[@]}" -gt 0 ]; then
  wait "${train_pids[@]}"
fi

read -r -a EVAL_GPU_LIST <<< "${EVAL_GPUS}"
if [ "${#EVAL_GPU_LIST[@]}" -eq 0 ]; then
  echo "EVAL_GPUS must contain at least one GPU id." >&2
  exit 1
fi

mkdir -p "${EVAL_ROOT}/logs"

eval_one() {
  local disease="$1"
  local gpu="$2"
  local checkpoint="${OUTPUT_ROOT}/${DATASET_NAME}/${disease}/checkpoint_best.pt"

  if [ ! -f "${checkpoint}" ]; then
    echo "Missing checkpoint for ${disease}: ${checkpoint}" >&2
    return 1
  fi

  echo "==> Evaluating ${VARIANT}: disease=${disease} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python scripts/eval/evaluate_cvae_prior_probability.py \
    --checkpoint "${checkpoint}" \
    --target-group "${disease}" \
    --variant "${VARIANT}" \
    --output-dir "${EVAL_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-name "${DATASET_NAME}" \
    --device "${EVAL_DEVICE}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --num-samples "${NUM_SAMPLES}" \
    --threshold-strategy validation_balanced_accuracy
}

if [ "${EVAL_PARALLEL}" = "1" ]; then
  eval_pids=()
  for i in "${!DISEASES[@]}"; do
    disease="${DISEASES[$i]}"
    gpu="${EVAL_GPU_LIST[$((i % ${#EVAL_GPU_LIST[@]}))]}"
    eval_one "${disease}" "${gpu}" > "${EVAL_ROOT}/logs/${disease}.log" 2>&1 &
    eval_pids+=("$!")

    if [ "${#eval_pids[@]}" -eq "${#EVAL_GPU_LIST[@]}" ]; then
      wait "${eval_pids[@]}"
      eval_pids=()
    fi
  done

  if [ "${#eval_pids[@]}" -gt 0 ]; then
    wait "${eval_pids[@]}"
  fi
else
  for i in "${!DISEASES[@]}"; do
    disease="${DISEASES[$i]}"
    gpu="${EVAL_GPU_LIST[$((i % ${#EVAL_GPU_LIST[@]}))]}"
    eval_one "${disease}" "${gpu}"
  done
fi

VARIANT="${VARIANT}" \
DATASET_NAME="${DATASET_NAME}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
EVAL_ROOT="${EVAL_ROOT}" \
COMPARE_ROOT="${COMPARE_ROOT}" \
BASELINE_SUMMARY="${BASELINE_SUMMARY}" \
uv run python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


disease_order = [
    "diabetes",
    "hypertension",
    "dyslipidemia",
    "liver_disease",
    "hepatitis_b",
    "hepatitis_c",
    "kidney_disease",
    "anemia",
]
variant = os.environ["VARIANT"]
variant_order = ["cvae", "cvae_masked_prior", variant]
dataset_name = os.environ["DATASET_NAME"]
output_root = Path(os.environ["OUTPUT_ROOT"])
eval_root = Path(os.environ["EVAL_ROOT"])
compare_root = Path(os.environ["COMPARE_ROOT"])
baseline_summary = Path(os.environ["BASELINE_SUMMARY"])
train_root = output_root / dataset_name
compare_root.mkdir(parents=True, exist_ok=True)


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
if len(rows) != len(disease_order):
    raise RuntimeError(f"Expected {len(disease_order)} metrics files, got {len(rows)}.")

model_summary = pd.DataFrame(rows)
model_summary.to_csv(eval_root / "summary.csv", index=False)

baseline = pd.read_csv(baseline_summary)
baseline = baseline[baseline["variant"].isin(["cvae", "cvae_masked_prior"])].copy()
combined = pd.concat([baseline, model_summary], ignore_index=True)
combined["variant"] = pd.Categorical(
    combined["variant"],
    categories=variant_order,
    ordered=True,
)
combined["disease"] = pd.Categorical(
    combined["disease"],
    categories=disease_order,
    ordered=True,
)
combined = combined.sort_values(["disease", "variant"]).reset_index(drop=True)
combined.to_csv(compare_root / "summary.csv", index=False)

wide = combined.pivot(index="disease", columns="variant", values="balanced_accuracy")
wide = wide.loc[disease_order]
wide["best_model"] = wide[variant_order].idxmax(axis=1)
wide[f"{variant}_minus_cvae"] = wide[variant] - wide["cvae"]
wide[f"{variant}_minus_cvae_masked_prior"] = wide[variant] - wide["cvae_masked_prior"]
wide.to_csv(compare_root / "balanced_accuracy_wide.csv")

means = combined.groupby("variant", observed=True)[
    ["sensitivity", "specificity", "balanced_accuracy"]
].mean()
means = means.loc[variant_order]
means.to_csv(compare_root / "mean_metrics_by_model.csv")

metric_plot = compare_root / "metric_comparison.png"
metric_specs = [
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("balanced_accuracy", "Balanced accuracy"),
]
fig, axes = plt.subplots(len(metric_specs), 1, figsize=(13.5, 12), sharex=True)
x = range(len(disease_order))
width = 0.8 / len(variant_order)
for ax, (metric, label) in zip(axes, metric_specs):
    for idx, candidate in enumerate(variant_order):
        sub = combined[combined["variant"] == candidate].set_index("disease")
        values = [float(sub.loc[disease, metric]) for disease in disease_order]
        offset = (idx - (len(variant_order) - 1) / 2) * width
        positions = [i + offset for i in x]
        ax.bar(positions, values, width=width, label=candidate)
    ax.set_title(label)
    ax.set_ylabel("score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(disease_order, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
axes[0].legend(ncol=len(variant_order), loc="upper right")
fig.suptitle("CVAE vs Masked-Prior CVAE vs LayerNorm Masked-Prior CVAE", fontsize=14)
fig.tight_layout()
fig.savefig(metric_plot, dpi=180)
plt.close(fig)

fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=False)
axes = axes.ravel()
for ax, disease in zip(axes, disease_order):
    metrics_path = train_root / disease / "metrics.csv"
    if not metrics_path.exists():
        ax.axis("off")
        continue
    df = pd.read_csv(metrics_path)
    for split, style in [("train", "-"), ("valid", "--")]:
        split_df = df[df["split"] == split].sort_values("step")
        if split_df.empty:
            continue
        ax.plot(split_df["step"], split_df["loss"], style, linewidth=1.1, label=split)
    ax.set_title(disease)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
for ax in axes[len(disease_order):]:
    ax.axis("off")
fig.suptitle(f"{variant} Loss Curves", fontsize=14)
fig.tight_layout()
fig.savefig(compare_root / "loss_curves_layer_norm.png", dpi=180)
plt.close(fig)

print("==> Wrote eval summary:", eval_root / "summary.csv")
print("==> Wrote comparison summary:", compare_root / "summary.csv")
print("==> Wrote metric comparison plot:", metric_plot)
print("==> Wrote loss curve plot:", compare_root / "loss_curves_layer_norm.png")
print("==> Mean metrics")
print(means.to_string(float_format=lambda value: f"{value:.4f}"))
PY

echo "==> Done."
echo "==> Comparison directory: ${COMPARE_ROOT}"
