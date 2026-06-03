#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/train/train_conditional_vae_eddi_masked_prior.yaml}"
VARIANT="${VARIANT:-cvae_masked_prior}"
ORIGINAL_VARIANT="${ORIGINAL_VARIANT:-cvae}"
INCLUDE_CATBOOST="${INCLUDE_CATBOOST:-0}"
EVAL_ORIGINAL="${EVAL_ORIGINAL:-1}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/conditional_vae_${VARIANT}/${RUN_NAME}}"
EVAL_ROOT="${EVAL_ROOT:-outputs/eval_${VARIANT}/${RUN_NAME}/${DATASET_NAME}}"
ORIGINAL_OUTPUT_ROOT="${ORIGINAL_OUTPUT_ROOT:-outputs/conditional_vae_original}"
ORIGINAL_EVAL_ROOT="${ORIGINAL_EVAL_ROOT:-outputs/eval_${ORIGINAL_VARIANT}_for_${VARIANT}/${RUN_NAME}/${DATASET_NAME}}"
COMPARE_ROOT="${COMPARE_ROOT:-outputs/eval_comparison_${VARIANT}/${RUN_NAME}/${DATASET_NAME}}"
ORIGINAL_SUMMARY="${ORIGINAL_SUMMARY:-outputs/eval_baseline_20260519/${DATASET_NAME}/summary.csv}"
DEVICE="${DEVICE:-cuda}"
GPUS="${GPUS:-0 1 2 3}"
EVAL_GPU="${EVAL_GPU:-0}"
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

echo "==> EDDI CVAE masked-prior run: ${RUN_NAME}"
echo "==> Variant: ${VARIANT}"
echo "==> Config: ${CONFIG}"
echo "==> Training outputs: ${OUTPUT_ROOT}/${DATASET_NAME}"
echo "==> Eval outputs: ${EVAL_ROOT}"
echo "==> Eval GPUs: ${EVAL_GPUS}"
echo "==> Original CVAE outputs: ${ORIGINAL_OUTPUT_ROOT}/${DATASET_NAME}"
echo "==> Original CVAE eval outputs: ${ORIGINAL_EVAL_ROOT}"
echo "==> Original summary: ${ORIGINAL_SUMMARY}"
echo "==> Comparison outputs: ${COMPARE_ROOT}"

if [ "${OVERWRITE}" = "1" ]; then
  echo "==> Overwriting existing run outputs"
  rm -rf "${OUTPUT_ROOT:?}/${DATASET_NAME}" "${EVAL_ROOT:?}" "${ORIGINAL_EVAL_ROOT:?}" "${COMPARE_ROOT:?}"
fi

mkdir -p "${OUTPUT_ROOT}/${DATASET_NAME}/logs"

train_one() {
  local disease="$1"
  local gpu="$2"
  local output_dir="${OUTPUT_ROOT}/${DATASET_NAME}/${disease}"

  echo "==> Training ${VARIANT}: dataset=${DATASET_NAME} disease=${disease} gpu=${gpu}"
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

PENDING_DISEASES=()
for disease in "${DISEASES[@]}"; do
  output_dir="${OUTPUT_ROOT}/${DATASET_NAME}/${disease}"
  if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${output_dir}/checkpoint_best.pt" ]; then
    echo "==> Skipping ${VARIANT}: dataset=${DATASET_NAME} disease=${disease} checkpoint exists"
  else
    PENDING_DISEASES+=("${disease}")
  fi
done

read -r -a GPU_LIST <<< "${GPUS}"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  echo "GPUS must contain at least one GPU id." >&2
  exit 1
fi

pids=()
for i in "${!PENDING_DISEASES[@]}"; do
  disease="${PENDING_DISEASES[$i]}"
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  train_one "${disease}" "${gpu}" > "${OUTPUT_ROOT}/${DATASET_NAME}/logs/${disease}.log" 2>&1 &
  pids+=("$!")

  if [ "${#pids[@]}" -eq "${#GPU_LIST[@]}" ]; then
    wait "${pids[@]}"
    pids=()
  fi
done

if [ "${#pids[@]}" -gt 0 ]; then
  wait "${pids[@]}"
fi

mkdir -p "${EVAL_ROOT}/logs"

eval_one() {
  local disease="$1"
  local gpu="$2"
  local checkpoint="${OUTPUT_ROOT}/${DATASET_NAME}/${disease}/checkpoint_best.pt"

  echo "==> Evaluating ${VARIANT}: disease=${disease} gpu=${gpu} checkpoint=${checkpoint}"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python scripts/eval/evaluate_cvae_prior_probability.py \
    --checkpoint "${checkpoint}" \
    --target-group "${disease}" \
    --variant "${VARIANT}" \
    --output-dir "${EVAL_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-name "${DATASET_NAME}" \
    --device cuda \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --num-samples "${NUM_SAMPLES}" \
    --threshold-strategy validation_balanced_accuracy
}

if [ "${EVAL_PARALLEL}" = "1" ]; then
  read -r -a EVAL_GPU_LIST <<< "${EVAL_GPUS}"
  if [ "${#EVAL_GPU_LIST[@]}" -eq 0 ]; then
    echo "EVAL_GPUS must contain at least one GPU id." >&2
    exit 1
  fi

  eval_pids=()
  for i in "${!DISEASES[@]}"; do
    disease="${DISEASES[$i]}"
    gpu="${EVAL_GPU_LIST[$((i % ${#EVAL_GPU_LIST[@]}))]}"
    echo "==> Queue eval ${VARIANT}: disease=${disease} gpu=${gpu}"
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
  for disease in "${DISEASES[@]}"; do
    eval_one "${disease}" "${EVAL_GPU}"
  done
fi

eval_original_one() {
  local disease="$1"
  local gpu="$2"
  local checkpoint="${ORIGINAL_OUTPUT_ROOT}/${DATASET_NAME}/${disease}/checkpoint_best.pt"

  echo "==> Evaluating ${ORIGINAL_VARIANT}: disease=${disease} gpu=${gpu} checkpoint=${checkpoint}"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python scripts/eval/evaluate_cvae_prior_probability.py \
    --checkpoint "${checkpoint}" \
    --target-group "${disease}" \
    --variant "${ORIGINAL_VARIANT}" \
    --output-dir "${ORIGINAL_EVAL_ROOT}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-name "${DATASET_NAME}" \
    --device cuda \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --num-samples "${NUM_SAMPLES}" \
    --threshold-strategy validation_balanced_accuracy
}

if [ "${EVAL_ORIGINAL}" = "1" ]; then
  mkdir -p "${ORIGINAL_EVAL_ROOT}/logs"
  if [ "${EVAL_PARALLEL}" = "1" ]; then
    read -r -a EVAL_GPU_LIST <<< "${EVAL_GPUS}"
    if [ "${#EVAL_GPU_LIST[@]}" -eq 0 ]; then
      echo "EVAL_GPUS must contain at least one GPU id." >&2
      exit 1
    fi

    original_eval_pids=()
    for i in "${!DISEASES[@]}"; do
      disease="${DISEASES[$i]}"
      gpu="${EVAL_GPU_LIST[$((i % ${#EVAL_GPU_LIST[@]}))]}"
      echo "==> Queue eval ${ORIGINAL_VARIANT}: disease=${disease} gpu=${gpu}"
      eval_original_one "${disease}" "${gpu}" > "${ORIGINAL_EVAL_ROOT}/logs/${disease}.log" 2>&1 &
      original_eval_pids+=("$!")

      if [ "${#original_eval_pids[@]}" -eq "${#EVAL_GPU_LIST[@]}" ]; then
        wait "${original_eval_pids[@]}"
        original_eval_pids=()
      fi
    done

    if [ "${#original_eval_pids[@]}" -gt 0 ]; then
      wait "${original_eval_pids[@]}"
    fi
  else
    for disease in "${DISEASES[@]}"; do
      eval_original_one "${disease}" "${EVAL_GPU}"
    done
  fi
fi

VARIANT="${VARIANT}" \
ORIGINAL_VARIANT="${ORIGINAL_VARIANT}" \
INCLUDE_CATBOOST="${INCLUDE_CATBOOST}" \
EVAL_ORIGINAL="${EVAL_ORIGINAL}" \
DATASET_NAME="${DATASET_NAME}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
EVAL_ROOT="${EVAL_ROOT}" \
ORIGINAL_EVAL_ROOT="${ORIGINAL_EVAL_ROOT}" \
COMPARE_ROOT="${COMPARE_ROOT}" \
ORIGINAL_SUMMARY="${ORIGINAL_SUMMARY}" \
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
original_variant = os.environ["ORIGINAL_VARIANT"]
include_catboost = os.environ["INCLUDE_CATBOOST"] == "1"
eval_original = os.environ["EVAL_ORIGINAL"] == "1"
variant_order = [original_variant, variant]
if include_catboost:
    variant_order = ["catboost", *variant_order]

dataset_name = os.environ["DATASET_NAME"]
output_root = Path(os.environ["OUTPUT_ROOT"])
eval_root = Path(os.environ["EVAL_ROOT"])
original_eval_root = Path(os.environ["ORIGINAL_EVAL_ROOT"])
compare_root = Path(os.environ["COMPARE_ROOT"])
original_summary = Path(os.environ["ORIGINAL_SUMMARY"])
train_root = output_root / dataset_name
compare_root.mkdir(parents=True, exist_ok=True)

def metrics_rows(root: Path, candidate_variant: str) -> list[dict[str, object]]:
    rows = []
    for path in sorted(root.glob(f"{candidate_variant}_*_metrics.json")):
        data = json.loads(path.read_text())
        sensitivity = float(data["sensitivity"])
        specificity = float(data["specificity"])
        rows.append(
            {
                "variant": candidate_variant,
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
    return rows


model_summary = pd.DataFrame(metrics_rows(eval_root, variant))
model_summary.to_csv(eval_root / "summary.csv", index=False)

if eval_original:
    baseline = pd.DataFrame(metrics_rows(original_eval_root, original_variant))
    baseline.to_csv(original_eval_root / "summary.csv", index=False)
    if include_catboost:
        catboost = pd.read_csv(original_summary)
        catboost = catboost[catboost["variant"] == "catboost"].copy()
        baseline = pd.concat([catboost, baseline], ignore_index=True)
else:
    baseline = pd.read_csv(original_summary)
    baseline_variants = [original_variant]
    if include_catboost:
        baseline_variants.insert(0, "catboost")
    baseline = baseline[baseline["variant"].isin(baseline_variants)].copy()
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
wide[f"{variant}_minus_{original_variant}"] = wide[variant] - wide[original_variant]
if include_catboost:
    wide[f"{variant}_minus_catboost"] = wide[variant] - wide["catboost"]
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
fig.suptitle("Original CVAE vs Masked-Prior CVAE", fontsize=14)
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
fig.savefig(compare_root / "loss_curves.png", dpi=180)
plt.close(fig)

print("==> Wrote eval summary:", eval_root / "summary.csv")
if eval_original:
    print("==> Wrote original eval summary:", original_eval_root / "summary.csv")
print("==> Wrote comparison summary:", compare_root / "summary.csv")
print("==> Wrote metric comparison plot:", metric_plot)
print("==> Wrote loss curve plot:", compare_root / "loss_curves.png")
print("==> Mean metrics")
print(means.to_string(float_format=lambda value: f"{value:.4f}"))
PY

echo "==> Done."
echo "==> Comparison directory: ${COMPARE_ROOT}"
