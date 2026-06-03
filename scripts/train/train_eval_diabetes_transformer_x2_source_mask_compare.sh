#!/usr/bin/env bash
set -euo pipefail

DISEASE="${DISEASE:-diabetes}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
RUN_NAME="${RUN_NAME:-transformer_x2_source_mask_${DISEASE}_$(date +%Y%m%d_%H%M%S)}"

CONFIG="${CONFIG:-configs/train/train_conditional_vae_transformer_oldstruct_x2_source_mask.yaml}"
VARIANT="${VARIANT:-cvae-transformerx2-source-mask}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/conditional_vae_${VARIANT}/${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET_NAME}/${DISEASE}"
EVAL_ROOT="${EVAL_ROOT:-outputs/eval_${VARIANT}/${RUN_NAME}/${DATASET_NAME}}"
COMPARE_ROOT="${COMPARE_ROOT:-outputs/eval_comparison_${VARIANT}/${RUN_NAME}/${DATASET_NAME}}"

BASELINE_SUMMARY="${BASELINE_SUMMARY:-outputs/eval_baseline_20260519/${DATASET_NAME}/summary.csv}"
OLD_TRANSFORMER_SUMMARY="${OLD_TRANSFORMER_SUMMARY:-outputs/eval_comparison_transformer/transformer_decoder_masked_prior_500/${DATASET_NAME}/summary.csv}"

GPUS="${GPUS:-2,3}"
NPROC="${NPROC:-2}"
EVAL_GPU="${EVAL_GPU:-1}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
SOURCE_MASK_DROPOUT="${SOURCE_MASK_DROPOUT:-0.1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
OVERWRITE="${OVERWRITE:-0}"

effective_batch=$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC))

echo "==> Diabetes transformer x2 source-mask comparison"
echo "==> Run: ${RUN_NAME}"
echo "==> Dataset: ${DATASET_NAME}; disease: ${DISEASE}"
echo "==> GPUs: ${GPUS}; nproc=${NPROC}; eval_gpu=${EVAL_GPU}"
echo "==> Per-GPU batch: ${BATCH_SIZE}; accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "==> Effective total batch: ${effective_batch}"
echo "==> Config: ${CONFIG}"
echo "==> Source mask dropout: ${SOURCE_MASK_DROPOUT}"
echo "==> Beta override: ${BETA:-<config default>}"
echo "==> Learning rate override: ${LEARNING_RATE:-<config default>}"
echo "==> Baseline summary: ${BASELINE_SUMMARY}"
echo "==> Old transformer summary: ${OLD_TRANSFORMER_SUMMARY}"

for required in "${CONFIG}" "${BASELINE_SUMMARY}" "${OLD_TRANSFORMER_SUMMARY}"; do
  if [ ! -f "${required}" ]; then
    echo "Missing required file: ${required}" >&2
    exit 1
  fi
done

if [ "${OVERWRITE}" = "1" ]; then
  rm -rf "${OUTPUT_DIR:?}" "${EVAL_ROOT:?}" "${COMPARE_ROOT:?}"
fi

mkdir -p "${OUTPUT_ROOT}/${DATASET_NAME}/logs" "${EVAL_ROOT}/logs" "${COMPARE_ROOT}"

train_log="${OUTPUT_ROOT}/${DATASET_NAME}/logs/${DISEASE}_${VARIANT}.log"
echo "==> Training ${VARIANT}: output=${OUTPUT_DIR}"
if [ "${NPROC}" -gt 1 ]; then
  CUDA_VISIBLE_DEVICES="${GPUS}" uv run torchrun --standalone --nproc_per_node "${NPROC}" \
    scripts/train/train_conditional_vae.py \
    --config "${CONFIG}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-name "${DATASET_NAME}" \
    --target-group "${DISEASE}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --source-mask-dropout "${SOURCE_MASK_DROPOUT}" \
    ${BETA:+--beta "${BETA}"} \
    ${LEARNING_RATE:+--learning-rate "${LEARNING_RATE}"} \
    2>&1 | tee "${train_log}"
else
  CUDA_VISIBLE_DEVICES="${GPUS}" uv run python scripts/train/train_conditional_vae.py \
    --config "${CONFIG}" \
    --dataset-root "${DATASET_ROOT}" \
    --dataset-name "${DATASET_NAME}" \
    --target-group "${DISEASE}" \
    --output-dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --source-mask-dropout "${SOURCE_MASK_DROPOUT}" \
    ${BETA:+--beta "${BETA}"} \
    ${LEARNING_RATE:+--learning-rate "${LEARNING_RATE}"} \
    2>&1 | tee "${train_log}"
fi

checkpoint="${OUTPUT_DIR}/checkpoint_best.pt"
if [ ! -f "${checkpoint}" ]; then
  echo "Missing checkpoint: ${checkpoint}" >&2
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
  2>&1 | tee "${EVAL_ROOT}/logs/${DISEASE}.log"

DISEASE="${DISEASE}" \
VARIANT="${VARIANT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
EVAL_ROOT="${EVAL_ROOT}" \
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
variant = os.environ["VARIANT"]
output_dir = Path(os.environ["OUTPUT_DIR"])
eval_root = Path(os.environ["EVAL_ROOT"])
compare_root = Path(os.environ["COMPARE_ROOT"])
baseline_summary = Path(os.environ["BASELINE_SUMMARY"])
old_transformer_summary = Path(os.environ["OLD_TRANSFORMER_SUMMARY"])
compare_root.mkdir(parents=True, exist_ok=True)

def balanced_accuracy(row: pd.Series) -> float:
    return 0.5 * (float(row["sensitivity"]) + float(row["specificity"]))

def row_from_metrics(eval_root: Path, variant: str) -> pd.DataFrame:
    data = json.loads((eval_root / f"{variant}_{disease}_metrics.json").read_text())
    sensitivity = float(data["sensitivity"])
    specificity = float(data["specificity"])
    return pd.DataFrame([
        {
            "variant": variant,
            "disease": data["target_group"],
            "threshold": data["threshold"],
            "tp": data.get("tp"),
            "tn": data.get("tn"),
            "fp": data.get("fp"),
            "fn": data.get("fn"),
            "missing_label": data.get("missing_label"),
            "missing_prediction": data.get("missing_prediction"),
            "n_evaluable": data.get("n_evaluable"),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": 0.5 * (sensitivity + specificity),
        }
    ])

new_row = row_from_metrics(eval_root, variant)
new_row.to_csv(eval_root / "summary.csv", index=False)

baseline = pd.read_csv(baseline_summary)
catboost = baseline[(baseline["disease"] == disease) & (baseline["variant"] == "catboost")].copy()
cvae = baseline[(baseline["disease"] == disease) & (baseline["variant"] == "cvae")].copy()
cvae["variant"] = "cvae-mlp"

old = pd.read_csv(old_transformer_summary)
old_transformer = old[(old["disease"] == disease) & (old["variant"] == "transformer")].copy()
old_transformer["variant"] = "cvae-transformer-old"

combined = pd.concat([catboost, cvae, old_transformer, new_row], ignore_index=True)
for column in ("sensitivity", "specificity"):
    combined[column] = combined[column].astype(float)
combined["balanced_accuracy"] = combined.apply(balanced_accuracy, axis=1)

order = ["catboost", "cvae-mlp", "cvae-transformer-old", variant]
missing = [name for name in order if name not in set(combined["variant"])]
if missing:
    raise RuntimeError(f"Missing comparison rows for variants: {missing}")
combined["variant"] = pd.Categorical(combined["variant"], categories=order, ordered=True)
combined = combined.sort_values("variant").reset_index(drop=True)
combined.to_csv(compare_root / "summary.csv", index=False)

wide = combined.set_index("variant")[["sensitivity", "specificity", "balanced_accuracy"]].T
wide[f"{variant}_minus_old"] = wide[variant] - wide["cvae-transformer-old"]
wide.to_csv(compare_root / f"{disease}_metrics_wide.csv")

plot_path = compare_root / f"{disease}_metric_comparison.png"
metric_specs = [
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("balanced_accuracy", "Balanced accuracy"),
]
fig, ax = plt.subplots(figsize=(10.5, 5.2))
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

metrics_path = output_dir / "metrics.csv"
if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for split, style in [("train", "-"), ("valid", "--")]:
        split_df = metrics[metrics["split"] == split]
        if split_df.empty:
            continue
        ax.plot(split_df["step"], split_df["loss"], style, linewidth=1.2, label=f"{split} loss")
        if "loss_reconstruction" in split_df:
            ax.plot(
                split_df["step"],
                split_df["loss_reconstruction"],
                style,
                linewidth=0.9,
                alpha=0.65,
                label=f"{split} rc",
            )
    ax.set_title(f"{variant} Loss Curves")
    ax.set_xlabel("step")
    ax.set_ylabel("loss per observed feature")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(compare_root / f"{disease}_loss_curves.png", dpi=180)
    plt.close(fig)

print(combined[["variant", "sensitivity", "specificity", "balanced_accuracy"]].to_string(index=False))
print("summary:", compare_root / "summary.csv")
print("wide:", compare_root / f"{disease}_metrics_wide.csv")
print("compare png:", plot_path)
if metrics_path.exists():
    print("loss png:", compare_root / f"{disease}_loss_curves.png")
PY

echo "==> Done."
echo "==> Compare directory: ${COMPARE_ROOT}"
