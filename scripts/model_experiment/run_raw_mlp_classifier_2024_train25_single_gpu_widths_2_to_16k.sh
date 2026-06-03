#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/train/train_raw_mlp_disease_classifier_diabetes_2024_train25_valid25_test50.yaml}"
ROOT="${ROOT:-model_experiment/classification_2024_train25_valid25_test50_single_lr3e-4_10k_bs512}"
PYTHON_BIN="${PYTHON_BIN:-uv run python}"
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"

WIDTHS=(${WIDTHS:-2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384})
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
DROPOUT="${DROPOUT:-0}"
NORMALIZATION="${NORMALIZATION:-none}"
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-0}"
CLASS_WEIGHT="${CLASS_WEIGHT:-balanced}"
VALIDATE_EVERY="${VALIDATE_EVERY:-100}"
LOG_EVERY="${LOG_EVERY:-50}"

mkdir -p "${ROOT}/logs" "${ROOT}/models/mlp" "${ROOT}/figures/loss/mlp" "${ROOT}/figures/loss"

SUMMARY_CSV="${ROOT}/summary.csv"
FIGURES_DIR="${ROOT}/figures"
LOSS_DIR="${ROOT}/figures/loss/mlp"

launch_one() {
  local width="$1"
  local gpu="$2"
  local out_dir="${ROOT}/models/mlp/width_${width}/rep_01"
  local log_path="${ROOT}/logs/mlp_width${width}_gpu${gpu}.log"
  local seed="$((4000 + width))"
  echo "==> launch width=${width} on GPU ${gpu}, batch=${BATCH_SIZE}, lr=${LR}, steps=${STEPS}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PYTHON_CMD[@]}" \
      scripts/model_experiment/train_raw_mlp_disease_classifier.py \
        --config "${CONFIG}" \
        --hidden-layers "${width}" "${width}" \
        --normalization "${NORMALIZATION}" \
        --dropout "${DROPOUT}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --learning-rate "${LR}" \
        --steps "${STEPS}" \
        --batch-size "${BATCH_SIZE}" \
        --gradient-clip-norm "${GRADIENT_CLIP_NORM}" \
        --class-weight "${CLASS_WEIGHT}" \
        --log-every "${LOG_EVERY}" \
        --validate-every "${VALIDATE_EVERY}" \
        --checkpoint-every "${STEPS}" \
        --seed "${seed}" \
        --device cuda \
        --output-dir "${out_dir}" \
        --summary-csv "${SUMMARY_CSV}" \
        --figures-dir "${FIGURES_DIR}" \
        --loss-dir "${LOSS_DIR}" 2>&1 | tee "${log_path}"
  ) &
}

index=0
while [[ "${index}" -lt "${#WIDTHS[@]}" ]]; do
  PIDS=()
  for gpu in "${GPUS[@]}"; do
    if [[ "${index}" -ge "${#WIDTHS[@]}" ]]; then
      break
    fi
    launch_one "${WIDTHS[$index]}" "${gpu}"
    PIDS+=("$!")
    index=$((index + 1))
  done
  for pid in "${PIDS[@]}"; do
    wait "${pid}"
  done
done

"${PYTHON_CMD[@]}" scripts/model_experiment/plot_raw_mlp_loss_by_width.py \
  --root "${ROOT}" \
  --output-dir "${ROOT}/figures/loss" \
  --prefix train_valid_loss_by_width

"${PYTHON_CMD[@]}" - "${ROOT}" <<'PY'
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

root = Path(sys.argv[1])
out_dir = root / "figures"
rows = []
for metrics_path in sorted(root.glob("models/mlp/width_*/rep_01/metrics.csv"), key=lambda p: int(p.parts[-3].split("_")[-1])):
    width = int(metrics_path.parts[-3].split("_")[-1])
    metadata_path = metrics_path.parent / "model_metadata.json"
    if not metadata_path.exists():
        continue
    metadata = json.loads(metadata_path.read_text())
    df = pd.read_csv(metrics_path)
    train = df[df["split"].eq("train")]
    valid = df[df["split"].eq("valid")]
    if valid.empty:
        continue
    rows.append(
        {
            "width": width,
            "parameter_count": int(metadata["parameter_count"]),
            "last_train_loss": float(train["loss"].iloc[-1]) if not train.empty else float("nan"),
            "mean_last20_train_loss": float(train.tail(20)["loss"].mean()) if not train.empty else float("nan"),
            "last_valid_loss": float(valid["loss"].iloc[-1]),
            "mean_last20_valid_loss": float(valid.tail(20)["loss"].mean()),
            "last_valid_auroc": float(valid["auroc"].iloc[-1]),
            "mean_last20_valid_auroc": float(valid.tail(20)["auroc"].mean()),
            "best_valid_loss": float(valid["loss"].min()),
            "best_valid_auroc": float(valid["auroc"].max()),
        }
    )

result = pd.DataFrame(rows).sort_values("parameter_count")
csv_path = out_dir / "last_and_last20_loss_vs_parameters.csv"
result.to_csv(csv_path, index=False)

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
ax.plot(result["parameter_count"], result["last_valid_loss"], marker="o", linewidth=1.5, label="last valid loss")
ax.plot(result["parameter_count"], result["mean_last20_valid_loss"], marker="s", linewidth=1.5, label="last 20 valid loss mean")
for _, row in result.iterrows():
    ax.annotate(str(int(row["width"])), (row["parameter_count"], row["mean_last20_valid_loss"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("parameter count (log scale)")
ax.set_ylabel("valid BCE loss")
ax.set_title("Valid loss vs model size")
ax.grid(True, which="both", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "last_and_last20_valid_loss_vs_parameters.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
ax.plot(result["parameter_count"], result["mean_last20_train_loss"], marker="o", linewidth=1.5, label="last 20 train loss mean")
ax.plot(result["parameter_count"], result["mean_last20_valid_loss"], marker="s", linewidth=1.5, label="last 20 valid loss mean")
for _, row in result.iterrows():
    ax.annotate(str(int(row["width"])), (row["parameter_count"], row["mean_last20_valid_loss"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=1e-5)
ax.set_xlabel("parameter count (log scale)")
ax.set_ylabel("BCE loss (symlog)")
ax.set_title("Last 20-step averaged train/valid loss vs model size")
ax.grid(True, which="both", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "last20_train_valid_loss_vs_parameters.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
ax.plot(result["parameter_count"], result["mean_last20_valid_auroc"], marker="o", linewidth=1.5, label="last 20 valid AUROC mean")
ax.plot(result["parameter_count"], result["best_valid_auroc"], marker="s", linewidth=1.5, label="best valid AUROC")
for _, row in result.iterrows():
    ax.annotate(str(int(row["width"])), (row["parameter_count"], row["mean_last20_valid_auroc"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("parameter count (log scale)")
ax.set_ylabel("valid AUROC")
ax.set_title("Valid AUROC vs model size")
ax.grid(True, which="both", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "last20_and_best_valid_auroc_vs_parameters.png")
plt.close(fig)

print(csv_path)
PY

echo "done: ${ROOT}"
echo "summary: ${SUMMARY_CSV}"
echo "loss figures: ${ROOT}/figures/loss"
echo "model-size figures: ${ROOT}/figures"
