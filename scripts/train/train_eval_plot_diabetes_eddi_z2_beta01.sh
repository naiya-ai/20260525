#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-eddi_z2_beta01_diabetes_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/conditional_vae_eddi_z2_beta01/${RUN_NAME}}"
EVAL_ROOT="${EVAL_ROOT:-outputs/eval_eddi_z2_beta01/${RUN_NAME}/harmonized_knhanes_1998_2024}"
PLOT_ROOT="${PLOT_ROOT:-outputs/plots/${RUN_NAME}}"
CONFIG="${CONFIG:-configs/train/train_conditional_vae_eddi_z2_beta01_diabetes.yaml}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
DISEASE="${DISEASE:-diabetes}"
GPU="${GPU:-1}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4096}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
LATENT_PLOT_SAMPLES="${LATENT_PLOT_SAMPLES:-6}"

nvidia-smi

RUN_NAME="${RUN_NAME}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
CONFIG="${CONFIG}" \
DATASET_NAME="${DATASET_NAME}" \
DATASET_ROOT="${DATASET_ROOT}" \
DISEASE="${DISEASE}" \
GPU="${GPU}" \
BATCH_SIZE="${BATCH_SIZE}" \
./scripts/train/train_diabetes_eddi_z2_beta01.sh

CHECKPOINT="${OUTPUT_ROOT}/${DATASET_NAME}/${DISEASE}/checkpoint_best.pt"
if [ ! -f "${CHECKPOINT}" ]; then
  echo "Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${EVAL_ROOT}" "${PLOT_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU}" uv run python scripts/eval/evaluate_cvae_prior_probability.py \
  --checkpoint "${CHECKPOINT}" \
  --target-group "${DISEASE}" \
  --variant eddi_z2_beta01_best \
  --output-dir "${EVAL_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --device cuda \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers 0 \
  --num-samples "${NUM_SAMPLES}" \
  --save-probabilities

PROBABILITIES="${EVAL_ROOT}/eddi_z2_beta01_best_${DISEASE}_probabilities.csv"
AUROC_JSON="${EVAL_ROOT}/eddi_z2_beta01_best_${DISEASE}_auroc.json"
uv run python - "${PROBABILITIES}" "${AUROC_JSON}" <<'PY'
import csv
import json
import sys
from pathlib import Path

import numpy as np


def parse_bool(value: str) -> int | None:
    value = value.strip().lower()
    if value in {"true", "1", "yes"}:
        return 1
    if value in {"false", "0", "no"}:
        return 0
    if value in {"", "nan"}:
        return None
    return int(float(value))


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


probability_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
labels = []
scores = []
with probability_path.open(newline="") as file:
    reader = csv.DictReader(file)
    for row in reader:
        label = parse_bool(row.get("label", ""))
        score = row.get("disease_probability", "")
        if label is None or score == "":
            continue
        labels.append(label)
        scores.append(float(score))

labels_array = np.asarray(labels, dtype=int)
scores_array = np.asarray(scores, dtype=float)
result = {
    "probabilities": str(probability_path),
    "n_evaluable": int(len(labels_array)),
    "n_positive": int(labels_array.sum()),
    "auroc": auroc(labels_array, scores_array),
}
output_path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY

PYTHONPATH=src uv run python scripts/analysis/plot_latent_prior_posterior_heatmaps.py \
  --checkpoint "${CHECKPOINT}" \
  --target-group "${DISEASE}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --split test \
  --num-samples "${LATENT_PLOT_SAMPLES}" \
  --output "${PLOT_ROOT}/${DISEASE}_eddi_z2_beta01_latent_prior_posterior_heatmaps.png" \
  --title "Diabetes EDDI z2 beta=0.01 latent prior/posterior on test samples"

echo "==> AUROC: ${AUROC_JSON}"
echo "==> Heatmap: ${PLOT_ROOT}/${DISEASE}_eddi_z2_beta01_latent_prior_posterior_heatmaps.png"
