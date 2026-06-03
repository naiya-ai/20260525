#!/usr/bin/env bash
set -euo pipefail

DISEASE="${DISEASE:-diabetes}"
DATASET_NAME="${DATASET_NAME:-harmonized_knhanes_1998_2024}"
DATASET_ROOT="${DATASET_ROOT:-datasets/preprocessed/gaussian_quantile}"
RUN_NAME="${RUN_NAME:-diabetes_auroc_compare_$(date +%Y%m%d_%H%M%S)}"

CVAE_CONFIG="${CVAE_CONFIG:-configs/train/train_conditional_vae_eddi_z2_beta003_diabetes.yaml}"
TRANSFORMER_CONFIG="${TRANSFORMER_CONFIG:-configs/train/train_conditional_vae_transformer_small_diabetes.yaml}"

ROOT="${ROOT:-outputs/diabetes_auroc_compare/${RUN_NAME}/${DATASET_NAME}}"
CATBOOST_DIR="${CATBOOST_DIR:-${ROOT}/catboost}"
CVAE_DIR="${CVAE_DIR:-${ROOT}/cvae}"
TRANSFORMER_DIR="${TRANSFORMER_DIR:-${ROOT}/cvae_transformer}"
EVAL_DIR="${EVAL_DIR:-${ROOT}/eval}"
COMPARE_DIR="${COMPARE_DIR:-${ROOT}/compare}"

DEVICE="${DEVICE:-cuda}"
GPU="${GPU:-0}"
STEPS="${STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-}"
TRANSFORMER_BATCH_SIZE="${TRANSFORMER_BATCH_SIZE:-${BATCH_SIZE}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OVERWRITE="${OVERWRITE:-0}"

if [ "${DISEASE}" != "diabetes" ]; then
  echo "This comparison script is intentionally diabetes-only. Got DISEASE=${DISEASE}" >&2
  exit 1
fi

if [ "${DEVICE}" = "cuda" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi

echo "==> Diabetes AUROC comparison"
echo "==> Dataset: ${DATASET_NAME}"
echo "==> Device: ${DEVICE}"
if [ "${DEVICE}" = "cuda" ]; then
  echo "==> CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
fi
echo "==> CVAE config: ${CVAE_CONFIG}"
echo "==> Transformer config: ${TRANSFORMER_CONFIG}"
echo "==> Output root: ${ROOT}"

if [ "${OVERWRITE}" = "1" ]; then
  rm -rf "${ROOT:?}"
fi
mkdir -p "${CATBOOST_DIR}" "${CVAE_DIR}" "${TRANSFORMER_DIR}" "${EVAL_DIR}" "${COMPARE_DIR}"

uv run python scripts/baselines/train_eval_catboost_disease.py \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --target-group "${DISEASE}" \
  --variant catboost \
  --output-dir "${CATBOOST_DIR}" \
  --task-type CPU \
  --save-predictions

uv run python scripts/train/train_conditional_vae.py \
  --config "${CVAE_CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --target-group "${DISEASE}" \
  --output-dir "${CVAE_DIR}/${DISEASE}" \
  --device "${DEVICE}" \
  --steps "${STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  ${BATCH_SIZE:+--batch-size "${BATCH_SIZE}"}

uv run python scripts/eval/evaluate_cvae_prior_probability.py \
  --checkpoint "${CVAE_DIR}/${DISEASE}/checkpoint_best.pt" \
  --target-group "${DISEASE}" \
  --variant cvae \
  --output-dir "${EVAL_DIR}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --device "${DEVICE}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-samples "${NUM_SAMPLES}" \
  --threshold-strategy validation_balanced_accuracy \
  --save-probabilities

uv run python scripts/train/train_conditional_vae.py \
  --config "${TRANSFORMER_CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --target-group "${DISEASE}" \
  --output-dir "${TRANSFORMER_DIR}/${DISEASE}" \
  --device "${DEVICE}" \
  --steps "${STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  ${TRANSFORMER_BATCH_SIZE:+--batch-size "${TRANSFORMER_BATCH_SIZE}"}

uv run python scripts/eval/evaluate_cvae_prior_probability.py \
  --checkpoint "${TRANSFORMER_DIR}/${DISEASE}/checkpoint_best.pt" \
  --target-group "${DISEASE}" \
  --variant cvae_transformer \
  --output-dir "${EVAL_DIR}" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-name "${DATASET_NAME}" \
  --device "${DEVICE}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-samples "${NUM_SAMPLES}" \
  --threshold-strategy validation_balanced_accuracy \
  --save-probabilities

CATBOOST_METRICS="${CATBOOST_DIR}/catboost_${DISEASE}_metrics.json" \
CVAE_METRICS="${EVAL_DIR}/cvae_${DISEASE}_metrics.json" \
TRANSFORMER_METRICS="${EVAL_DIR}/cvae_transformer_${DISEASE}_metrics.json" \
COMPARE_DIR="${COMPARE_DIR}" \
uv run python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

rows = []
for variant, env_name in [
    ("catboost", "CATBOOST_METRICS"),
    ("cvae", "CVAE_METRICS"),
    ("cvae_transformer", "TRANSFORMER_METRICS"),
]:
    data = json.loads(Path(os.environ[env_name]).read_text())
    sensitivity = data.get("sensitivity")
    specificity = data.get("specificity")
    rows.append(
        {
            "variant": variant,
            "disease": data.get("target_group", "diabetes"),
            "auroc": data.get("auroc"),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": (
                None
                if sensitivity is None or specificity is None
                else 0.5 * (float(sensitivity) + float(specificity))
            ),
            "threshold": data.get("threshold"),
            "n_evaluable": data.get("n_evaluable"),
        }
    )

compare_dir = Path(os.environ["COMPARE_DIR"])
compare_dir.mkdir(parents=True, exist_ok=True)
summary = pd.DataFrame(rows)
summary.to_csv(compare_dir / "summary.csv", index=False)
wide = summary.set_index("variant")[["auroc", "balanced_accuracy", "sensitivity", "specificity"]].T
wide["cvae_minus_catboost"] = wide["cvae"] - wide["catboost"]
wide["cvae_transformer_minus_catboost"] = wide["cvae_transformer"] - wide["catboost"]
wide["cvae_transformer_minus_cvae"] = wide["cvae_transformer"] - wide["cvae"]
wide.to_csv(compare_dir / "metrics_wide.csv")
print(summary.to_string(index=False))
print("summary:", compare_dir / "summary.csv")
print("wide:", compare_dir / "metrics_wide.csv")
PY

echo "==> Done."
echo "==> Compare directory: ${COMPARE_DIR}"
