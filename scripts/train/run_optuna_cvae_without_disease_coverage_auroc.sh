#!/usr/bin/env bash
set -euo pipefail

DISEASE="${DISEASE:-diabetes}"
TOTAL_TRIALS="${TOTAL_TRIALS:-200}"
GPUS=(${GPUS:-4 5 6 7})
WORKERS="${#GPUS[@]}"
TRIALS_PER_WORKER="${TRIALS_PER_WORKER:-$(( (TOTAL_TRIALS + WORKERS - 1) / WORKERS ))}"
STEPS="${STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
AUROC_SAMPLES="${AUROC_SAMPLES:-200}"
COVERAGE_SAMPLES="${COVERAGE_SAMPLES:-200}"
MAX_COV90_ERR="${MAX_COV90_ERR:-0.05}"
SWEEP_ID="${SWEEP_ID:-optuna_cvae_without_disease_${DISEASE}_cov005_auroc_${TOTAL_TRIALS}t_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/optuna/${SWEEP_ID}}"
STUDY_NAME="${STUDY_NAME:-${SWEEP_ID}}"
STORAGE="${STORAGE:-sqlite:///${OUTPUT_ROOT}/optuna_study.db}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

mkdir -p "${OUTPUT_ROOT}" logs/optuna

echo "SWEEP_ID=${SWEEP_ID}" | tee "${OUTPUT_ROOT}/run.info"
echo "DISEASE=${DISEASE}" | tee -a "${OUTPUT_ROOT}/run.info"
echo "GPUS=${GPUS[*]}" | tee -a "${OUTPUT_ROOT}/run.info"
echo "TRIALS_PER_WORKER=${TRIALS_PER_WORKER}" | tee -a "${OUTPUT_ROOT}/run.info"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}" | tee -a "${OUTPUT_ROOT}/run.info"

for GPU in "${GPUS[@]}"; do
  WORKER_LOG="logs/optuna/${SWEEP_ID}_gpu${GPU}.log"
  nohup "${PYTHON_BIN}" scripts/train/optuna_cvae_coverage_constrained_auroc.py \
    --config configs/train/train_conditional_vae_eddi_z16_beta010.yaml \
    --output-root "${OUTPUT_ROOT}" \
    --study-name "${STUDY_NAME}" \
    --storage "${STORAGE}" \
    --n-trials "${TRIALS_PER_WORKER}" \
    --target-group "${DISEASE}" \
    --dataset-root datasets/preprocessed/gaussian_quantile \
    --dataset-name harmonized_knhanes_1998_2024 \
    --source-groups questionnaire_without_disease dietary \
    --device cuda \
    --cuda-visible-devices "${GPU}" \
    --steps "${STEPS}" \
    --batch-sizes "${BATCH_SIZE}" \
    --auroc-num-samples "${AUROC_SAMPLES}" \
    --coverage-num-samples "${COVERAGE_SAMPLES}" \
    --max-cov90-abs-error "${MAX_COV90_ERR}" \
    --beta-min 0.001 \
    --beta-max 0.3 \
    --latent-dims 16 32 64 128 \
    --beta-warmup-steps 500 1000 1500 \
    --width-multipliers 1 2 4 \
    --use-latest-checkpoint \
    > "${WORKER_LOG}" 2>&1 &
  echo "$!" | tee "${OUTPUT_ROOT}/worker_gpu${GPU}.pid"
  echo "started gpu ${GPU}: pid $(cat "${OUTPUT_ROOT}/worker_gpu${GPU}.pid"), log ${WORKER_LOG}"
done

echo "tail -f logs/optuna/${SWEEP_ID}_gpu4.log"
echo "watch -n 30 '${PYTHON_BIN} - <<PY
import pandas as pd
from pathlib import Path
p=Path(\"${OUTPUT_ROOT}/trials.csv\")
print(p)
if p.exists():
    df=pd.read_csv(p)
    print(df[[\"number\",\"value\",\"state\",\"user_attrs_metric_auroc\",\"user_attrs_metric_cov90_abs_error\",\"user_attrs_status\"]].tail(10).to_string(index=False))
PY'"
