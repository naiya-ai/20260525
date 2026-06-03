#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-model_experiment/classification_ddp_layer_norm_lr1e-4}" \
LR="${LR:-0.0001}" \
NORMALIZATION="${NORMALIZATION:-layer_norm}" \
./scripts/model_experiment/run_raw_mlp_classifier_ddp_no_bn_width_sweep.sh
