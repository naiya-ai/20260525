#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/train/train_conditional_vae_transformer_large.yaml}" \
VARIANT="${VARIANT:-transformer_large}" \
./scripts/train/train_eval_transformer_cvae.sh "$@"
