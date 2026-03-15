#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <model_name> <cuda_device> [population_size]" >&2
  exit 1
fi

MODEL_NAME="$1"
CUDA_DEVICE="$2"
POPULATION_SIZE="${3:-100}"
SIGMAS=(0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01)

model_slug="$(basename "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr '/.' '_' | tr '-' '_')"

for sigma in "${SIGMAS[@]}"; do
  logfile="codex/density_${model_slug}_sigma_${sigma}.log"
  echo "[$(date --iso-8601=seconds)] starting sigma=${sigma}" >> "$logfile"
  .venv/bin/python -u randopt.py \
    --dataset countdown \
    --model_name "$MODEL_NAME" \
    --train_samples 200 \
    --test_samples 2000 \
    --population_size "$POPULATION_SIZE" \
    --top_k_ratios 0.01 \
    --sigma_values "$sigma" \
    --num_engines 1 \
    --tp 1 \
    --cuda_devices "$CUDA_DEVICE" \
    --perturb_seed_mode global_stream \
    --perturb_scale_mode absolute \
    --global_seed 42 \
    --max_tokens 1024 \
    --experiment_dir codex/experiments >> "$logfile" 2>&1
  echo "[$(date --iso-8601=seconds)] finished sigma=${sigma}" >> "$logfile"
done
