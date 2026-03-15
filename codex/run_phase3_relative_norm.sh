#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <model_name> <cuda_devices> [population_size]" >&2
  exit 1
fi

MODEL_NAME="$1"
CUDA_DEVICES="$2"
POPULATION_SIZE="${3:-200}"
SIGMAS=(0.0025 0.005 0.01 0.02)

model_slug="$(basename "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr '/.' '_' | tr '-' '_')"

for sigma in "${SIGMAS[@]}"; do
  logfile="codex/phase3_${model_slug}_tensor_std_sigma_${sigma}.log"
  echo "[$(date --iso-8601=seconds)] starting sigma=${sigma}" >> "$logfile"
  .venv/bin/python -u randopt.py \
    --dataset countdown \
    --model_name "$MODEL_NAME" \
    --train_samples 100 \
    --test_samples 100 \
    --population_size "$POPULATION_SIZE" \
    --top_k_ratios 0.025,0.05,0.1 \
    --sigma_values "$sigma" \
    --num_engines 1 \
    --tp 1 \
    --cuda_devices "$CUDA_DEVICES" \
    --perturb_seed_mode global_stream \
    --perturb_scale_mode tensor_std \
    --global_seed 42 \
    --max_tokens 1024 \
    --experiment_dir codex/experiments >> "$logfile" 2>&1
  echo "[$(date --iso-8601=seconds)] finished sigma=${sigma}" >> "$logfile"
done
