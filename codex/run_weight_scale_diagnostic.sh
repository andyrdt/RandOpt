#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <cuda_device> [model ...]" >&2
  exit 1
fi

CUDA_DEVICE="$1"
shift

if [ "$#" -eq 0 ]; then
  set -- \
    "Qwen/Qwen2.5-0.5B-Instruct" \
    "Qwen/Qwen2.5-1.5B-Instruct" \
    "Qwen/Qwen2.5-3B-Instruct" \
    "Qwen/Qwen2.5-7B-Instruct" \
    "Qwen/Qwen2.5-32B-Instruct"
fi

.venv/bin/python codex/weight_scale_diagnostic.py \
  --cuda-device "cuda:${CUDA_DEVICE}" \
  --models "$@"

.venv/bin/python codex/plot_weight_scales.py
