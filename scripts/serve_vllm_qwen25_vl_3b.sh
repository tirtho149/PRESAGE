#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server for PlantSwarm smoke runs using
# Qwen/Qwen2.5-VL-3B-Instruct (vision-language; fits many ~8GB GPUs; tune if you have ~6GB).
#
# Requires: Linux + NVIDIA GPU + CUDA. vLLM does not run this stack on macOS.
#
# Install (on the GPU machine, separate venv recommended):
#   pip install -U vllm
#
# Usage:
#   bash scripts/serve_vllm_qwen25_vl_3b.sh
# Then in another shell (or over SSH -L 8000:localhost:8000):
#   python scripts/run_plantswarm.py --config configs/qwen25_vl_3b_smoke.yaml --subset 5
#
# If you hit CUDA OOM, try lowering --max-model-len (e.g. 2048) or use an AWQ/FP8
# checkpoint if your vLLM version supports it for this model.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec vllm serve "Qwen/Qwen2.5-VL-3B-Instruct" \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image":4,"video":0}'
