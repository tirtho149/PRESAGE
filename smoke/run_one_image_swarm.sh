#!/bin/bash
# smoke/run_one_image_swarm.sh
# One-image full-swarm smoke for Soybean. Run on a GPU node you ALREADY
# hold (salloc) — uses bash, not sbatch. Sets the same env the real
# Phase 0R job uses (HF cache on /work, .env/HF_TOKEN, verifier OFF),
# then runs smoke/run_one_image_swarm.py.
#
#   bash smoke/run_one_image_swarm.sh
#
# Override knobs inline, e.g.:
#   SWARM_GRANULARITY=grouped VLLM_N_RUNS=2 bash smoke/run_one_image_swarm.sh
#   SMOKE_DISEASE="Brown Stem Rot" bash smoke/run_one_image_swarm.sh
set -uo pipefail

REPO="${PATHOME_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

# venv
VENV="${PATHOME_VENV:-$REPO/.venv}"
[ -f "$VENV/bin/activate" ] || VENV="$(dirname "$REPO")/.venv"
# shellcheck disable=SC1091
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

# secrets (HF_TOKEN) — gitignored .env
if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi

# HF model cache on the big shared fs (download once, reuse)
export HF_HOME="${HF_HOME:-$REPO/.hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
mkdir -p "$HF_HOME"

# swarm-only smoke: no Claude verifier on a GPU node
export PATHOME_USE_VERIFIER="${PATHOME_USE_VERIFIER:-0}"
export PATHOME_IMAGE_CACHE_DIR="${PATHOME_IMAGE_CACHE_DIR:-$REPO/.bugwood_cache}"
# faithful but quick defaults (override inline)
export CROP="${CROP:-Soybean}"
export VLLM_N_RUNS="${VLLM_N_RUNS:-3}"
export VLLM_AGREEMENT_MIN="${VLLM_AGREEMENT_MIN:-2}"
export VLLM_SWARM_ROUNDS="${VLLM_SWARM_ROUNDS:-2}"   # 2 = full swarm

echo "[smoke] repo=$REPO venv=$VENV HF_HOME=$HF_HOME"
echo "[smoke] CROP=$CROP N=$VLLM_N_RUNS K=$VLLM_AGREEMENT_MIN rounds=$VLLM_SWARM_ROUNDS gran=${SWARM_GRANULARITY:-routed}"
python smoke/run_one_image_swarm.py
