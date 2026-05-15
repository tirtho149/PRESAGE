#!/bin/bash
#SBATCH --job-name=vllm_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --partition=nova
#SBATCH --output=logs/vllm_smoke-%j.out
#SBATCH --error=logs/vllm_smoke-%j.err

# ============================================================================
# vLLM smoke test — boots vLLM with the same flags as Phase 0R, then runs
# tests/test_vllm_smoke.py against the local server. No crops, no CSV
# iteration; verifies every request path in 5-10 minutes.
#
# Override at submit time (same knobs as Phase 0R):
#   VLLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
#   VLLM_MAX_MODEL_LEN=32768
#   VLLM_MM_KWARGS='{"min_pixels":50176,"max_pixels":1003520}'
#   VLLM_PORT=8000
# ============================================================================

set -e
PATHOME_REPO="${PATHOME_REPO:-$(pwd)}"
cd "$PATHOME_REPO"

echo "================================"
echo "vLLM smoke  Job: $SLURM_JOB_ID  Start: $(date)"
echo "================================"

module load python cuda/12.8

VENV="${PATHOME_VENV:-$PATHOME_REPO/.venv}"
if [ ! -f "$VENV/bin/activate" ]; then
  if [ -f "$(dirname "$PATHOME_REPO")/.venv/bin/activate" ]; then
    VENV="$(dirname "$PATHOME_REPO")/.venv"
  else
    echo "ERROR: no venv at $VENV"; exit 2
  fi
fi
echo "venv: $VENV"
source "$VENV/bin/activate"
mkdir -p logs

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
MM_KWARGS_DEFAULT='{"min_pixels":50176,"max_pixels":1003520}'
MM_KWARGS="${VLLM_MM_KWARGS:-$MM_KWARGS_DEFAULT}"

VLLM_LOG="logs/vllm_smoke-vllm-${SLURM_JOB_ID}.log"
echo "[vllm] booting $MODEL on :$PORT (max_model_len=$MAX_MODEL_LEN)"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port  "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --mm-processor-kwargs "$MM_KWARGS" \
  --limit-mm-per-prompt image=1 \
  --trust-remote-code \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'echo "[trap] killing vllm pid=$VLLM_PID"; kill $VLLM_PID 2>/dev/null || true' EXIT

export VLLM_BASE_URL="http://localhost:${PORT}/v1"
export VLLM_MODEL="$MODEL"
echo "[vllm] waiting for $VLLM_BASE_URL/models ..."
for i in $(seq 1 60); do
  if curl -sf --max-time 5 "$VLLM_BASE_URL/models" >/dev/null 2>&1; then
    echo "[vllm] up after $((i*10))s"
    break
  fi
  sleep 10
done
if ! curl -sf --max-time 5 "$VLLM_BASE_URL/models" >/dev/null 2>&1; then
  echo "[vllm] FAILED to come up — $VLLM_LOG context:"
  echo "---- head -n 80 ----"; head -n 80 "$VLLM_LOG"
  echo "---- tail -n 200 ----"; tail -n 200 "$VLLM_LOG"
  exit 1
fi

echo "================================"
echo "running smoke tests"
echo "================================"
python tests/test_vllm_smoke.py
RC=$?

echo
echo "vLLM smoke complete: $(date) (exit=$RC)"
exit $RC
