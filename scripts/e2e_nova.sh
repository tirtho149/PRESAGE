#!/bin/bash
# ============================================================================
# scripts/e2e_nova.sh
# ============================================================================
# GPU-host part of the end-to-end pipeline. Run THIS on Nova (or any host
# with vLLM-capable GPUs):
#   1. git pull canonical artefacts pushed by e2e_local.sh
#   2. Phase 0R: vLLM boot + Qwen swarm + Claude web-search verifier
#   3. OBSERVE training (BC on per-step routing traces)
#   4. OBSERVE held-out evaluation
#   5. git push results
#
# This script DOES NOT block waiting for sbatch jobs — it submits each
# phase with `sbatch --wait` so the chain runs sequentially in the
# foreground and returns only after every phase completes.
#
# Knobs (env vars; safe defaults shown):
#   PATHOME_USABLE_CSV       BugWood_Diseases_usable.csv
#   PATHOME_SEED_FILE        artifacts/pathome_seed/symptoms_seed.json
#   PATHOME_TRACE_DIR        artifacts/observe_traces
#   OBSERVE_SAVE_DIR         observe/checkpoints
#   OBSERVE_EVAL_OUT         results/observe_eval.json
#   PATHOME_ONLY_CROPS       (optional crop allowlist)
#   PATHOME_GIT_REMOTE       origin
#   PATHOME_GIT_BRANCH       main
#   PATHOME_SKIP_PHASE0R     0
#   PATHOME_SKIP_TRAIN       0
#   PATHOME_SKIP_EVAL        0
#   PATHOME_SKIP_PUSH        0
# ============================================================================
set -euo pipefail

PATHOME_REPO="${PATHOME_REPO:-$(pwd)}"
cd "$PATHOME_REPO"

TRACE_DIR="${PATHOME_TRACE_DIR:-artifacts/observe_traces}"
SEED="${PATHOME_SEED_FILE:-artifacts/pathome_seed/symptoms_seed.json}"
SAVE_DIR="${OBSERVE_SAVE_DIR:-observe/checkpoints}"
EVAL_OUT="${OBSERVE_EVAL_OUT:-results/observe_eval.json}"
GIT_REMOTE="${PATHOME_GIT_REMOTE:-origin}"
GIT_BRANCH="${PATHOME_GIT_BRANCH:-main}"

echo "================================================================="
echo "  e2e_nova : pull + Phase 0R + OBSERVE train + eval + push"
echo "================================================================="

# ---- 1. git pull ----------------------------------------------------------
echo
echo "[1/5] git pull canonical artefacts"
git pull "$GIT_REMOTE" "$GIT_BRANCH" --ff-only

mkdir -p logs "$TRACE_DIR" "$SAVE_DIR" "$(dirname "$EVAL_OUT")"

# ---- 2. Phase 0R (sbatch --wait) ------------------------------------------
if [ "${PATHOME_SKIP_PHASE0R:-0}" != "1" ]; then
  echo
  echo "[2/5] Phase 0R — Qwen swarm + Claude verifier (sbatch --wait)"
  PATHOME_TRACE_DIR="$TRACE_DIR" \
    sbatch --wait scripts/submit_phase0r_regional.sh
else
  echo "  [skip] PATHOME_SKIP_PHASE0R=1"
fi

# ---- 3. OBSERVE training (sbatch --wait) ----------------------------------
if [ "${PATHOME_SKIP_TRAIN:-0}" != "1" ]; then
  echo
  echo "[3/5] OBSERVE training (sbatch --wait)"
  PATHOME_TRACE_FILE="$TRACE_DIR/phase0r_traces.jsonl" \
    OBSERVE_SAVE_DIR="$SAVE_DIR" \
    sbatch --wait scripts/submit_observe_train.sh
else
  echo "  [skip] PATHOME_SKIP_TRAIN=1"
fi

# ---- 4. OBSERVE evaluation (sbatch --wait) --------------------------------
if [ "${PATHOME_SKIP_EVAL:-0}" != "1" ]; then
  echo
  echo "[4/5] OBSERVE held-out evaluation (sbatch --wait)"
  OBSERVE_CKPT="$SAVE_DIR/observe_best.pt" \
    OBSERVE_EVAL_OUT="$EVAL_OUT" \
    PATHOME_TRACE_FILE="$TRACE_DIR/phase0r_traces.jsonl" \
    sbatch --wait scripts/submit_evaluate_observe.sh
else
  echo "  [skip] PATHOME_SKIP_EVAL=1"
fi

# ---- 5. git push results --------------------------------------------------
echo
echo "[5/5] Git push results"
git add -f "$SEED" \
           "$EVAL_OUT" \
           "$SAVE_DIR/history.json" \
           "$TRACE_DIR/phase0r_traces.jsonl" \
           artifacts/pathome_kb/*/final_registry.json 2>/dev/null || true
# observe_best.pt is typically too large for git; explicit add only if user opts in.
if [ "${PATHOME_PUSH_CHECKPOINT:-0}" = "1" ]; then
  git add -f "$SAVE_DIR/observe_best.pt"
fi
if git diff --cached --quiet; then
  echo "  no result artefacts changed; skipping commit"
else
  git commit -m "Phase 0R + OBSERVE: results ($(date -u +%Y-%m-%dT%H:%MZ))"
fi
if [ "${PATHOME_SKIP_PUSH:-0}" = "1" ]; then
  echo "  PATHOME_SKIP_PUSH=1 — committed but not pushing"
else
  git push "$GIT_REMOTE" "$GIT_BRANCH"
fi

echo
echo "e2e_nova complete."
echo "Next step: back on the LOCAL machine, run scripts/e2e_visualize.sh"
