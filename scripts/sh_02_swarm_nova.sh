#!/bin/bash
# ============================================================================
# scripts/sh_02_swarm_nova.sh           STEP 2 — NOVA
# ============================================================================
# Pull canonical KB from GitHub, run the 24-agent 2-round real swarm on
# Nova (Qwen2.5-VL-7B on vLLM) WITHOUT the Claude verifier — verification
# moves to LOCAL in step 3. Then push the unverified-deltas KB back to
# GitHub.
#
# What this produces
#   For each crop in $CROPS, each disease, each state with a cached image:
#     - 24 specialists × 2 rounds × N=10 stochastic passes
#     - K-of-N agreement filter
#     - merged into artifacts/pathome_kb/<Crop>/final_registry.json
#       under regional_observations[<state>].deltas[]
#     - each delta tagged verification_status="unverified" (LOCAL fills
#       this in during step 3 via Claude web verifier)
#
# Per-tuple cost ~15-25 min on one A100. For one disease in one state.
# Total walltime: smoke (2 crops, ~30 states) ~3-6 h; production
# (484 (crop, disease), avg 5 states each = ~2400 tuples) ~24-48 h.
#
# Knobs
#   CROPS                  default "smoke" (Soybean+Tomato); "all" = no filter
#   VLLM_N_RUNS            stochastic passes per tuple (default 10; smoke 5)
#   VLLM_SWARM_ROUNDS      real-swarm rounds (default 2; 1 = legacy)
#   VLLM_AGREEMENT_MIN     K-of-N agreement floor (default 3; smoke 2)
#   PATHOME_TRACE_DIR      set to capture per-pass traces
# ============================================================================
set -euo pipefail

REPO_ROOT="${PATHOME_REPO:-$(pwd)}"
cd "$REPO_ROOT"

CROPS="${CROPS:-smoke}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"

case "$CROPS" in
  smoke) PATHOME_ONLY_CROPS="Soybean,Tomato"; SEED_QUICK=1;;
  all)   PATHOME_ONLY_CROPS="";              SEED_QUICK=0;;
  *)     PATHOME_ONLY_CROPS="$CROPS";        SEED_QUICK=0;;
esac

echo "================================================================="
echo " STEP 2 — Phase 0R 24-agent real swarm (NOVA)"
echo "================================================================="
echo "  CROPS                : ${PATHOME_ONLY_CROPS:-(all)}"
echo "  PATHOME_SEED_QUICK   : $SEED_QUICK"
echo "  VLLM_N_RUNS          : ${VLLM_N_RUNS:-default}"
echo "  VLLM_SWARM_ROUNDS    : ${VLLM_SWARM_ROUNDS:-2}"
echo "  VLLM_AGREEMENT_MIN   : ${VLLM_AGREEMENT_MIN:-default}"
echo "  verifier             : OFF (PATHOME_USE_VERIFIER=0; LOCAL handles it in step 3)"
echo

# Pull canonical KB pushed by step 1.
echo "[1/3] git pull canonical artifacts"
git pull "$GIT_REMOTE" "$GIT_BRANCH" --ff-only
mkdir -p logs "${PATHOME_TRACE_DIR:-artifacts/swarm_traces}"

# Run the swarm (no verifier on Nova).
echo
echo "[2/3] sbatch --wait scripts/submit_phase0r_regional.sh"
echo "       (24 specialists, 2 rounds, N=${VLLM_N_RUNS:-default} passes)"
PATHOME_ONLY_CROPS="$PATHOME_ONLY_CROPS" \
PATHOME_SEED_QUICK="$SEED_QUICK" \
PATHOME_USE_VERIFIER=0 \
VLLM_SWARM_ROUNDS="${VLLM_SWARM_ROUNDS:-2}" \
${VLLM_N_RUNS:+VLLM_N_RUNS=$VLLM_N_RUNS} \
${VLLM_AGREEMENT_MIN:+VLLM_AGREEMENT_MIN=$VLLM_AGREEMENT_MIN} \
${PATHOME_TRACE_DIR:+PATHOME_TRACE_DIR=$PATHOME_TRACE_DIR} \
  sbatch --wait scripts/submit_phase0r_regional.sh

# Push unverified KB back.
echo
echo "[3/3] git push unverified-deltas KB back to $GIT_REMOTE $GIT_BRANCH"
git add -f artifacts/pathome_kb/*/final_registry.json
if [ -n "${PATHOME_TRACE_DIR:-}" ] && [ -d "$PATHOME_TRACE_DIR" ]; then
  git add -f "$PATHOME_TRACE_DIR"/*.jsonl 2>/dev/null || true
fi
if git diff --cached --quiet; then
  echo "  no new deltas to commit"
else
  git commit -m "Phase 0R (NOVA): 24-agent swarm deltas (unverified) for ${PATHOME_ONLY_CROPS:-all crops}"
  if [ "${PATHOME_SKIP_PUSH:-0}" = "1" ]; then
    echo "  PATHOME_SKIP_PUSH=1 — committed but not pushing"
  else
    git push "$GIT_REMOTE" "$GIT_BRANCH"
  fi
fi

echo
echo "STEP 2 done."
echo "  Next: back on LOCAL, run scripts/sh_03_validate_local.sh"
