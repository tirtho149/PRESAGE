#!/bin/bash
# ============================================================================
# submit_pathome_all.sh
# ============================================================================
# Master submission script for the symptom-centric Pathome pipeline.
# Queues all six Pathome phases on Nova with sbatch dependency chains:
#
#   Phase 0 (CPU)       → seed PathomeDB visual blocks via `claude -p`
#   Phase 1 (CPU+net)   → build PathomeDB v1_seed (Claude visuals + geo + refs)
#   Phase 2 (A100+vLLM) → 101,640 PlantSwarm routing traces (seed DB)
#   Phase 3 (CPU)       → enhance DB from traces → v1_enhanced
#   Phase 4 (A100)      → train OBSERVE × 2  (seed DB and enhanced DB)
#   Phase 5 (A100+CPU)  → eval × 2 on PV + PW, then comparison.{json,md,tex}
#
# Usage:
#   bash scripts/submit_pathome_all.sh
#
# Override per-phase tunables via environment variables (see each script's
# header). The chain-script itself accepts:
#   PATHOME_SKIP="0,1"           # skip phase 0 + 1 (e.g. seed file already on disk)
#   PATHOME_FROM_PHASE=2         # start from phase 2
# ============================================================================

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   Submitting Pathome Pipeline (symptom-centric) to Nova       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo

phase_in_skip() {
  local phase="$1"; local skip="${PATHOME_SKIP:-}"
  [[ ",$skip," == *",$phase,"* ]]
}

phase_active() {
  local phase="$1"
  if phase_in_skip "$phase"; then return 1; fi
  if [ -n "${PATHOME_FROM_PHASE:-}" ] && [ "$phase" -lt "$PATHOME_FROM_PHASE" ]; then
    return 1
  fi
  return 0
}

submit() {
  # submit <name> <script> [dependency-job-id]
  local name="$1" script="$2" dep="$3"
  local dep_arg=""
  [ -n "$dep" ] && dep_arg="--dependency=afterok:$dep"
  local jid
  jid=$(sbatch $dep_arg "$script" 2>&1 | grep "Submitted batch job" | awk '{print $NF}')
  echo "✓ $name submitted: $jid${dep:+  (depends on $dep)}"
  echo "$jid"
}

chmod +x scripts/submit_pathome_phase*.sh

PREV=""

if phase_active 0; then
  echo "── Phase 0: Claude headless seed (~15 min, CPU) ──"
  PREV=$(submit "Phase 0" scripts/submit_pathome_phase0_seed.sh "$PREV")
fi

if phase_active 1; then
  echo "── Phase 1: Build PathomeDB v1_seed (~30 min, CPU+net) ──"
  PREV=$(submit "Phase 1" scripts/submit_pathome_phase1_build.sh "$PREV")
fi

if phase_active 2; then
  echo "── Phase 2: 101,640 PlantSwarm traces (~36-50 h, A100+vLLM) ──"
  PREV=$(submit "Phase 2" scripts/submit_pathome_phase2_traces.sh "$PREV")
fi

if phase_active 3; then
  echo "── Phase 3: Enhance DB from traces (~5 min, CPU) ──"
  PREV=$(submit "Phase 3" scripts/submit_pathome_phase3_enhance.sh "$PREV")
fi

if phase_active 4; then
  echo "── Phase 4: Train OBSERVE × 2 (~20-24 h, A100) ──"
  PREV=$(submit "Phase 4" scripts/submit_pathome_phase4_train.sh "$PREV")
fi

if phase_active 5; then
  echo "── Phase 5: Eval × 4 + comparison (~6-8 h, A100+CPU) ──"
  PREV=$(submit "Phase 5" scripts/submit_pathome_phase5_eval.sh "$PREV")
fi

echo
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    Pipeline Submitted ✓                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo
echo "Monitor:    squeue -u \$USER"
echo "Live logs:  tail -f logs/pathome_phase*-*.out"
echo "Final out:  results/pathome_compare/comparison.md"
echo "Submitted at: $(date)"
