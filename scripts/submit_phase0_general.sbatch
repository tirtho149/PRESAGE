#!/bin/bash
#SBATCH --job-name=pathome_phase0_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=nova
#SBATCH --output=logs/pathome_phase0_general-%j.out
#SBATCH --error=logs/pathome_phase0_general-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Phase 0 (canonical / GENERAL KB) — SLURM wrapper around run_phase0_local.sh
# ============================================================================
# Builds final_registry.json for crops that only have discovery_results.json.
# CPU-only + network: all model calls go through the headless `claude -p`
# CLI (WebSearch discovery + extraction + reconciliation). NO GPU, NO vLLM.
#
# Discovery is already cached per crop (discovery_results.json), so we
# resume from EXTRACTION — the pipeline reuses cached upstream artefacts
# and is idempotent (a crop with a final_registry.json is skipped).
#
# Reads:   artifacts/pathome_kb/<Crop>/discovery_results.json
#          BugWood_Diseases_usable.csv
# Writes:  artifacts/pathome_kb/<Crop>/final_registry.json  (+ registry.md, xlsx)
#          artifacts/pathome_seed/symptoms_seed_general25.json  (scoped seed —
#                                          does NOT clobber the full seed)
#
# Override at submit time:
#   PATHOME_SEED_ONLY_CROPS="Potato,Pepper"  sbatch scripts/submit_phase0_general.sbatch
# ============================================================================

set -e
PATHOME_REPO="${PATHOME_REPO:-$(pwd)}"
cd "$PATHOME_REPO"

# ---- persistent log (mirror of the regional script's approach) -------------
PATHOME_LOG_DIR="${PATHOME_LOG_DIR:-$PATHOME_REPO/logs}"
mkdir -p "$PATHOME_LOG_DIR"
RUNLOG="$PATHOME_LOG_DIR/phase0_general_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUNLOG") 2>&1
echo "[log] full job output -> $RUNLOG"

echo "================================"
echo "Phase 0: canonical/general KB (claude -p WebSearch, CPU)"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)  Start: $(date)"
echo "================================"

# connectivity: claude -p WebSearch needs the internet on the compute node
echo "[net] connectivity test:"
curl -sI --max-time 15 https://www.anthropic.com | head -1 || echo "  WARNING: NO INTERNET ON COMPUTE NODE"

module load python

VENV="${PATHOME_VENV:-$PATHOME_REPO/.venv}"
if [ ! -f "$VENV/bin/activate" ]; then
  if [ -f "$(dirname "$PATHOME_REPO")/.venv/bin/activate" ]; then
    VENV="$(dirname "$PATHOME_REPO")/.venv"
  else
    echo "ERROR: no venv found (tried $VENV). Set PATHOME_VENV=/path/to/venv."
    exit 2
  fi
fi
echo "venv: $VENV"
source "$VENV/bin/activate"
mkdir -p logs

# claude CLI must be on PATH for the headless `claude -p` calls.
export PATH="/work/mech-ai-scratch/tirtho/npm-global/bin:$PATH"
if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' CLI not on PATH"
  exit 2
fi
echo "[claude] $(command -v claude)  version: $(claude --version 2>/dev/null | head -1)"

# ---- the 24 buildable crops that have discovery but no final_registry ------
# ("Plum And Cherry" is intentionally excluded — 0 diseases in the usable CSV.)
DEFAULT_CROPS="Birch,Boxwood,Camellia,Juniper,Peach,Peanut,Pecan,Peony,Pepper,Phlox,Pine,Potato,Red Pine,Rhododendrons And Azaleas,Rose,Scots Pine,Serviceberry,Spinach,Squash,St. Augustinegrass,Strawberry,Sugar Beet,Sweet Cherry,Witch-Hazel"

export PATHOME_SEED_ONLY_CROPS="${PATHOME_SEED_ONLY_CROPS:-$DEFAULT_CROPS}"
export PATHOME_SEED_RESUME="${PATHOME_SEED_RESUME:-extraction}"
# scoped seed file so we don't overwrite the full artifacts/pathome_seed/symptoms_seed.json
export PATHOME_SEED_FILE="${PATHOME_SEED_FILE:-artifacts/pathome_seed/symptoms_seed_general25.json}"
export PYTHONUNBUFFERED=1

echo "[crops] $PATHOME_SEED_ONLY_CROPS"
echo "[resume] $PATHOME_SEED_RESUME   [out] $PATHOME_SEED_FILE"
echo "================================"

stdbuf -oL -eL bash scripts/run_phase0_local.sh

echo
echo "Phase 0 (general KB) complete: $(date)"
echo "built registries:"
for c in Birch Boxwood Camellia Juniper Peach Peanut Pecan Peony Pepper Phlox Pine Potato "Red Pine" "Rhododendrons And Azaleas" Rose "Scots Pine" Serviceberry Spinach Squash "St. Augustinegrass" Strawberry "Sugar Beet" "Sweet Cherry" Witch-Hazel; do
  f="artifacts/pathome_kb/$c/final_registry.json"
  [ -f "$f" ] && echo "  OK   $c" || echo "  MISS $c"
done
