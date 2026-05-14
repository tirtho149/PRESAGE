#!/bin/bash
# ============================================================================
# scripts/viz_kb.sh
# ============================================================================
# Generate KB summary visualizations from symptoms_seed.json.
#
# Inputs:
#   $PATHOME_SEED_FILE    (default: artifacts/pathome_seed/symptoms_seed.json)
#
# Outputs:
#   results/figures/kb_*.png
#   paper/auto_kb_stats.tex
#
# Documented in README under "Script reference".
# ============================================================================
set -euo pipefail

PATHOME_REPO="${PATHOME_REPO:-$(pwd)}"
cd "$PATHOME_REPO"

SEED="${PATHOME_SEED_FILE:-artifacts/pathome_seed/symptoms_seed.json}"
if [ ! -f "$SEED" ]; then
  echo "ERROR: seed JSON not found at $SEED"
  echo "Run Phase 0 (and Phase 0R) first to produce it."
  exit 1
fi

python3 scripts/viz/kb_stats.py --seed "$SEED"
