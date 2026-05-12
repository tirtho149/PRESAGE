#!/bin/bash
# ============================================================================
# scripts/viz_traces.sh
# ============================================================================
# Generate Phase 0R trace statistics visualizations.
#
# Inputs:
#   $PATHOME_TRACE_DIR/$PATHOME_TRACE_FILE
#     default: artifacts/observe_traces/phase0r_traces.jsonl
#
# Outputs:
#   results/figures/trace_*.png
#   plantswarm/latex/auto_trace_stats.tex
#
# Documented in README under "Script reference".
# ============================================================================
set -euo pipefail

PATHOME_REPO="${PATHOME_REPO:-$(pwd)}"
cd "$PATHOME_REPO"

TRACE_DIR="${PATHOME_TRACE_DIR:-artifacts/observe_traces}"
TRACE_FILE="${PATHOME_TRACE_FILE:-phase0r_traces.jsonl}"
TRACES="$TRACE_DIR/$TRACE_FILE"

if [ ! -f "$TRACES" ]; then
  echo "ERROR: trace JSONL not found at $TRACES"
  echo "Run Phase 0R with PATHOME_TRACE_DIR set to produce it."
  exit 1
fi

python3 scripts/viz/trace_stats.py --traces "$TRACES"
