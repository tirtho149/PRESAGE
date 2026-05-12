#!/bin/bash
# ============================================================================
# scripts/viz_observe.sh
# ============================================================================
# Generate OBSERVE training curves + held-out eval visualizations.
#
# Inputs:
#   $OBSERVE_SAVE_DIR/history.json   (default: observe/checkpoints/history.json)
#   $OBSERVE_EVAL_OUT                (default: results/observe_eval.json)
#
# Outputs:
#   results/figures/observe_*.png
#   plantswarm/latex/auto_observe_{curves,eval}.tex
#
# Documented in README under "Script reference".
# ============================================================================
set -euo pipefail

PATHOME_REPO="${PATHOME_REPO:-$(pwd)}"
cd "$PATHOME_REPO"

SAVE_DIR="${OBSERVE_SAVE_DIR:-observe/checkpoints}"
EVAL_OUT="${OBSERVE_EVAL_OUT:-results/observe_eval.json}"
HISTORY="$SAVE_DIR/history.json"

if [ -f "$HISTORY" ]; then
  echo "  observe_curves: reading $HISTORY"
  python3 scripts/viz/observe_curves.py --history "$HISTORY"
else
  echo "  (skipping observe_curves: $HISTORY not found)"
fi

if [ -f "$EVAL_OUT" ]; then
  echo "  observe_eval: reading $EVAL_OUT"
  python3 scripts/viz/observe_eval.py --eval "$EVAL_OUT"
else
  echo "  (skipping observe_eval: $EVAL_OUT not found)"
fi
