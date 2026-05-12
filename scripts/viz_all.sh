#!/bin/bash
# ============================================================================
# scripts/viz_all.sh
# ============================================================================
# Run every visualization script in sequence: KB stats, OBSERVE curves +
# eval, Phase 0R trace stats. Each writes both a PNG figure and a LaTeX
# snippet under plantswarm/latex/auto_*.tex for the paper to \input.
#
# Skips gracefully when inputs are missing (e.g. OBSERVE eval not yet run).
#
# Documented in README under "Script reference".
# ============================================================================
set -euo pipefail

PATHOME_REPO="${PATHOME_REPO:-$(pwd)}"
cd "$PATHOME_REPO"

echo "================================================================="
echo "  Running all visualizations"
echo "================================================================="

bash scripts/viz_kb.sh      || echo "  viz_kb.sh failed (continuing)"
bash scripts/viz_observe.sh || echo "  viz_observe.sh failed (continuing)"
bash scripts/viz_traces.sh  || echo "  viz_traces.sh failed (continuing)"

echo
echo "All viz outputs:"
echo "  results/figures/*.png"
echo "  plantswarm/latex/auto_*.tex"
