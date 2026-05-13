#!/bin/bash
# ============================================================================
# scripts/viz_all.sh
# ============================================================================
# Run every visualization script in sequence: KB stats and Phase 0R trace
# stats. Each writes both a PNG figure and a LaTeX snippet under
# plantswarm/latex/auto_*.tex for the paper to \input.
#
# The PathomeOOD paper-table reproduction is produced separately by
# scripts/aggregate_pathomeood_tables.py (writes markdown under results/tables/
# and a master results/pathomeood_report.md).
# ============================================================================
set -euo pipefail

PATHOME_REPO="${PATHOME_REPO:-$(pwd)}"
cd "$PATHOME_REPO"

echo "================================================================="
echo "  Running all visualizations"
echo "================================================================="

bash scripts/viz_kb.sh     || echo "  viz_kb.sh failed (continuing)"
bash scripts/viz_traces.sh || echo "  viz_traces.sh failed (continuing)"

echo
echo "All viz outputs:"
echo "  results/figures/*.png"
echo "  plantswarm/latex/auto_*.tex"
echo "  results/tables/*.md  (run scripts/aggregate_pathomeood_tables.py)"
echo "  results/pathomeood_report.md"
