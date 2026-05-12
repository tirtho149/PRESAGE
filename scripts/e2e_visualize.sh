#!/bin/bash
# ============================================================================
# scripts/e2e_visualize.sh
# ============================================================================
# Local post-processing: pull results pushed by e2e_nova, generate every
# visualization, drop figures + LaTeX snippets into the paper, and
# (optionally) rebuild the PDF.
#
#   1. git pull results
#   2. KB stats viz + LaTeX snippet
#   3. OBSERVE training-curve viz + held-out eval viz
#   4. Phase 0R trace stats viz
#   5. LaTeX build (latexmk or pdflatex)
#
# Knobs:
#   PATHOME_GIT_REMOTE    origin
#   PATHOME_GIT_BRANCH    main
#   PATHOME_SKIP_PDF      0   (set to 1 to skip the latexmk step)
# ============================================================================
set -euo pipefail

PATHOME_REPO="${PATHOME_REPO:-$(pwd)}"
cd "$PATHOME_REPO"

GIT_REMOTE="${PATHOME_GIT_REMOTE:-origin}"
GIT_BRANCH="${PATHOME_GIT_BRANCH:-main}"

echo "================================================================="
echo "  e2e_visualize : pull + viz + paper"
echo "================================================================="

# ---- 1. git pull ----------------------------------------------------------
echo
echo "[1/4] git pull results"
git pull "$GIT_REMOTE" "$GIT_BRANCH" --ff-only

# ---- 2-3. All visualizations ---------------------------------------------
echo
echo "[2/4] KB visualizations"
bash scripts/viz_kb.sh || echo "  (skipped)"

echo
echo "[3/4] OBSERVE visualizations"
bash scripts/viz_observe.sh || echo "  (skipped)"

echo
echo "[3.5/4] Phase 0R trace visualizations"
bash scripts/viz_traces.sh || echo "  (skipped)"

# ---- 4. LaTeX build -------------------------------------------------------
if [ "${PATHOME_SKIP_PDF:-0}" != "1" ]; then
  echo
  echo "[4/4] Rebuild paper PDF"
  bash scripts/build_latex_pdf.sh || echo "  (latex build failed — figures still produced)"
else
  echo "  [skip] PATHOME_SKIP_PDF=1"
fi

echo
echo "e2e_visualize complete."
echo "Outputs:"
echo "  results/figures/*.png"
echo "  plantswarm/latex/auto_*.tex"
echo "  plantswarm/latex/acl_latex.pdf  (if PDF build succeeded)"
