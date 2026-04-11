#!/usr/bin/env bash
# Local / CI smoke: Python bytecode check, LaTeX metric sync, ACL PDF build.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON_BIN:-${ROOT}/.venv311/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

echo "[smoke] compileall (core packages)…"
"$PYTHON" -m compileall -q agents plantswarm baselines ablations calibration bias utils scripts data

echo "[smoke] sync_latex_metrics…"
"$PYTHON" scripts/sync_latex_metrics.py --results-dir results --latex-dir plantswarm/latex

echo "[smoke] build_latex_pdf…"
bash scripts/build_latex_pdf.sh --latex-dir plantswarm/latex --results-dir results

if grep -qiE 'undefined references|undefined citations' plantswarm/latex/acl_latex.log 2>/dev/null; then
  echo "[smoke][WARN] acl_latex.log still reports undefined refs/citations — check .bib keys or rerun latexmk."
fi

echo "[smoke] OK"
