#!/usr/bin/env bash
# Build plantswarm/latex/acl_latex.tex into PDF.
# - Uses latexmk if available, else pdflatex/bibtex fallback.
# - Copies PDF to results dir when --results-dir is provided.
set -euo pipefail

LATEX_DIR="plantswarm/latex"
MAIN_TEX="acl_latex.tex"
RESULTS_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --latex-dir)
      LATEX_DIR="$2"
      shift 2
      ;;
    --main-tex)
      MAIN_TEX="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] Unknown arg: $1"
      echo "Usage: $0 [--latex-dir plantswarm/latex] [--main-tex acl_latex.tex] [--results-dir results/<run>]"
      exit 2
      ;;
  esac
done

if [[ ! -d "$LATEX_DIR" ]]; then
  echo "[ERROR] LaTeX dir not found: $LATEX_DIR"
  exit 1
fi

if [[ ! -f "$LATEX_DIR/$MAIN_TEX" ]]; then
  echo "[ERROR] Main tex not found: $LATEX_DIR/$MAIN_TEX"
  exit 1
fi

# TinyTeX / sandbox: stale ls-R or missing PATH can hide packages (e.g. caption.sty).
# Always prepend the distro tree to TEXINPUTS when we can locate texmf-dist.
TEXMFDIST=""
if command -v kpsewhich >/dev/null 2>&1; then
  TEXMFDIST="$(kpsewhich --var-value=TEXMFDIST 2>/dev/null || true)"
fi
if [[ -z "${TEXMFDIST}" || ! -d "${TEXMFDIST}" ]]; then
  for cand in \
    "${HOME}/Library/TinyTeX/texmf-dist" \
    "${HOME}/TinyTeX/texmf-dist" \
    "/usr/local/texlive/2025/texmf-dist" \
    "/usr/local/texlive/2024/texmf-dist"; do
    if [[ -d "$cand" ]]; then
      TEXMFDIST="$cand"
      break
    fi
  done
fi
if [[ -n "${TEXMFDIST}" && -d "${TEXMFDIST}" ]]; then
  export TEXINPUTS=".:${TEXMFDIST}/tex//:${TEXINPUTS:-}"
  export BIBINPUTS=".:${TEXMFDIST}/bibtex/bib//:${BIBINPUTS:-}"
  export BSTINPUTS=".:${TEXMFDIST}/bibtex/bst//:${BSTINPUTS:-}"
fi

if command -v latexmk >/dev/null 2>&1; then
  (
    cd "$LATEX_DIR"
    # Do not mask failures: CI/smoke tests rely on exit status and PDF output.
    latexmk -pdf -g -interaction=nonstopmode "$MAIN_TEX"
  )
else
  if ! command -v pdflatex >/dev/null 2>&1; then
    echo "[ERROR] Neither latexmk nor pdflatex is available."
    exit 1
  fi
  (
    cd "$LATEX_DIR"
    base="${MAIN_TEX%.tex}"
    pdflatex -interaction=nonstopmode "$MAIN_TEX"
    if [[ -f "${base}.aux" ]] && grep -qE '\\citation|\\bibdata' "${base}.aux" 2>/dev/null; then
      if command -v bibtex >/dev/null 2>&1; then
        bibtex "$base"
      fi
    fi
    pdflatex -interaction=nonstopmode "$MAIN_TEX"
    pdflatex -interaction=nonstopmode "$MAIN_TEX"
  )
fi

PDF_PATH="$LATEX_DIR/${MAIN_TEX%.tex}.pdf"
if [[ -f "$PDF_PATH" ]]; then
  echo "Built PDF: $PDF_PATH"
  if [[ -n "$RESULTS_DIR" ]]; then
    mkdir -p "$RESULTS_DIR"
    cp "$PDF_PATH" "$RESULTS_DIR/paper_${MAIN_TEX%.tex}.pdf"
    echo "Copied PDF: $RESULTS_DIR/paper_${MAIN_TEX%.tex}.pdf"
  fi
else
  echo "[ERROR] PDF build did not produce: $PDF_PATH"
  exit 1
fi

