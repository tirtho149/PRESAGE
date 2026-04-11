#!/usr/bin/env bash
# Run the full evaluation stack (PlantSwarm → baselines → ablations → calibration →
# routing → bias → LaTeX sync). Call from Slurm or locally after cd to repo root.
# Required env: PYTHON_BIN, CONFIG_PATH, RESULTS_DIR
# Optional: SUBSET (empty = all test images), ORCHESTRATOR (default autogen_swarm),
#           ROUTING_SUBSET (default SUBSET or 500)
set -euo pipefail

: "${PYTHON_BIN:?Set PYTHON_BIN}"
: "${CONFIG_PATH:?Set CONFIG_PATH}"
: "${RESULTS_DIR:?Set RESULTS_DIR}"

ORCHESTRATOR="${ORCHESTRATOR:-autogen_swarm}"
SUBSET="${SUBSET:-}"
ROUTING_SUBSET="${ROUTING_SUBSET:-}"
if [[ -z "${ROUTING_SUBSET}" ]]; then
  if [[ -n "${SUBSET}" ]]; then
    ROUTING_SUBSET="${SUBSET}"
  else
    ROUTING_SUBSET="500"
  fi
fi

STEP_LOG_DIR="${RESULTS_DIR}/step_logs"
mkdir -p "${STEP_LOG_DIR}"

run_step() {
  local step_name="$1"
  shift
  echo
  echo ">>> [START] ${step_name}"
  echo ">>> CMD: $*"
  "$@" 2>&1 | tee "${STEP_LOG_DIR}/${step_name}.log"
  echo ">>> [DONE]  ${step_name}"
}

opt_subset() {
  if [[ -n "${SUBSET}" ]]; then
    echo --subset "${SUBSET}"
  fi
}

run_step "01_plantswarm_main" \
  "${PYTHON_BIN}" scripts/run_plantswarm.py \
  --config "${CONFIG_PATH}" \
  $(opt_subset) \
  --orchestrator "${ORCHESTRATOR}" \
  --output_dir "${RESULTS_DIR}"

run_step "02_baselines" \
  "${PYTHON_BIN}" scripts/run_baselines.py \
  --config "${CONFIG_PATH}" \
  $(opt_subset) \
  --output_dir "${RESULTS_DIR}"

run_step "03_ablations" \
  "${PYTHON_BIN}" scripts/run_ablations.py \
  --config "${CONFIG_PATH}" \
  $(opt_subset) \
  --output_dir "${RESULTS_DIR}"

run_step "04_calibration" \
  "${PYTHON_BIN}" scripts/run_calibration.py \
  --config "${CONFIG_PATH}" \
  --predictions "${RESULTS_DIR}/plantswarm_predictions.jsonl" \
  --output_dir "${RESULTS_DIR}"

run_step "05_routing_analysis" \
  "${PYTHON_BIN}" scripts/run_routing_analysis.py \
  --config "${CONFIG_PATH}" \
  --subset "${ROUTING_SUBSET}" \
  --output_dir "${RESULTS_DIR}"

run_step "06_bias_analysis" \
  "${PYTHON_BIN}" scripts/run_bias_analysis.py \
  --config "${CONFIG_PATH}" \
  --traces "${RESULTS_DIR}/traces/plantswarm_traces.jsonl" \
  --predictions "${RESULTS_DIR}/plantswarm_predictions.jsonl" \
  --output_dir "${RESULTS_DIR}"

export SUBSET
if [[ "${SKIP_LATEX_SYNC:-0}" == "1" ]]; then
  echo ">>> [SKIP] 07_sync_latex (SKIP_LATEX_SYNC=1)"
else
  run_step "07_sync_latex" \
    "${PYTHON_BIN}" scripts/sync_latex_metrics.py \
    --results-dir "${RESULTS_DIR}" \
    --subset-hint "${SUBSET:-full}"
fi

if [[ "${BUILD_LATEX_PDF:-1}" == "1" ]]; then
  if ! run_step "08_build_pdf" \
    bash scripts/build_latex_pdf.sh \
    --latex-dir plantswarm/latex \
    --main-tex acl_latex.tex \
    --results-dir "${RESULTS_DIR}"; then
    if [[ "${STRICT_PDF_BUILD:-0}" == "1" ]]; then
      echo "[FATAL] PDF build failed and STRICT_PDF_BUILD=1"
      exit 1
    fi
    echo "[WARN] PDF build failed; continuing (set STRICT_PDF_BUILD=1 to fail job)."
  fi
else
  echo ">>> [SKIP] 08_build_pdf (BUILD_LATEX_PDF=${BUILD_LATEX_PDF:-0})"
fi

echo
echo "Bundle complete. Artifacts: ${RESULTS_DIR}"
if [[ "${SKIP_LATEX_SYNC:-0}" != "1" ]]; then
  echo "LaTeX snippets: plantswarm/latex/auto_*.tex"
fi
if [[ "${BUILD_LATEX_PDF:-1}" == "1" ]]; then
  echo "Paper PDF: ${RESULTS_DIR}/paper_acl_latex.pdf"
fi
