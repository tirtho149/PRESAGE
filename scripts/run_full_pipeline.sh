#!/bin/bash
"""
scripts/run_full_pipeline.sh
=============================
Master orchestrator for complete PlantSwarm + OBSERVE pipeline.

Runs all phases in sequence:
  Phase 1: PlantSwarm on PlantVillage
  Phase 2: Baselines, Ablations, Calibration, Routing Analysis
  Phase 3: Train OBSERVE
  Phase 4: Evaluate OBSERVE on PlantWild (OOD)
  Phase 5: Sync metrics to LaTeX paper

Usage:
    bash scripts/run_full_pipeline.sh --results-dir results/plant_village_tfds --subset 10
    bash scripts/run_full_pipeline.sh --results-dir results/plant_village_tfds  # Full run
"""

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

RESULTS_DIR="${RESULTS_DIR:-results/plant_village_tfds}"
SUBSET="${SUBSET:-}"
LATEX_DIR="${LATEX_DIR:-plantswarm/latex}"
DEVICE="${DEVICE:-cuda}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_section() {
    echo -e "\n${GREEN}===================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}===================================================${NC}\n"
}

log_error() {
    echo -e "${RED}ERROR: $1${NC}"
    exit 1
}

log_warn() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

# Verify vLLM server is running
check_vllm() {
    log_section "Checking vLLM server..."
    if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        log_error "vLLM server not running on http://localhost:8000/v1"
        echo "Start it with:"
        echo "  python -m vllm.entrypoints.openai_api_server --model Qwen/Qwen2.5-VL-3B-Instruct --gpu-memory-utilization 0.8 --port 8000"
    fi
    echo "✓ vLLM server is reachable"
}

# Phase 1: PlantSwarm
phase1_plantswarm() {
    log_section "PHASE 1: Generate PlantSwarm routing traces"
    python "$REPO_ROOT/scripts/run_plantswarm.py" \
        --config "$REPO_ROOT/configs/plant_village_tfds.yaml" \
        ${SUBSET:+--subset $SUBSET}
    echo "✓ Routing traces saved to $RESULTS_DIR/traces/plantswarm_traces.jsonl"
}

# Phase 2: Baselines, Ablations, Calibration
phase2_experiments() {
    log_section "PHASE 2: Run baselines, ablations, and calibration"

    echo "Running baselines..."
    python "$REPO_ROOT/scripts/run_baselines.py" \
        --config "$REPO_ROOT/configs/plant_village_tfds.yaml" \
        ${SUBSET:+--subset $SUBSET}

    echo "Running ablations..."
    python "$REPO_ROOT/scripts/run_ablations.py" \
        --config "$REPO_ROOT/configs/plant_village_tfds.yaml" \
        ${SUBSET:+--subset $SUBSET}

    echo "Running calibration analysis..."
    python "$REPO_ROOT/scripts/run_calibration.py" \
        --config "$REPO_ROOT/configs/plant_village_tfds.yaml" \
        --predictions "$RESULTS_DIR/plantswarm_predictions.jsonl" \
        ${SUBSET:+--subset $SUBSET}

    echo "Running routing analysis..."
    python "$REPO_ROOT/scripts/run_routing_analysis.py" \
        --config "$REPO_ROOT/configs/plant_village_tfds.yaml" \
        ${SUBSET:+--subset $SUBSET}

    echo "✓ All experiments complete"
}

# Phase 3: Train OBSERVE
phase3_observe() {
    log_section "PHASE 3: Train OBSERVE on routing traces"

    if [ ! -f "$RESULTS_DIR/traces/plantswarm_traces.jsonl" ]; then
        log_error "Routing traces not found. Run Phase 1 first."
    fi

    python "$REPO_ROOT/scripts/train_observe.py" \
        --traces "$RESULTS_DIR/traces/plantswarm_traces.jsonl" \
        --output "$REPO_ROOT/observe/checkpoints/observe_final.pt" \
        --epochs 50 \
        --batch-size 8 \
        --device "$DEVICE"

    echo "✓ OBSERVE model trained and saved"
}

# Phase 4: Evaluate on PlantWild (OOD)
phase4_ood() {
    log_section "PHASE 4: Evaluate on PlantWild (OOD)"

    log_warn "PlantWild evaluation requires separate OOD dataset setup"
    echo "To evaluate on PlantWild:"
    echo "  python scripts/run_plantswarm.py --config configs/plantwild_hf.yaml"
    echo "  python scripts/evaluate_observe.py --model observe/checkpoints/observe_final.pt --traces results/plantwild/traces/plantswarm_traces.jsonl --output results/plantwild/observe_evaluation.json"
}

# Phase 5: Sync metrics to LaTeX
phase5_latex() {
    log_section "PHASE 5: Sync metrics to LaTeX paper"

    mkdir -p "$REPO_ROOT/$LATEX_DIR/auto"

    python "$REPO_ROOT/scripts/sync_latex_metrics.py" \
        --results-dir "$REPO_ROOT/$RESULTS_DIR" \
        --latex-dir "$REPO_ROOT/$LATEX_DIR"

    echo "✓ LaTeX tables synced:"
    ls -lh "$REPO_ROOT/$LATEX_DIR/auto/"auto_*.tex
}

# Main execution
main() {
    echo "PlantSwarm Full Pipeline Orchestrator"
    echo "======================================"
    echo "Results directory: $RESULTS_DIR"
    echo "LaTeX directory: $LATEX_DIR"
    echo "Device: $DEVICE"
    [ -n "$SUBSET" ] && echo "Subset: $SUBSET images"

    # Check vLLM
    check_vllm

    # Run phases
    phase1_plantswarm
    phase2_experiments
    phase3_observe
    phase4_ood
    phase5_latex

    log_section "Pipeline Complete! ✓"
    echo "Results saved to: $RESULTS_DIR"
    echo "LaTeX tables in: $REPO_ROOT/$LATEX_DIR/auto/"
    echo "Compile paper with: latexmk -pdf plantswarm/latex/acl_latex.tex"
}

main "$@"
