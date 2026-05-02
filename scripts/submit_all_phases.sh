#!/bin/bash
# ============================================================================
# submit_all_phases.sh
# ============================================================================
# Master submission script: Queue all 5 phases with proper dependencies.
# Runs sequentially: Phase 1 → 2 → 3 → 4 → 5
#
# Usage:
#   bash scripts/submit_all_phases.sh
#
# Output:
#   - Job IDs for all 5 phases printed to console
#   - All jobs queued with automatic dependencies
#   - Monitor with: squeue -u $USER
# ============================================================================

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   Submitting Full PlantSwarm + OBSERVE Pipeline to Nova      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "This will queue 5 phases with automatic dependencies:"
echo "  Phase 1: PlantSwarm (12-18h)  → generates routing traces"
echo "  Phase 2: Experiments (2-3h)    → baselines, ablations, calibration"
echo "  Phase 3: OBSERVE Training (4-6h) → fine-tune on traces"
echo "  Phase 4: OOD Evaluation (2-3h) → PlantWild evaluation"
echo "  Phase 5: LaTeX Sync (<1min)    → auto-fill paper tables"
echo ""
echo "Total time: ~25-35 hours (depending on GPU)"
echo ""

# Verify scripts exist
if [ ! -f "scripts/submit_phase1_plantswarm.sh" ]; then
    echo "ERROR: SLURM scripts not found in scripts/ directory"
    exit 1
fi

# Make scripts executable
chmod +x scripts/submit_*.sh

# ============================================================================
# Phase 1: PlantSwarm
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Submitting Phase 1: PlantSwarm Routing Traces..."
JOB1=$(sbatch scripts/submit_phase1_plantswarm.sh 2>&1 | grep "Submitted batch job" | awk '{print $NF}')
echo "✓ Phase 1 submitted: Job ID $JOB1"
echo "  Time: 12-18 hours"
echo "  Output: plantswarm_metrics.json, traces/plantswarm_traces.jsonl"
echo ""

# ============================================================================
# Phase 2: Experiments
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Submitting Phase 2: Baselines, Ablations, Calibration..."
JOB2=$(sbatch --dependency=afterok:$JOB1 scripts/submit_phase2_experiments.sh 2>&1 | grep "Submitted batch job" | awk '{print $NF}')
echo "✓ Phase 2 submitted: Job ID $JOB2"
echo "  Depends on: Phase 1 ($JOB1)"
echo "  Time: 2-3 hours"
echo "  Output: baseline_results.json, ablation_metrics_*.json, etc."
echo ""

# ============================================================================
# Phase 3: OBSERVE Training
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Submitting Phase 3: OBSERVE Training..."
JOB3=$(sbatch --dependency=afterok:$JOB2 scripts/submit_phase3_observe_training.sh 2>&1 | grep "Submitted batch job" | awk '{print $NF}')
echo "✓ Phase 3 submitted: Job ID $JOB3"
echo "  Depends on: Phase 2 ($JOB2)"
echo "  Time: 4-6 hours"
echo "  Output: observe_final.pt, training_history.json"
echo ""

# ============================================================================
# Phase 4: OOD Evaluation
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Submitting Phase 4: OOD Evaluation (PlantWild)..."
JOB4=$(sbatch --dependency=afterok:$JOB3 scripts/submit_phase4_ood_evaluation.sh 2>&1 | grep "Submitted batch job" | awk '{print $NF}')
echo "✓ Phase 4 submitted: Job ID $JOB4"
echo "  Depends on: Phase 3 ($JOB3)"
echo "  Time: 2-3 hours"
echo "  Output: observe_evaluation.json (OOD metrics)"
echo ""

# ============================================================================
# Phase 5: LaTeX Sync
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Submitting Phase 5: LaTeX Sync..."
JOB5=$(sbatch --dependency=afterok:$JOB3 scripts/submit_phase5_latex_sync.sh 2>&1 | grep "Submitted batch job" | awk '{print $NF}')
echo "✓ Phase 5 submitted: Job ID $JOB5"
echo "  Depends on: Phase 3 ($JOB3)"
echo "  Time: <1 minute"
echo "  Output: plantswarm/latex/auto_*.tex (paper tables)"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    Pipeline Submitted ✓                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job Chain:"
echo "  Phase 1 [$JOB1] → Phase 2 [$JOB2] → Phase 3 [$JOB3] → Phase 5 [$JOB5]"
echo "                                    → Phase 4 [$JOB4] (optional)"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER              # See all queued jobs"
echo "  squeue -j $JOB1                # Check Phase 1 status"
echo "  tail -f logs/phase1*.out       # Watch Phase 1 progress"
echo ""
echo "When all phases complete:"
echo "  git add results/ observe/checkpoints/ plantswarm/latex/auto/"
echo "  git commit -m 'Pipeline complete'"
echo "  git push origin main"
echo ""
echo "Compile paper with synced metrics:"
echo "  cd plantswarm/latex && latexmk -pdf acl_latex.tex"
echo ""
echo "Submitted at: $(date)"
