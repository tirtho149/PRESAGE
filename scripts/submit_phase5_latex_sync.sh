#!/bin/bash
#SBATCH --job-name=latex_sync
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=nova
#SBATCH --chdir=/work/mech-ai/tirtho/ObservePlantSwarm
#SBATCH --output=/work/mech-ai/tirtho/ObservePlantSwarm/logs/phase5_latex_sync-%j.out
#SBATCH --error=/work/mech-ai/tirtho/ObservePlantSwarm/logs/phase5_latex_sync-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Phase 5: Sync Metrics to LaTeX Paper
# ============================================================================
# Time: <1 minute
# Requirements: CPU only (no GPU needed)
# Output: plantswarm/latex/auto_*.tex (6 auto-generated table fragments)
# ============================================================================

set -e

echo "================================"
echo "Phase 5: LaTeX Sync"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Load modules
module load python

# Activate Python environment
source /work/mech-ai/tirtho/ObservePlantSwarm/.venv/bin/activate

# Create logs directory
mkdir -p logs plantswarm/latex/auto

echo "Collecting metrics from all experiments..."
python scripts/collect_metrics.py \
  --results-dir results/plant_village_tfds \
  --output results/unified_metrics.json

echo "Syncing metrics to LaTeX paper..."
python scripts/sync_latex_metrics.py \
  --results-dir results/plant_village_tfds \
  --latex-dir plantswarm/latex

echo ""
echo "✓ Phase 5 Complete"
echo "Generated LaTeX table fragments:"
ls -lh plantswarm/latex/auto/auto_*.tex

echo ""
echo "Next: Compile paper with metrics"
echo "  cd plantswarm/latex"
echo "  latexmk -pdf acl_latex.tex"
echo ""
echo "End time: $(date)"
