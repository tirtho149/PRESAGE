#!/bin/bash
#SBATCH --job-name=plantswarm_experiments
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --chdir=/work/mech-ai/tirtho/ObservePlantSwarm
#SBATCH --output=/work/mech-ai/tirtho/ObservePlantSwarm/logs/phase2_experiments-%j.out
#SBATCH --error=/work/mech-ai/tirtho/ObservePlantSwarm/logs/phase2_experiments-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Phase 2: Baselines, Ablations, Calibration, Routing Analysis
# ============================================================================
# Time: 2-3 hours (can run in parallel or sequentially)
# Output: baseline_results.json, ablation_metrics_*.json, calibration_report.json,
#         routing_analysis.json
# ============================================================================

set -e

echo "================================"
echo "Phase 2: Experimental Comparisons"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Load modules
module load python cuda/11.8

# Activate Python environment
source /work/mech-ai/tirtho/ObservePlantSwarm/.venv/bin/activate

# Create logs directory
mkdir -p logs

# ============================================================================
# 2a: Baselines (30-45 min)
# ============================================================================
echo "[2a] Running baseline comparisons..."
python scripts/run_baselines.py --config configs/plant_village_tfds.yaml
echo "✓ Baselines complete (baseline_results.json)"

# ============================================================================
# 2b: Ablations (45 min)
# ============================================================================
echo "[2b] Running ablation study..."
python scripts/run_ablations.py --config configs/plant_village_tfds.yaml
echo "✓ Ablations complete (ablation_metrics_*.json)"

# ============================================================================
# 2c: Calibration Analysis (30-45 min)
# ============================================================================
echo "[2c] Running calibration analysis..."
python scripts/run_calibration.py \
  --config configs/plant_village_tfds.yaml \
  --predictions results/plant_village_tfds/plantswarm_predictions.jsonl
echo "✓ Calibration complete (calibration_report.json)"

# ============================================================================
# 2d: Routing Analysis (15-30 min)
# ============================================================================
echo "[2d] Running routing analysis (P1-P4)..."
python scripts/run_routing_analysis.py --config configs/plant_village_tfds.yaml
echo "✓ Routing analysis complete (routing_analysis.json)"

echo ""
echo "✓ Phase 2 Complete"
echo "Output files:"
echo "  - results/plant_village_tfds/baseline_results.json"
echo "  - results/plant_village_tfds/ablation_metrics_*.json"
echo "  - results/plant_village_tfds/calibration_report.json"
echo "  - results/plant_village_tfds/routing_analysis.json"
echo "End time: $(date)"
