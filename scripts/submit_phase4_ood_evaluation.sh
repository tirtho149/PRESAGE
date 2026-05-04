#!/bin/bash
#SBATCH --job-name=observe_ood_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --chdir=/work/mech-ai-scratch/tirtho/PlantSwarm
#SBATCH --output=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/phase4_ood_eval-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/phase4_ood_eval-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Phase 4: OOD Evaluation (OBSERVE on PlantWild)
# ============================================================================
# Time: 2-3 hours
# Requirements: GPU, trained OBSERVE model from Phase 3
# Output: observe_evaluation.json (OOD metrics)
# ============================================================================

set -e

echo "================================"
echo "Phase 4: OOD Evaluation (PlantWild)"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Load modules
module load python cuda/11.8

# Activate Python environment
source /work/mech-ai-scratch/tirtho/PlantSwarm/.venv/bin/activate

# Create logs directory
mkdir -p logs results/plantwild/traces

# Verify trained model exists
if [ ! -f "observe/checkpoints/observe_final.pt" ]; then
    echo "ERROR: Trained OBSERVE model not found!"
    echo "Run Phase 3 first: sbatch scripts/submit_phase3_observe_training.sh"
    exit 1
fi

# ============================================================================
# 4a: Generate PlantWild traces (PlantSwarm on OOD data)
# ============================================================================
echo "[4a] Generating PlantWild routing traces..."
python scripts/run_plantswarm.py --config configs/plantwild_hf.yaml
echo "✓ PlantWild traces complete"

# ============================================================================
# 4b: Evaluate OBSERVE on PlantWild (OOD)
# ============================================================================
echo "[4b] Evaluating OBSERVE on PlantWild (OOD)..."
python scripts/evaluate_observe.py \
  --model observe/checkpoints/observe_final.pt \
  --traces results/plantwild/traces/plantswarm_traces.jsonl \
  --output results/plantwild/observe_evaluation.json \
  --device cuda

echo ""
echo "✓ Phase 4 Complete"
echo "Output files:"
echo "  - results/plantwild/plantswarm_metrics.json (PlantWild PlantSwarm results)"
echo "  - results/plantwild/observe_evaluation.json (OOD metrics)"
echo "End time: $(date)"
echo ""
echo "Expected: OBSERVE OOD ECE ~0.16 (52% improvement over PlantSwarm's 0.33)"
