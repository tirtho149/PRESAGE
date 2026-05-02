#!/bin/bash
#SBATCH --job-name=observe_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --chdir=/work/mech-ai/tirtho/ObservePlantSwarm
#SBATCH --output=/work/mech-ai/tirtho/ObservePlantSwarm/logs/phase3_observe_training-%j.out
#SBATCH --error=/work/mech-ai/tirtho/ObservePlantSwarm/logs/phase3_observe_training-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Phase 3: Train OBSERVE (Vision-Language-Action Model)
# ============================================================================
# Time: 4-6 hours (A100) on 8,000-10,000 routing traces
# Requirements: GPU with 40GB+ memory recommended
# Output: observe/checkpoints/observe_final.pt, training_history.json
# ============================================================================

set -e

echo "================================"
echo "Phase 3: OBSERVE Training"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $SLURM_GPUS"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo ""

# Load modules
module load python cuda/11.8

# Activate Python environment
source /work/mech-ai/tirtho/ObservePlantSwarm/.venv/bin/activate

# Create logs and checkpoints directories
mkdir -p logs observe/checkpoints

# Verify routing traces exist
if [ ! -f "results/plant_village_tfds/traces/plantswarm_traces.jsonl" ]; then
    echo "ERROR: Routing traces not found!"
    echo "Run Phase 1 first: sbatch scripts/submit_phase1_plantswarm.sh"
    exit 1
fi

echo "Training OBSERVE model on routing traces..."
echo "Traces file: results/plant_village_tfds/traces/plantswarm_traces.jsonl"
echo ""

# ============================================================================
# Train OBSERVE
# ============================================================================
python scripts/train_observe.py \
  --traces results/plant_village_tfds/traces/plantswarm_traces.jsonl \
  --output observe/checkpoints/observe_final.pt \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --device cuda \
  --seed 42

echo ""
echo "✓ Phase 3 Complete"
echo "Output files:"
echo "  - observe/checkpoints/observe_final.pt (model weights)"
echo "  - observe/checkpoints/training_history.json (loss curves)"
echo "End time: $(date)"
echo ""
echo "Next: Run Phase 4 (OOD evaluation on PlantWild) or Phase 5 (LaTeX sync)"
