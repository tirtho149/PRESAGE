#!/bin/bash
#SBATCH --job-name=evaluate_observe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --chdir=/work/mech-ai/tirtho/ObservePlantSwarm
#SBATCH --output=/work/mech-ai/tirtho/ObservePlantSwarm/logs/evaluate_observe-%j.out
#SBATCH --error=/work/mech-ai/tirtho/ObservePlantSwarm/logs/evaluate_observe-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Evaluate OBSERVE Model on Routing Traces
# ============================================================================
# Time: 30-60 min
# Requirements: GPU with 20GB+ memory
# Output: observe_evaluation.json
# ============================================================================

set -e

echo "================================"
echo "Evaluate OBSERVE Model"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $SLURM_GPUS"
echo "Start time: $(date)"
echo ""

# Load modules
module load python cuda/11.8

# Activate Python environment
source /work/mech-ai/tirtho/ObservePlantSwarm/.venv/bin/activate

# Create logs directory
mkdir -p logs

# Check if model exists
if [ ! -f "observe/checkpoints/observe_final.pt" ]; then
    echo "ERROR: Model not found at observe/checkpoints/observe_final.pt"
    echo "Train OBSERVE first: sbatch scripts/submit_phase3_observe_training.sh"
    exit 1
fi

# Evaluate on PlantVillage (ID)
echo "Evaluating OBSERVE on PlantVillage (ID)..."
python scripts/evaluate_observe.py \
  --model observe/checkpoints/observe_final.pt \
  --traces results/plant_village_tfds/traces/plantswarm_traces.jsonl \
  --output results/plant_village_tfds/observe_evaluation.json

echo ""
echo "✓ Evaluation Complete"
echo "Output: results/plant_village_tfds/observe_evaluation.json"
echo "End time: $(date)"
