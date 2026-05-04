#!/bin/bash
#SBATCH --job-name=plantswarm_traces
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --chdir=/work/mech-ai-scratch/tirtho/PlantSwarm
#SBATCH --output=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/phase1_plantswarm-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/phase1_plantswarm-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Phase 1: Generate PlantSwarm Routing Traces on PlantVillage
# ============================================================================
# Time: 12-18 hours
# Output: plantswarm_metrics.json, traces/plantswarm_traces.jsonl
# ============================================================================

set -e  # Exit on error

echo "================================"
echo "Phase 1: PlantSwarm Routing Traces"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $SLURM_GPUS"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo ""

# Load modules
module load python cuda/11.8

# Activate Python environment (adjust path as needed)
source /work/mech-ai-scratch/tirtho/PlantSwarm/.venv/bin/activate

# Create logs directory
mkdir -p logs

# Run Phase 1: PlantSwarm on PlantVillage (full dataset, 10K images)
# hf_direct: runs Qwen2.5-VL-7B in-process — no separate vLLM server needed
echo "Starting PlantSwarm generation..."
python scripts/run_plantswarm.py \
  --config configs/plant_village_tfds.yaml \
  --orchestrator hf_direct

echo ""
echo "✓ Phase 1 Complete"
echo "Output:"
echo "  - results/plant_village_tfds/plantswarm_metrics.json"
echo "  - results/plant_village_tfds/traces/plantswarm_traces.jsonl"
echo "End time: $(date)"
