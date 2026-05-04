#!/bin/bash
#SBATCH --job-name=setup_plantwild
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=nova
#SBATCH --chdir=/work/mech-ai-scratch/tirtho/PlantSwarm
#SBATCH --output=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/setup_plantwild-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/setup_plantwild-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Setup: Download PlantWild Dataset from HuggingFace
# ============================================================================
# Time: 2-4 hours (depending on internet bandwidth)
# Storage: ~50GB required
# Output: data/PlantWild/ (cloned dataset directory)
# ============================================================================

set -e

echo "================================"
echo "Setup: PlantWild Dataset"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Load modules
module load python

# Create data directory
mkdir -p data logs

cd data

# ============================================================================
# Option 1: Clone using HuggingFace CLI (Recommended)
# ============================================================================
echo "Installing HuggingFace CLI..."
curl -LsSf https://hf.co/cli/install.sh | bash

echo ""
echo "Downloading PlantWild dataset..."
hf download uqtwei2/PlantWild --repo-type=dataset --local-dir ./PlantWild

# ============================================================================
# Alternative Option 2: Clone using git-xet (if HF CLI fails)
# ============================================================================
# Uncomment below if Option 1 fails
# echo "Installing git-xet..."
# brew install git-xet
# git xet install
#
# echo "Cloning PlantWild with git-xet..."
# cd ..
# git clone https://huggingface.co/datasets/uqtwei2/PlantWild data/PlantWild

# ============================================================================
# Verify download
# ============================================================================
echo ""
echo "Verifying dataset..."
if [ -d "PlantWild" ]; then
    echo "✓ PlantWild dataset downloaded successfully"
    echo "Location: $(pwd)/PlantWild"
    echo "Size: $(du -sh PlantWild | awk '{print $1}')"
    echo "Files: $(find PlantWild -type f | wc -l)"
else
    echo "ERROR: Dataset not found at expected location"
    exit 1
fi

echo ""
echo "✓ Setup Complete"
echo "Dataset ready at: data/PlantWild/"
echo "End time: $(date)"
echo ""
echo "Next: Run Phase 1-5 pipeline"
echo "  bash scripts/submit_all_phases.sh"
