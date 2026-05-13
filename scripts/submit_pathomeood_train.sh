#!/bin/bash
# scripts/submit_pathomeood_train.sh
# ============================================================================
# SLURM submitter for ONE BioCAP variant.
#
# Env vars:
#   VARIANT          required, e.g. "T04" (see scripts/pathomeood_variants.sh)
#   CROP             default "Tomato"
#   NPROC_PER_NODE   default 1 (single GPU; bump for multi-GPU on Nova)
#   BATCH_SIZE       default 256 (small for ~600-image crops)
#
# Submit with:
#   VARIANT=T04 sbatch scripts/submit_pathomeood_train.sh
# ============================================================================
#SBATCH --job-name=pathomeood-train
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/biocap_%x_%j.out

set -euo pipefail

: "${VARIANT:?must set VARIANT env var (e.g., T04). See scripts/pathomeood_variants.sh}"
: "${CROP:=Tomato}"
: "${NPROC_PER_NODE:=1}"
: "${BATCH_SIZE:=256}"

REPO_ROOT="${PATHOME_REPO:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

echo "==============================================="
echo " pathomeood_train  variant=$VARIANT  crop=$CROP"
echo "==============================================="
date
echo "  host:       $(hostname)"
echo "  cwd:        $(pwd)"
echo "  GPU(s):"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "  (no nvidia-smi)"
echo

python scripts/train_pathomeood.py \
    --variant        "$VARIANT" \
    --crop           "$CROP" \
    --batch-size     "$BATCH_SIZE" \
    --nproc-per-node "$NPROC_PER_NODE"

echo "=== done at $(date) ==="
