#!/bin/bash
#SBATCH --job-name=pathome_phase4_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/work/mech-ai-scratch/tirtho/PlantSwarm
#SBATCH --output=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_phase4_train-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_phase4_train-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Phase 4: Train OBSERVE twice — seeded DB and enhanced DB — for a clean
# before/after comparison.
# ============================================================================
# A100, 24h walltime. Sequential: seed-trained first, enhanced-trained second.
# Each run executes Phase A (Decision Transformer) then Phase B (GRPO) on the
# same trace JSONL — only the PathomeDB the agents read from differs.
#
# Override targets at submit time:
#   PATHOME_SEED_DB=artifacts/pathome_v1_seed     \
#   PATHOME_ENHANCED_DB=artifacts/pathome_v1_enhanced \
#   sbatch scripts/submit_pathome_phase4_train.sh
#
# Output:
#   observe/checkpoints/seed/observe_grpo_epoch_*.pt
#   observe/checkpoints/enhanced/observe_grpo_epoch_*.pt
# ============================================================================

set -e
echo "================================"
echo "Phase 4: OBSERVE training (×2)"
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
nvidia-smi || true
echo "================================"

module load python cuda/11.8
source /work/mech-ai-scratch/tirtho/PlantSwarm/.venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

CONFIG="${PATHOME_CONFIG:-configs/bugwood_pathome.yaml}"
SEED_DB="${PATHOME_SEED_DB:-artifacts/pathome_v1_seed}"
ENH_DB="${PATHOME_ENHANCED_DB:-artifacts/pathome_v1_enhanced}"
SEED_OUT="observe/checkpoints/seed"
ENH_OUT="observe/checkpoints/enhanced"

mkdir -p logs "$SEED_OUT" "$ENH_OUT"

run_one() {
  local tag="$1" db="$2" ckpt_dir="$3"
  echo
  echo "── training OBSERVE [$tag] against $db ──"
  python -c "
import sys, yaml, tempfile, subprocess, os
cfg = yaml.safe_load(open(sys.argv[1]))
cfg.setdefault('pathome', {})['load_dir'] = sys.argv[2]
cfg.setdefault('observe', {})['checkpoint_dir'] = sys.argv[3]
with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as tf:
    yaml.safe_dump(cfg, tf); patched = tf.name
ret = subprocess.call([
    'python', 'scripts/train_observe_pathome.py',
    '--config', patched, '--phase', 'both',
])
os.unlink(patched); sys.exit(ret)
" "$CONFIG" "$db" "$ckpt_dir"
}

run_one "seed"     "$SEED_DB" "$SEED_OUT"
run_one "enhanced" "$ENH_DB"  "$ENH_OUT"

echo
echo "Phase 4 complete: $(date)"
echo "  seed     ckpts:  $SEED_OUT/"
echo "  enhanced ckpts:  $ENH_OUT/"
