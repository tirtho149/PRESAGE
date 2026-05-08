#!/bin/bash
#SBATCH --job-name=pathome_phase0_seed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --partition=nova
#SBATCH --chdir=/work/mech-ai-scratch/tirtho/PlantSwarm
#SBATCH --output=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_phase0_seed-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_phase0_seed-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Phase 0: Seed PathomeDB visual symptom blocks via Claude headless
# ============================================================================
# CPU-only. No GPU needed — this stage only shells out to `claude -p`.
# Requires the Claude Code CLI to be installed and logged-in on the compute
# node:
#   curl -fsSL https://claude.ai/install.sh | bash    # one-time per user
#   claude auth login                                  # one-time per user
#
# Output:
#   artifacts/pathome_seed/symptoms_seed.json
#   artifacts/pathome_seed/failed.jsonl  (per-profile errors, retry with --retry-failed)
#
# Cost / time:
#   ~484 profiles * ~7 s/call (sonnet) / 4 workers ≈ 15 min wall time.
#   Resumable: re-running picks up from where it stopped.
# ============================================================================

set -e
echo "================================"
echo "Phase 0: PathomeDB Claude seed"
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
echo "================================"

module load python
source /work/mech-ai-scratch/tirtho/PlantSwarm/.venv/bin/activate
mkdir -p logs artifacts/pathome_seed

# Confirm `claude` CLI is on PATH and authenticated.
if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' CLI not found on PATH"
  echo "Install with: curl -fsSL https://claude.ai/install.sh | bash"
  exit 1
fi
claude --version || { echo "ERROR: claude not callable"; exit 1; }

WORKERS="${PATHOME_SEED_WORKERS:-4}"
MODEL="${PATHOME_SEED_MODEL:-sonnet}"

echo "workers=$WORKERS  model=$MODEL"
python scripts/seed_pathome_with_claude.py \
  --config configs/bugwood_pathome.yaml \
  --workers "$WORKERS" \
  --model   "$MODEL"

echo
echo "Phase 0 complete: $(date)"
echo "Output: artifacts/pathome_seed/symptoms_seed.json"
