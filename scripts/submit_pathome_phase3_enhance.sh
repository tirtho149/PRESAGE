#!/bin/bash
#SBATCH --job-name=pathome_phase3_enhance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=nova
#SBATCH --chdir=/work/mech-ai-scratch/tirtho/PlantSwarm
#SBATCH --output=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_phase3_enhance-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_phase3_enhance-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Phase 3: Enhance the seeded PathomeDB with PlantSwarm trace observations
# ============================================================================
# CPU-only. Mines results/bugwood_seed/traces/plantswarm_traces.jsonl into
# per-(crop, disease) SwarmObservations and writes a new DB version.
#
# Inputs:
#   $SEED_DB     (default artifacts/pathome_v1_seed/)
#   $TRACES      (default results/bugwood_seed/traces/plantswarm_traces.jsonl)
#
# Output:
#   artifacts/pathome_v1_enhanced/
#     ├── symptoms.json              (Claude visual + auto geo + swarm_observations)
#     ├── refs/                       (copied through)
#     ├── version.txt                 ("v2.0+swarm")
#     └── enhancement_summary.json
# ============================================================================

set -e
echo "================================"
echo "Phase 3: Enhance PathomeDB"
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
echo "================================"

module load python
source /work/mech-ai-scratch/tirtho/PlantSwarm/.venv/bin/activate
mkdir -p logs

SEED_DB="${PATHOME_SEED_DB:-artifacts/pathome_v1_seed}"
TRACES="${PATHOME_TRACES:-results/bugwood_seed/traces/plantswarm_traces.jsonl}"
OUT_DIR="${PATHOME_OUT_DIR:-artifacts/pathome_v1_enhanced}"

if [ ! -d "$SEED_DB" ]; then
  echo "ERROR: seed DB not found at $SEED_DB — run Phase 1 first"
  exit 1
fi
if [ ! -f "$TRACES" ]; then
  echo "ERROR: traces not found at $TRACES — run Phase 2 first"
  exit 1
fi

python scripts/enhance_pathome_from_traces.py \
  --seed-db "$SEED_DB" \
  --traces  "$TRACES" \
  --out     "$OUT_DIR"

echo
echo "Phase 3 complete: $(date)"
echo "Output: $OUT_DIR/"
