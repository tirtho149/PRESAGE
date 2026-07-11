#!/usr/bin/env bash
#SBATCH --job-name=presage_verify
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/verify_%A_%a.out
#SBATCH --time=1-00:00:00
# Per-crop Claude+WebSearch re-verification of uncited regional deltas.
# One array task per crop (crop = line SLURM_ARRAY_TASK_ID+1 of all_crops.txt).
# CPU-only: claude CLI does the work over the network. validate_kb writes the
# crop's registry at the end of its run and only touches verification_status==
# "unverified" deltas, so re-running a crop is naturally resumable/idempotent.
#
# Submit (non-preemptible nova via mech-ai/normal QOS):
#   sbatch --account=mech-ai --qos=normal --partition=nova \
#          --array=0-75%4 scripts/verify_array.sbatch
set -uo pipefail
cd /work/mech-ai-scratch/tirtho/PRESAGE
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PATH="/work/mech-ai-scratch/tirtho/npm-global/bin:$PATH"
export HF_TOKEN=$(grep -oE "hf_[A-Za-z0-9]+" /work/mech-ai-scratch/tirtho/CyAg/.env | head -1)

IDX=${SLURM_ARRAY_TASK_ID:?set via --array}
CROP=$(sed -n "$((IDX+1))p" all_crops.txt)
if [ -z "$CROP" ]; then echo "no crop at line $((IDX+1))"; exit 0; fi
echo "VERIFY crop='$CROP' idx=$IDX $(date)"

CROPS="$CROP" ALLOW_UNVERIFIED=1 \
    python scripts/validate_kb.py --kb-root artifacts/pathome_kb --timeout 600 --max-turns 30
rc=$?
echo "VERIFY crop='$CROP' done rc=$rc $(date)"
# rc=3 == validate_kb "some deltas still unverified" (classifier holdouts).
# Treat as non-fatal so the array continues; the babysitter reports holdouts.
exit 0
