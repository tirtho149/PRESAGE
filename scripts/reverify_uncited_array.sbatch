#!/usr/bin/env bash
#SBATCH --job-name=presage_reverify
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/reverify_%A_%a.out
#SBATCH --time=1-00:00:00
# RETRY pass over regional deltas that have a verdict but NO web citation
# (provisional / novel_plausible / unverified). One array task per crop
# (crop = line SLURM_ARRAY_TASK_ID+1 of reverify_crops.txt).
#
# For each such delta the Claude+WebSearch verifier runs a SECOND time:
#   - now finds support -> verified / weakly_supported (web-cited)
#   - web contradicts    -> dropped
#   - still no support    -> relabelled verification_status="field_observation"
# CPU-only; claude CLI does the work over the network. Idempotent: a delta
# that already earned a web citation is frozen context and never re-judged.
#
# Submit (non-preemptible nova via mech-ai/normal QOS):
#   sbatch --account=mech-ai --qos=normal --partition=nova \
#          --array=0-70%4 scripts/reverify_uncited_array.sbatch
set -uo pipefail
cd /work/mech-ai-scratch/tirtho/PRESAGE
source /work/mech-ai-scratch/tirtho/.venv/bin/activate
export PATH="/work/mech-ai-scratch/tirtho/npm-global/bin:$PATH"
export HF_TOKEN=$(grep -oE "hf_[A-Za-z0-9]+" /work/mech-ai-scratch/tirtho/CyAg/.env | head -1)

IDX=${SLURM_ARRAY_TASK_ID:?set via --array}
CROP=$(sed -n "$((IDX+1))p" reverify_crops.txt)
if [ -z "$CROP" ]; then echo "no crop at line $((IDX+1))"; exit 0; fi
echo "REVERIFY crop='$CROP' idx=$IDX $(date)"

CROPS="$CROP" REVERIFY_UNCITED=1 ALLOW_UNVERIFIED=1 \
    python scripts/validate_kb.py --kb-root artifacts/pathome_kb \
    --reverify-uncited --timeout 600 --max-turns 30
rc=$?
echo "REVERIFY crop='$CROP' done rc=$rc $(date)"
exit 0
