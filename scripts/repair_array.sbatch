#!/usr/bin/env bash
#SBATCH --job-name=presage_repair
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/repair_%A_%a.out
#SBATCH --time=1-00:00:00
# Per-crop quote/link repair. One array task per crop (crop = line
# SLURM_ARRAY_TASK_ID+1 of all_crops.txt). Each task:
#   1. requote  - Claude reads the CACHED page text and returns a VERBATIM
#                 supporting sentence from the SAME page (fast, no web). Misses
#                 (page doesn't support the claim) are escalated.
#   2. websearch- Claude+WebSearch finds an ALTERNATIVE url + verbatim quote for
#                 the direct-no-text URLs and the escalated requote-misses.
# Writes artifacts/repair_patches/<Crop>.json (per-crop file = parallel-safe).
# Central verify + apply run afterward on a node with Playwright. Idempotent:
# a node already patched (node_key present) is skipped, so re-submitting resumes.
# CPU-only; claude CLI does the work over the network (compute nodes have it).
#
# Submit (non-preemptible nova):
#   sbatch --account=mech-ai --qos=normal --partition=nova \
#          --array=0-75%20 scripts/repair_array.sbatch
set -e
cd /work/mech-ai-scratch/tirtho/PRESAGE
source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PATH="/work/mech-ai-scratch/tirtho/npm-global/bin:$PATH"

CROP=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" all_crops.txt)
echo "[$(date)] task $SLURM_ARRAY_TASK_ID -> crop='$CROP'"
[ -z "$CROP" ] && { echo "no crop for index, exiting"; exit 0; }

P=scripts/repair_kb_quotes.py
echo "=== REQUOTE $CROP ==="
python3 $P run --crop "$CROP" --mode requote  --workers 6
echo "=== WEBSEARCH $CROP ==="
python3 $P run --crop "$CROP" --mode websearch --workers 4
echo "[$(date)] DONE crop='$CROP'"
