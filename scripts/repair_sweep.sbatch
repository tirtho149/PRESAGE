#!/usr/bin/env bash
#SBATCH --job-name=presage_sweep
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --time=12:00:00
# Websearch-only sweep for escalations the main array's buggy escalation-gather
# skipped (migrated pilot crops had verified=False, which the old code required
# to be None). Fixed code gathers on (escalate & !new_quote & !_ws_done).
# One task per crop in sweep_crops.txt. Idempotent.
#   sbatch --account=mech-ai --qos=normal --partition=nova \
#          --array=0-6 scripts/repair_sweep.sbatch
set -e
cd /work/mech-ai-scratch/tirtho/PRESAGE
source /work/mech-ai-scratch/tirtho/.venv/bin/activate 2>/dev/null || true
export PATH="/work/mech-ai-scratch/tirtho/npm-global/bin:$PATH"
CROP=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" sweep_crops.txt)
[ -z "$CROP" ] && { echo "no crop"; exit 0; }
echo "[$(date)] SWEEP websearch crop='$CROP'"
python3 scripts/repair_kb_quotes.py run --crop "$CROP" --mode websearch --workers 4
echo "[$(date)] DONE crop='$CROP'"
