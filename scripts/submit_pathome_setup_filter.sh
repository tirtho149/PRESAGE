#!/bin/bash
#SBATCH --job-name=pathome_setup_filter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --partition=nova
#SBATCH --output=logs/pathome_setup_filter-%j.out
#SBATCH --error=logs/pathome_setup_filter-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

# Portable paths: PATHOME_REPO=/path/to/repo sbatch [--mail-user=...] this.sh
PATHOME_REPO="${PATHOME_REPO:-$(pwd)}"
cd "$PATHOME_REPO"

# ============================================================================
# Setup (one-time): filter the Bugwood IPMNet CSV into the usable subset
# ============================================================================
# Reads:   BugWood_Diseases.csv         (~19,749 rows, raw IPMNet export)
# Writes:  BugWood_Diseases_usable.csv  (~11,513 rows, 484 classes)
#          bugwood_classes_report.tsv   (per-class candidate counts)
#
# CPU-only, ~30 s. Downstream phases all read from the filtered CSV.
# Re-run after pulling a new IPMNet export, or when changing the threshold.
#
# Override at submit time:
#   PATHOME_THRESHOLD=15 sbatch scripts/submit_pathome_setup_filter.sh
#   (15 → 263 classes; 10 → 484; 5 → 982. See bugwood_classes_report.tsv)
# ============================================================================

set -e
echo "================================"
echo "Setup: filter Bugwood CSV"
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
echo "================================"

module load python

# Resolve the venv. Default: $PATHOME_REPO/.venv (in-repo). Override with
# PATHOME_VENV=/path/to/venv (e.g. one level above the repo, shared
# across projects). Falls back to ../.venv if neither exists.
VENV="${PATHOME_VENV:-$PATHOME_REPO/.venv}"
if [ ! -f "$VENV/bin/activate" ]; then
  if [ -f "$(dirname "$PATHOME_REPO")/.venv/bin/activate" ]; then
    VENV="$(dirname "$PATHOME_REPO")/.venv"
  else
    echo "ERROR: no venv found. Tried:"
    echo "  $PATHOME_VENV (PATHOME_VENV)"
    echo "  $PATHOME_REPO/.venv"
    echo "  $(dirname "$PATHOME_REPO")/.venv"
    echo "Set PATHOME_VENV=/path/to/venv and re-sbatch."
    exit 2
  fi
fi
echo "venv: $VENV"
source "$VENV/bin/activate"
mkdir -p logs

THRESHOLD="${PATHOME_THRESHOLD:-10}"
INPUT="${PATHOME_RAW_CSV:-BugWood_Diseases.csv}"
OUTPUT="${PATHOME_USABLE_CSV:-BugWood_Diseases_usable.csv}"
REPORT="${PATHOME_CLASS_REPORT:-bugwood_classes_report.tsv}"

if [ ! -f "$INPUT" ]; then
  echo "ERROR: input CSV not found at $INPUT"
  echo "Pull from IPMNet (https://www.bugwood.org/ipmnet) and place at the repo root,"
  echo "or set PATHOME_RAW_CSV to its path."
  exit 1
fi

echo "input:     $INPUT"
echo "threshold: $THRESHOLD rows/class"
echo "output:    $OUTPUT"
echo "report:    $REPORT"
echo
python scripts/filter_bugwood_csv.py \
  --input     "$INPUT" \
  --output    "$OUTPUT" \
  --threshold "$THRESHOLD" \
  --report    "$REPORT"

echo
echo "Setup complete: $(date)"
echo "next: sbatch scripts/submit_pathome_phase0_seed.sh"
