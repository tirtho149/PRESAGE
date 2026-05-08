#!/bin/bash
#SBATCH --job-name=pathome_phase1_build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=nova
#SBATCH --chdir=/work/mech-ai-scratch/tirtho/PlantSwarm
#SBATCH --output=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_phase1_build-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_phase1_build-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Phase 1: Build PathomeDB v1 (seeded) from BugWood_Diseases_usable.csv
# ============================================================================
# CPU + outbound HTTPS. The first run downloads ~11k Bugwood image thumbnails
# into .bugwood_cache/ (≈600 MB on disk); subsequent builds are cached.
#
# Inputs:
#   configs/bugwood_pathome.yaml      (csv_path, per_class, trace_split, …)
#   artifacts/pathome_seed/symptoms_seed.json   (Phase 0 output, optional)
#
# Output:
#   artifacts/pathome_v1_seed/
#     ├── symptoms.json    (Claude visual blocks + auto-derived state/AEZ counts + ref_ids)
#     ├── refs/            (held-out reference image registry, lazy CLIP index)
#     ├── version.txt
#     └── build_summary.json
#
# Re-run cheaply once seed_pathome_with_claude.py emits a new file.
# ============================================================================

set -e
echo "================================"
echo "Phase 1: PathomeDB build"
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
echo "================================"

module load python
source /work/mech-ai-scratch/tirtho/PlantSwarm/.venv/bin/activate
mkdir -p logs artifacts/pathome_v1_seed .bugwood_cache

CONFIG="${PATHOME_CONFIG:-configs/bugwood_pathome.yaml}"
SEED="${PATHOME_SEED_FILE:-artifacts/pathome_seed/symptoms_seed.json}"
OUT="${PATHOME_OUT_DIR:-artifacts/pathome_v1_seed}"

# Wire the seed JSON into pathome.symptoms_path before invoking the builder.
# We use a small Python override so we don't have to maintain a parallel YAML.
echo "config=$CONFIG  seed=$SEED  out=$OUT"
python -c "
import sys, yaml, tempfile, subprocess, os
cfg_path = sys.argv[1]; seed = sys.argv[2]; out = sys.argv[3]
cfg = yaml.safe_load(open(cfg_path))
if os.path.isfile(seed):
    cfg.setdefault('pathome', {})['symptoms_path'] = seed
    print(f'  Using Claude-seeded symptoms: {seed}')
else:
    print(f'  [warn] no seed file at {seed} — building empty visual blocks')
cfg.setdefault('pathome', {})['out_dir'] = out
with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as tf:
    yaml.safe_dump(cfg, tf); patched = tf.name
ret = subprocess.call(['python', 'scripts/build_pathome.py', '--config', patched])
os.unlink(patched)
sys.exit(ret)
" "$CONFIG" "$SEED" "$OUT"

echo
echo "Phase 1 complete: $(date)"
echo "Output: $OUT/"
