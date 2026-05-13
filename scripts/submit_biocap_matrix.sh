#!/bin/bash
# scripts/submit_biocap_matrix.sh
# ============================================================================
# Submit every variant in scripts/biocap_variants.sh as a separate SLURM
# job. Each variant produces shards under data/wds_shards/<crop>_<strategy>/
# (built first by this script if missing) and a checkpoint under
# train_and_eval/checkpoints/<VARIANT>/.
#
# Knobs:
#   CROP                     default Tomato
#   PATHOME_SKIP_CAPTIONS    set =1 to skip caption + shard build
#   PATHOME_DRY_RUN          set =1 to echo sbatch commands without submit
#   PATHOME_WAIT             set =1 to use sbatch --wait (sequential, foreground)
#
# Usage:
#   bash scripts/submit_biocap_matrix.sh                  # background sbatch
#   PATHOME_WAIT=1 bash scripts/submit_biocap_matrix.sh   # block per job
# ============================================================================
set -euo pipefail

REPO_ROOT="${PATHOME_REPO:-$(pwd)}"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source scripts/biocap_variants.sh

CROP="${CROP:-Tomato}"
WAIT_FLAG=""
if [ "${PATHOME_WAIT:-0}" = "1" ]; then
    WAIT_FLAG="--wait"
fi

mkdir -p logs data/bugwood_captions data/wds_shards

# ---- 1. Build captions + shards once per unique strategy --------------------
declare -A SEEN_STRATEGY=()
if [ "${PATHOME_SKIP_CAPTIONS:-0}" != "1" ]; then
  for v in "${BIOCAP_VARIANTS[@]}"; do
    biocap_parse_variant "$v"
    if [ -n "${SEEN_STRATEGY[$STRATEGY]:-}" ]; then continue; fi
    SEEN_STRATEGY[$STRATEGY]=1
    capt_out="data/bugwood_captions/${CROP}_${STRATEGY}.parquet"
    shards_root="data/wds_shards/${CROP}_${STRATEGY}"
    if [ ! -f "$capt_out" ] && [ ! -f "${capt_out%.parquet}.tsv" ]; then
      echo "[matrix] building captions: $STRATEGY"
      python scripts/build_biocap_captions.py --strategy "$STRATEGY" --crop "$CROP" --out "$capt_out"
    fi
    if [ ! -d "$shards_root/train" ]; then
      caps_path="$capt_out"
      [ -f "$caps_path" ] || caps_path="${capt_out%.parquet}.tsv"
      echo "[matrix] building shards: $STRATEGY"
      python scripts/build_biocap_shards.py --captions "$caps_path" --out-dir "$shards_root"
    fi
  done
fi

# ---- 2. Submit each variant -------------------------------------------------
for v in "${BIOCAP_VARIANTS[@]}"; do
    biocap_parse_variant "$v"
    cmd=(sbatch $WAIT_FLAG --job-name="biocap-$VARIANT_TAG" --export=ALL,VARIANT="$VARIANT_TAG",CROP="$CROP" scripts/submit_biocap_train.sh)
    echo "[matrix] $VARIANT_TAG  strategy=$STRATEGY  proj=$PROJ  epochs=$EPOCHS  subset=$SUBSET  tables=$PAPER_TABLES"
    if [ "${PATHOME_DRY_RUN:-0}" = "1" ]; then
        echo "  (dry) ${cmd[*]}"
    else
        "${cmd[@]}"
    fi
done

echo "[matrix] all jobs submitted (or dry-run printed)."
