#!/bin/bash
# ============================================================================
# scripts/e2e_nova.sh
# ============================================================================
# GPU-host part of the end-to-end pipeline. Run THIS on Nova (or any host
# with vLLM-capable GPUs):
#   1. git pull canonical artefacts pushed by e2e_local.sh
#   2. Phase 0R: vLLM boot + Qwen swarm + Claude web-search verifier
#                (populates artifacts/pathome_kb/<crop>/final_registry.json
#                 with regional_observations / deltas — required by 0deltas
#                 caption strategies)
#   3. BioCAP captions + shards build (one (caption, shard) bundle per
#      unique strategy across the variant matrix)
#   4. BioCAP training matrix — variants T01..T11 (paper Tables 3, 4, 6,
#      17, 18, 19, 20 + Fig 3)
#   5. Off-shelf baseline cache (CLIP, SigLIP, FG-CLIP, BioTrove-CLIP,
#      BioCLIP, BioCLIP-2, BioCAP-HF)
#   6. BioCAP eval suite — zero-shot classification on PV/PW/PlantDoc +
#      retrieval bench + few-shot, for every variant + baseline
#   7. Aggregate paper tables -> results/biocap_report.md
#   8. git push results
#
# This script uses `sbatch --wait` so each phase blocks until completion.
#
# Knobs (env vars):
#   PATHOME_GIT_REMOTE       origin
#   PATHOME_GIT_BRANCH       main
#   CROP                     Tomato
#   PV_ROOT, PW_ROOT, PLANTDOC_ROOT   eval dataset roots
#   PATHOME_SKIP_{PHASE0R,CAPTIONS,TRAIN,EVAL,PUSH}   0/1 toggles
# ============================================================================
set -euo pipefail

PATHOME_REPO="${PATHOME_REPO:-$(pwd)}"
cd "$PATHOME_REPO"

GIT_REMOTE="${PATHOME_GIT_REMOTE:-origin}"
GIT_BRANCH="${PATHOME_GIT_BRANCH:-main}"
CROP="${CROP:-Tomato}"
PV_ROOT="${PV_ROOT:-data/eval/PlantVillage}"
PW_ROOT="${PW_ROOT:-data/eval/PlantWild}"
PLANTDOC_ROOT="${PLANTDOC_ROOT:-data/eval/PlantDoc/test}"
RESULTS_DIR="${RESULTS_DIR:-results/biocap_eval}"

echo "================================================================="
echo "  e2e_nova : pull + Phase 0R + BioCAP matrix + eval + push"
echo "================================================================="

# ---- 1. git pull ----------------------------------------------------------
echo
echo "[1/8] git pull canonical artefacts"
git pull "$GIT_REMOTE" "$GIT_BRANCH" --ff-only
mkdir -p logs "$RESULTS_DIR" data/bugwood_captions data/wds_shards

# ---- 2. Phase 0R (sbatch --wait) ------------------------------------------
if [ "${PATHOME_SKIP_PHASE0R:-0}" != "1" ]; then
  echo
  echo "[2/8] Phase 0R — Qwen swarm + Claude verifier (sbatch --wait)"
  sbatch --wait scripts/submit_phase0r_regional.sh
else
  echo "  [skip] PATHOME_SKIP_PHASE0R=1"
fi

# ---- 3. BioCAP captions + shards (per strategy, one-shot) -----------------
if [ "${PATHOME_SKIP_CAPTIONS:-0}" != "1" ]; then
  echo
  echo "[3/8] BioCAP captions + shards (foreground)"
  # Pre-bake every strategy referenced in the variant matrix.
  # shellcheck disable=SC1091
  source scripts/biocap_variants.sh
  declare -A SEEN=()
  for v in "${BIOCAP_VARIANTS[@]}"; do
    biocap_parse_variant "$v"
    if [ -n "${SEEN[$STRATEGY]:-}" ]; then continue; fi
    SEEN[$STRATEGY]=1
    capt="data/bugwood_captions/${CROP}_${STRATEGY}.parquet"
    shards="data/wds_shards/${CROP}_${STRATEGY}"
    if [ ! -f "$capt" ] && [ ! -f "${capt%.parquet}.tsv" ]; then
      echo "  [captions] strategy=$STRATEGY"
      python scripts/build_biocap_captions.py --strategy "$STRATEGY" --crop "$CROP" --out "$capt"
    fi
    if [ ! -d "$shards/train" ]; then
      caps_path="$capt"
      [ -f "$caps_path" ] || caps_path="${capt%.parquet}.tsv"
      echo "  [shards] strategy=$STRATEGY"
      python scripts/build_biocap_shards.py --captions "$caps_path" --out-dir "$shards"
    fi
  done
else
  echo "  [skip] PATHOME_SKIP_CAPTIONS=1"
fi

# ---- 4. BioCAP training matrix --------------------------------------------
if [ "${PATHOME_SKIP_TRAIN:-0}" != "1" ]; then
  echo
  echo "[4/8] BioCAP training matrix (sbatch --wait, sequential)"
  PATHOME_WAIT=1 PATHOME_SKIP_CAPTIONS=1 CROP="$CROP" bash scripts/submit_biocap_matrix.sh
else
  echo "  [skip] PATHOME_SKIP_TRAIN=1"
fi

# ---- 5. Pre-cache off-shelf baselines -------------------------------------
if [ "${PATHOME_SKIP_BASELINES:-0}" != "1" ]; then
  echo
  echo "[5/8] Caching off-shelf baselines"
  python scripts/fetch_baselines.py --skip-if-cached || true
else
  echo "  [skip] PATHOME_SKIP_BASELINES=1"
fi

# ---- 6. Eval matrix -------------------------------------------------------
if [ "${PATHOME_SKIP_EVAL:-0}" != "1" ]; then
  echo
  echo "[6/8] BioCAP eval suite (zero-shot + retrieval + few-shot)"
  # All variants + baselines × {zero-shot PV/PW/PlantDoc, retrieval, few-shot}
  ALL_RUNS=()
  # shellcheck disable=SC1091
  source scripts/biocap_variants.sh
  for v in "${BIOCAP_VARIANTS[@]}"; do
    biocap_parse_variant "$v"
    ALL_RUNS+=("$VARIANT_TAG:checkpoints/$VARIANT_TAG/$VARIANT_TAG/checkpoints/epoch_50.pt")
  done
  # Baselines: name + tag (paired)
  BASELINES_LIST=("clip_vitb16:ViT-B-16:openai"
                  "siglip_vitb16:hf-hub:timm/ViT-B-16-SigLIP-256:"
                  "fgclip:hf-hub:qihoo360/fg-clip-base:"
                  "biotrove:hf-hub:BGLab/BioTrove-CLIP:"
                  "bioclip:hf-hub:imageomics/bioclip:"
                  "bioclip2:hf-hub:imageomics/bioclip-2:"
                  "biocap_hf:hf-hub:imageomics/biocap:")
  for bspec in "${BASELINES_LIST[@]}"; do
    IFS=':' read -r run_id model pretrained <<<"$bspec"
    out_dir="$RESULTS_DIR/$run_id"
    mkdir -p "$out_dir"
    python scripts/evaluate_biocap.py --model "$model" --pretrained "$pretrained" \
        --crop "$CROP" \
        --pv-root "$PV_ROOT" --pw-root "$PW_ROOT" --plantdoc-root "$PLANTDOC_ROOT" \
        --out-dir "$out_dir" || true
    # Retrieval (needs holdout split — only run if any captions parquet has holdout)
    for capt in data/bugwood_captions/${CROP}_*.parquet; do
      [ -f "$capt" ] || continue
      python scripts/evaluate_biocap_retrieval.py --model "$model" --pretrained "$pretrained" \
          --captions "$capt" --out-dir "$out_dir" || true
      break
    done
    python scripts/evaluate_biocap_fewshot.py --model "$model" --pretrained "$pretrained" \
        --crop "$CROP" \
        --pv-root "$PV_ROOT" --pw-root "$PW_ROOT" --plantdoc-root "$PLANTDOC_ROOT" \
        --shots 1 5 --out-dir "$out_dir" || true
  done
  # Trained variants (local ckpts)
  for v in "${BIOCAP_VARIANTS[@]}"; do
    biocap_parse_variant "$v"
    ckpt_dir="train_and_eval/checkpoints/$VARIANT_TAG/$VARIANT_TAG/checkpoints"
    last_ckpt=$(ls "$ckpt_dir"/epoch_*.pt 2>/dev/null | sort -V | tail -n 1 || true)
    if [ -z "$last_ckpt" ]; then
      echo "  [eval] $VARIANT_TAG: no checkpoint found in $ckpt_dir, skipping"
      continue
    fi
    out_dir="$RESULTS_DIR/$VARIANT_TAG"
    mkdir -p "$out_dir"
    python scripts/evaluate_biocap.py --model "$last_ckpt" \
        --crop "$CROP" \
        --pv-root "$PV_ROOT" --pw-root "$PW_ROOT" --plantdoc-root "$PLANTDOC_ROOT" \
        --out-dir "$out_dir" || true
    python scripts/evaluate_biocap_fewshot.py --model "$last_ckpt" \
        --crop "$CROP" \
        --pv-root "$PV_ROOT" --pw-root "$PW_ROOT" --plantdoc-root "$PLANTDOC_ROOT" \
        --shots 1 5 --out-dir "$out_dir" || true
  done
else
  echo "  [skip] PATHOME_SKIP_EVAL=1"
fi

# ---- 7. Aggregate paper tables --------------------------------------------
echo
echo "[7/8] Aggregating paper tables"
python scripts/aggregate_biocap_tables.py --results-dir "$RESULTS_DIR" \
    --out-dir results/tables --report results/biocap_report.md

# ---- 8. git push results --------------------------------------------------
echo
echo "[8/8] Git push results"
git add -f results/biocap_report.md \
           results/tables/*.md \
           "$RESULTS_DIR"/*/*.json \
           artifacts/pathome_kb/*/final_registry.json 2>/dev/null || true
if git diff --cached --quiet; then
  echo "  no result artefacts changed; skipping commit"
else
  git commit -m "BioCAP-on-Bugwood: results ($(date -u +%Y-%m-%dT%H:%MZ))"
fi
if [ "${PATHOME_SKIP_PUSH:-0}" = "1" ]; then
  echo "  PATHOME_SKIP_PUSH=1 — committed but not pushing"
else
  git push "$GIT_REMOTE" "$GIT_BRANCH"
fi

echo
echo "e2e_nova complete."
echo "Master report -> results/biocap_report.md"
