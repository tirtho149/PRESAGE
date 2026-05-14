#!/bin/bash
# ============================================================================
# scripts/sh_04_finetune_nova.sh           STEP 4 — NOVA
# ============================================================================
# Pull the verified-KB (step 3 output) from GitHub, build PathomeOOD
# captions + WebDataset shards from the KB, sbatch the 11-variant
# training matrix, run the eval suite (PV / PD / PW + retrieval +
# few-shot), aggregate paper-style tables, push results to GitHub.
#
# Each step has a clear skip-knob so a partial re-run is cheap.
#
# Wall-clock on one A100:
#   - captions + shards   ~5-15 min per strategy (7 strategies)
#   - training matrix     ~5 GPU-h for all 11 variants
#   - baselines + eval    ~2-3 GPU-h
#   - aggregation         ~1 min
# Smoke: ~1-2 GPU-h total. Production: ~8-12 GPU-h.
#
# Knobs
#   CROPS                   "smoke" = Tomato (only KB-covered crop with
#                                     reasonable image counts), "all" =
#                                     all crops (captioner fallback fills
#                                     in classes lacking a KB profile)
#                           default "smoke"
#   PATHOME_SKIP_{CAPTIONS,TRAIN,BASELINES,EVAL,AGG,PUSH}   0/1 toggles
# ============================================================================
set -euo pipefail

REPO_ROOT="${PATHOME_REPO:-$(pwd)}"
cd "$REPO_ROOT"

CROPS="${CROPS:-smoke}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"

case "$CROPS" in
  smoke) CROP_TAG="Tomato";;
  all)   CROP_TAG="all";;
  *)     CROP_TAG="$CROPS";;
esac

# Where everything lands.
RESULTS_DIR="${RESULTS_DIR:-results/pathomeood_eval}"
PV_ROOT="${PV_ROOT:-data/eval/PlantVillage}"
PW_ROOT="${PW_ROOT:-data/eval/PlantWild}"
PLANTDOC_ROOT="${PLANTDOC_ROOT:-data/eval/PlantDoc/test}"

echo "================================================================="
echo " STEP 4 — PathomeOOD CLIP fine-tuning (NOVA)"
echo "================================================================="
echo "  CROP_TAG          : $CROP_TAG"
echo "  RESULTS_DIR       : $RESULTS_DIR"
echo "  eval roots        : PV=$PV_ROOT  PD=$PLANTDOC_ROOT  PW=$PW_ROOT"

# Pull verified KB.
echo
echo "[1/7] git pull verified KB"
git pull "$GIT_REMOTE" "$GIT_BRANCH" --ff-only
mkdir -p logs "$RESULTS_DIR" data/bugwood_captions data/wds_shards

# Build captions + shards per unique strategy.
if [ "${PATHOME_SKIP_CAPTIONS:-0}" != "1" ]; then
  echo
  echo "[2/7] Build PathomeOOD captions + shards (per unique strategy)"
  # shellcheck disable=SC1091
  source scripts/pathomeood_variants.sh
  declare -A SEEN=()
  for v in "${PATHOMEOOD_VARIANTS[@]}"; do
    pathomeood_parse_variant "$v"
    if [ -n "${SEEN[$STRATEGY]:-}" ]; then continue; fi
    SEEN[$STRATEGY]=1
    capt="data/bugwood_captions/${CROP_TAG}_${STRATEGY}.parquet"
    shards="data/wds_shards/${CROP_TAG}_${STRATEGY}"
    if [ ! -f "$capt" ] && [ ! -f "${capt%.parquet}.tsv" ]; then
      echo "  [captions] strategy=$STRATEGY"
      if [ "$CROP_TAG" = "all" ]; then
        python scripts/build_pathomeood_captions.py --strategy "$STRATEGY" --out "$capt"
      else
        python scripts/build_pathomeood_captions.py --strategy "$STRATEGY" --crop "$CROP_TAG" --out "$capt"
      fi
    fi
    if [ ! -d "$shards/train" ]; then
      caps_path="$capt"
      [ -f "$caps_path" ] || caps_path="${capt%.parquet}.tsv"
      echo "  [shards] strategy=$STRATEGY"
      python scripts/build_pathomeood_shards.py --captions "$caps_path" --out-dir "$shards"
    fi
  done
else
  echo "  [skip] PATHOME_SKIP_CAPTIONS=1"
fi

# Train 11-variant matrix.
if [ "${PATHOME_SKIP_TRAIN:-0}" != "1" ]; then
  echo
  echo "[3/7] Train PathomeOOD 11-variant matrix"
  PATHOME_WAIT=1 PATHOME_SKIP_CAPTIONS=1 CROP="$CROP_TAG" \
    bash scripts/submit_pathomeood_matrix.sh
else
  echo "  [skip] PATHOME_SKIP_TRAIN=1"
fi

# Cache off-shelf baselines.
if [ "${PATHOME_SKIP_BASELINES:-0}" != "1" ]; then
  echo
  echo "[4/7] Cache 5 off-shelf CLIP baselines"
  python scripts/fetch_baselines.py --skip-if-cached || true
else
  echo "  [skip] PATHOME_SKIP_BASELINES=1"
fi

# Eval suite.
if [ "${PATHOME_SKIP_EVAL:-0}" != "1" ]; then
  echo
  echo "[5/7] Eval suite (zero-shot + retrieval + few-shot)"
  # Baselines:
  BASELINES_LIST=(
    "clip_vitb16:ViT-B-16:openai"
    "siglip_vitb16:hf-hub:timm/ViT-B-16-SigLIP-256:"
    "fgclip:hf-hub:qihoo360/fg-clip-base:"
    "biotrove:hf-hub:BGLab/BioTrove-CLIP:"
    "bioclip:hf-hub:imageomics/bioclip:"
    "bioclip2:hf-hub:imageomics/bioclip-2:"
  )
  for bspec in "${BASELINES_LIST[@]}"; do
    IFS=':' read -r run_id model pretrained <<<"$bspec"
    out_dir="$RESULTS_DIR/$run_id"
    mkdir -p "$out_dir"
    python scripts/evaluate_pathomeood.py --model "$model" --pretrained "$pretrained" \
        --crop "$CROP_TAG" \
        --pv-root "$PV_ROOT" --pw-root "$PW_ROOT" --plantdoc-root "$PLANTDOC_ROOT" \
        --out-dir "$out_dir" || true
    for capt in data/bugwood_captions/${CROP_TAG}_*.parquet; do
      [ -f "$capt" ] || continue
      python scripts/evaluate_pathomeood_retrieval.py --model "$model" --pretrained "$pretrained" \
          --captions "$capt" --out-dir "$out_dir" || true
      break
    done
    python scripts/evaluate_pathomeood_fewshot.py --model "$model" --pretrained "$pretrained" \
        --crop "$CROP_TAG" \
        --pv-root "$PV_ROOT" --pw-root "$PW_ROOT" --plantdoc-root "$PLANTDOC_ROOT" \
        --shots 1 5 --out-dir "$out_dir" || true
  done

  # Trained variants (local ckpts):
  # shellcheck disable=SC1091
  source scripts/pathomeood_variants.sh
  for v in "${PATHOMEOOD_VARIANTS[@]}"; do
    pathomeood_parse_variant "$v"
    ckpt_dir="train_and_eval/checkpoints/$VARIANT_TAG/$VARIANT_TAG/checkpoints"
    last_ckpt=$(ls "$ckpt_dir"/epoch_*.pt 2>/dev/null | sort -V | tail -n 1 || true)
    if [ -z "$last_ckpt" ]; then
      echo "  [eval] $VARIANT_TAG: no checkpoint in $ckpt_dir, skipping"
      continue
    fi
    out_dir="$RESULTS_DIR/$VARIANT_TAG"
    mkdir -p "$out_dir"
    python scripts/evaluate_pathomeood.py --model "$last_ckpt" \
        --crop "$CROP_TAG" \
        --pv-root "$PV_ROOT" --pw-root "$PW_ROOT" --plantdoc-root "$PLANTDOC_ROOT" \
        --out-dir "$out_dir" || true
    python scripts/evaluate_pathomeood_fewshot.py --model "$last_ckpt" \
        --crop "$CROP_TAG" \
        --pv-root "$PV_ROOT" --pw-root "$PW_ROOT" --plantdoc-root "$PLANTDOC_ROOT" \
        --shots 1 5 --out-dir "$out_dir" || true
  done
else
  echo "  [skip] PATHOME_SKIP_EVAL=1"
fi

# Aggregate paper-style tables.
if [ "${PATHOME_SKIP_AGG:-0}" != "1" ]; then
  echo
  echo "[6/7] Aggregate paper-style tables"
  python scripts/aggregate_pathomeood_tables.py --results-dir "$RESULTS_DIR" \
      --out-dir results/tables --report results/pathomeood_report.md
else
  echo "  [skip] PATHOME_SKIP_AGG=1"
fi

# Push results.
echo
echo "[7/7] git push results to $GIT_REMOTE $GIT_BRANCH"
git add -f results/pathomeood_report.md \
           results/tables/*.md \
           "$RESULTS_DIR"/*/*.json 2>/dev/null || true
if git diff --cached --quiet; then
  echo "  no result artefacts changed; skipping commit"
else
  git commit -m "PathomeOOD fine-tune + eval (NOVA): $CROP_TAG ($(date -u +%Y-%m-%dT%H:%MZ))"
  if [ "${PATHOME_SKIP_PUSH:-0}" = "1" ]; then
    echo "  PATHOME_SKIP_PUSH=1 — committed but not pushing"
  else
    git push "$GIT_REMOTE" "$GIT_BRANCH"
  fi
fi

echo
echo "STEP 4 done."
echo "  Master report: results/pathomeood_report.md"
