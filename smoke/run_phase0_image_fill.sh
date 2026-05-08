#!/bin/bash
# ============================================================================
# smoke/run_phase0_image_fill.sh
# ============================================================================
# Image-grounded fill stage for the smoke. Looks at every cached Bugwood
# photo and asks claude -p (with the Read tool) to fill in any visual
# fields that the text-grounded regional extraction left empty (color,
# shape, margin, texture, sporulation, progression). Fields the text
# pass already populated are LEFT UNTOUCHED — this is strictly additive.
#
# Pre-flight:
#   1. smoke/run_phase0_local.sh has run (or smoke/run_phase0_regional_only.sh)
#      → artifacts/pathome_kb/{Tomato,Soybean}/regional_registries.json exists
#   2. smoke/.bugwood_cache/*.jpg exists (Phase 1 build downloads these)
#   3. claude CLI on PATH and authenticated
#
# Output (overwritten):
#   artifacts/pathome_kb/{Tomato,Soybean}/regional_image_fills.json
#   smoke/artifacts/pathome_seed/symptoms_seed.json   (re-merged with grounding=image
#                                                      citations on previously-empty fields)
#
# Walltime: ~10-15 min for the smoke (~38 (profile, state) tuples × ~30s/tuple
# with 4 parallel workers, minus those where the primary image isn't cached).
# ============================================================================

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

USABLE_CSV="smoke/BugWood_Diseases_smoke_usable.csv"
OUT="smoke/artifacts/pathome_seed/symptoms_seed.json"

if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' CLI not on PATH"
  exit 1
fi

if [ ! -f "$USABLE_CSV" ]; then
  echo "ERROR: filtered smoke CSV not found at $USABLE_CSV"
  exit 1
fi

for crop in Tomato Soybean; do
  if [ ! -f "artifacts/pathome_kb/$crop/regional_registries.json" ]; then
    echo "ERROR: artifacts/pathome_kb/$crop/regional_registries.json missing"
    echo "Run smoke/run_phase0_regional_only.sh first."
    exit 1
  fi
done

if [ ! -d "smoke/.bugwood_cache" ] || [ -z "$(ls -A smoke/.bugwood_cache 2>/dev/null)" ]; then
  echo "ERROR: smoke/.bugwood_cache is empty"
  echo "Phase 1 build hasn't run yet — image bytes aren't on disk."
  echo "Run: python3 scripts/build_pathome.py --config smoke/bugwood_pathome_smoke.yaml"
  exit 1
fi

# Top up the cache so EVERY (crop, disease, state) tuple has an image.
# Phase 1's loader only downloaded per_class=4 per (crop, disease); without
# this top-up the image-fill stage skips ~58% of tuples for "no cached image".
echo "================================================================="
echo "  Pre-step: state-aware cache top-up"
echo "================================================================="
python3 scripts/ensure_state_image_cache.py \
  --csv "$USABLE_CSV" \
  --cache-dir "smoke/.bugwood_cache"

echo "================================================================="
echo "  Smoke Phase 0 — IMAGE-FILL pass (Tomato + Soybean, VLM)"
echo "================================================================="
python3 -m pathome_kb \
  --csv     "$USABLE_CSV" \
  --out     "$OUT" \
  --regional-image-only \
  --quick \
  --only-crops "Tomato,Soybean"

echo
echo "================================================================="
echo "  Done. Push the updated seed:"
echo "================================================================="
echo
echo "  git add -f $OUT \\"
echo "             artifacts/pathome_kb/Tomato/regional_image_fills.json \\"
echo "             artifacts/pathome_kb/Soybean/regional_image_fills.json"
echo "  git commit -m 'smoke: image-grounded regional fills'"
echo "  git push origin main"
