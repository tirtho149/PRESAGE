#!/bin/bash
# ============================================================================
# smoke/run_phase0_full.sh
# ============================================================================
# Full Phase 0 for the smoke (Tomato + Soybean) — every stage end-to-end,
# default to MAXIMUM COVERAGE (no --quick caps).
#
#   1. Filter the smoke CSV  → BugWood_Diseases_smoke_usable.csv
#   2. Discovery + extraction + reconciliation  (cross-region SAGE)
#      → all 47 + 61 source URLs (not just first 3 per crop)
#   3. Regional text extraction  (per-state, image_id-tagged)
#      → all 118 (crop, disease, state) tuples (not just first 2/disease)
#   4. State-aware image cache top-up (one image per (crop, disease, state))
#   5. Image-grounded fill via claude -p + Read tool
#      → all 8 visual fields per tuple (color, shape, margin, texture,
#        sporulation, progression, plant_parts, distinctive_signs)
#   6. Adapter merge → smoke/artifacts/pathome_seed/symptoms_seed.json
#
# Output: a fully state-aware, dual-grounded seed (text + image citations).
#
# Walltime / cost (full coverage):
#   ~45–90 min wall depending on web-fetch latency
#   ~$5–15 in claude -p OAuth quota
#
# For fast iteration, drop coverage with FULL_QUICK=1:
#   FULL_QUICK=1 bash smoke/run_phase0_full.sh
#   → caps sources/disease at 3, states/disease at 2; ~15–25 min, ~$1–3
#
# Auth requirements:
#   - claude CLI on PATH and authenticated (`claude auth login`)
#   - ANTHROPIC_API_KEY optional (the pipeline auto-falls-back to claude -p)
#
# Skip individual stages:
#   FULL_SKIP_SETUP=1   bash smoke/run_phase0_full.sh   # CSV already filtered
#   FULL_SKIP_KB=1      bash smoke/run_phase0_full.sh   # cross-region + regional already on disk
#   FULL_SKIP_CACHE=1   bash smoke/run_phase0_full.sh   # cache already topped-up
#   FULL_SKIP_IMAGE=1   bash smoke/run_phase0_full.sh   # image-fill already done
# ============================================================================

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RAW_CSV="smoke/BugWood_Diseases_smoke.csv"
USABLE_CSV="smoke/BugWood_Diseases_smoke_usable.csv"
OUT="smoke/artifacts/pathome_seed/symptoms_seed.json"
CACHE_DIR="smoke/.bugwood_cache"

# ----------------------------------------------------------------------------
# Auth pre-flight
# ----------------------------------------------------------------------------
if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' CLI not on PATH"
  echo "Install: curl -fsSL https://claude.ai/install.sh | bash"
  echo "Auth:    claude auth login"
  exit 1
fi

step() {
  echo
  echo "================================================================="
  echo "  $1"
  echo "================================================================="
}

# Optional --quick flag (default OFF = full coverage)
QUICK_ARG=()
if [ "${FULL_QUICK:-0}" = "1" ]; then
  QUICK_ARG=(--quick)
  echo "[mode] FULL_QUICK=1 — capping sources/states/fields for fast iteration"
else
  echo "[mode] FULL coverage — all sources, all states, all visual fields"
fi

# ----------------------------------------------------------------------------
# 1. Setup — filter the smoke CSV
# ----------------------------------------------------------------------------
if [ "${FULL_SKIP_SETUP:-0}" != "1" ]; then
  step "1. Setup — filter smoke CSV"
  python3 scripts/filter_bugwood_csv.py \
    --input "$RAW_CSV" \
    --output "$USABLE_CSV" \
    --threshold "${SMOKE_THRESHOLD:-15}" \
    --report smoke/bugwood_classes_smoke.tsv
fi

# ----------------------------------------------------------------------------
# 2 + 3. Cross-region (SAGE) + regional text extraction
# ----------------------------------------------------------------------------
if [ "${FULL_SKIP_KB:-0}" != "1" ]; then
  step "2+3. Cross-region SAGE pipeline + regional text extraction"
  python3 -m pathome_kb \
    --csv "$USABLE_CSV" \
    --out "$OUT" \
    --regional \
    --only-crops "Tomato,Soybean" \
    "${QUICK_ARG[@]}"
fi

# ----------------------------------------------------------------------------
# 4. State-aware image cache top-up
# ----------------------------------------------------------------------------
if [ "${FULL_SKIP_CACHE:-0}" != "1" ]; then
  step "4. State-aware image cache top-up"
  python3 scripts/ensure_state_image_cache.py \
    --csv "$USABLE_CSV" \
    --cache-dir "$CACHE_DIR"
fi

# ----------------------------------------------------------------------------
# 5. Image-grounded fill + final merge
# ----------------------------------------------------------------------------
if [ "${FULL_SKIP_IMAGE:-0}" != "1" ]; then
  step "5. Image-grounded fill (VLM via claude -p Read)"
  python3 -m pathome_kb \
    --csv "$USABLE_CSV" \
    --out "$OUT" \
    --regional-image-only \
    --only-crops "Tomato,Soybean" \
    "${QUICK_ARG[@]}"
fi

# ----------------------------------------------------------------------------
# Summary + push instructions
# ----------------------------------------------------------------------------
step "Done"
python3 -c "
import json
s = json.load(open('$OUT'))
profiles = s['profiles']
n_text = n_image = 0
n_blocks = 0
for p in profiles:
    rv = p.get('regional_visuals') or {}
    n_blocks += len(rv)
    for state, v in rv.items():
        for field, cits in (v.get('sources') or {}).items():
            for c in cits:
                if c.get('grounding') == 'image': n_image += 1
                else: n_text += 1
print(f'profiles total:            {len(profiles)}')
print(f'profiles w/ visual data:   {sum(1 for p in profiles if (p.get(\"visual\") or {}).get(\"notes\") or (p.get(\"visual\") or {}).get(\"distinctive_signs\"))}')
print(f'profiles w/ regional data: {sum(1 for p in profiles if p.get(\"regional_visuals\"))}')
print(f'per-state blocks:          {n_blocks}')
print(f'text-grounded citations:   {n_text}')
print(f'image-grounded citations:  {n_image}')
"

echo
echo "Push the seed to GitHub:"
echo
echo "  git add -f $OUT \\"
echo "             $USABLE_CSV \\"
echo "             artifacts/pathome_kb/Tomato/{discovery_results,final_registry,regional_registries,regional_image_fills}.json \\"
echo "             artifacts/pathome_kb/Soybean/{discovery_results,final_registry,regional_registries,regional_image_fills}.json"
echo "  git commit -m 'smoke: full state-aware phase 0'"
echo "  git push origin main"
echo
echo "Then on Nova:"
echo "  ssh tirtho@hpc-login.iastate.edu"
echo "  cd /work/mech-ai-scratch/tirtho/PlantSwarm && git pull origin main"
echo "  sbatch smoke/submit_smoke.sh"
