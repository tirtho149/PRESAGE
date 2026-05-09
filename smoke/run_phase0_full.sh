#!/bin/bash
# ============================================================================
# smoke/run_phase0_full.sh
# ============================================================================
# Single-command, perfect-KB regenerate for Tomato + Soybean.
#
# What runs (every stage end-to-end, default = MAXIMUM COVERAGE):
#
#   1. Filter the smoke CSV               → BugWood_Diseases_smoke_usable.csv
#   2. State-aware image cache top-up     → one image per (crop, disease, state)
#   3. Cross-region SAGE pipeline:
#        discovery (claude -p WebSearch)  → all candidate URLs per disease
#        extraction (claude -p)           → verbatim quotes + treatments
#        reconciliation (claude -p)       → canonical entries with treatments
#        → final_registry.json per crop
#   4. Per-state VLM observation:
#        claude -p + Read tool looks at each cached Bugwood image
#        + reads canonical reference from step 3
#        → severity / lesion_morphology / affected_organs /
#          spread_pattern / variations_from_canonical
#        → regional_observations.json per crop
#   5. Adapter merge → smoke/artifacts/pathome_seed/symptoms_seed.json
#
# Output schema:
#   SymptomProfile {
#     canonical: CanonicalDisease           # one block per disease (text)
#     regional_observations: {state: ...}   # per-state VLM observations + variations
#   }
#
# Walltime / cost (full coverage):
#   ~45–90 min wall, ~$5–15 in claude -p OAuth quota
#
# Knobs:
#   FULL_QUICK=1         caps sources/states for fast iteration (~15-25 min, ~$1-3)
#   FULL_KEEP_CACHE=1    skip clearing the cached final_registry.json — reuse
#                        existing cross-region run (treatments may be missing
#                        if the cache predates the prompt update)
#   FULL_SKIP_SETUP=1    CSV already filtered
#   FULL_SKIP_CACHE=1    image cache already topped up
#   FULL_SKIP_KB=1       skip the python -m pathome_kb call (no-op smoke)
#
# Auth requirements:
#   - claude CLI on PATH and authenticated (`claude auth login`)
#   - ANTHROPIC_API_KEY optional (auto-falls-back to claude -p)
# ============================================================================

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RAW_CSV="smoke/BugWood_Diseases_smoke.csv"
USABLE_CSV="smoke/BugWood_Diseases_smoke_usable.csv"
OUT="smoke/artifacts/pathome_seed/symptoms_seed.json"
CACHE_DIR="smoke/.bugwood_cache"

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

QUICK_ARG=()
if [ "${FULL_QUICK:-0}" = "1" ]; then
  QUICK_ARG=(--quick)
  echo "[mode] FULL_QUICK=1 — capping sources/states for fast iteration"
else
  echo "[mode] FULL coverage — every source URL, every state, every visual field"
fi

# ----------------------------------------------------------------------------
# 1. Filter the smoke CSV
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
# 2. State-aware image cache top-up
# ----------------------------------------------------------------------------
if [ "${FULL_SKIP_CACHE:-0}" != "1" ]; then
  step "2. State-aware image cache top-up"
  python3 scripts/ensure_state_image_cache.py \
    --csv "$USABLE_CSV" \
    --cache-dir "$CACHE_DIR"
fi

# ----------------------------------------------------------------------------
# 3. Optionally drop stale per-crop registries so the new prompts run
#    (treatments was added to the extraction/reconciliation prompts; cached
#    registries from before that change won't have treatments populated).
# ----------------------------------------------------------------------------
if [ "${FULL_KEEP_CACHE:-0}" != "1" ]; then
  step "3a. Clearing stale registries to re-run with treatments prompt"
  for crop in Tomato Soybean; do
    rm -f "artifacts/pathome_kb/$crop/raw_extractions.json" \
          "artifacts/pathome_kb/$crop/final_registry.json" \
          "artifacts/pathome_kb/$crop/registry.md" \
          "artifacts/pathome_kb/$crop/internet.xlsx"
  done
  echo "  (kept discovery_results.json — re-using cached URLs)"
fi

# ----------------------------------------------------------------------------
# 4 + 5. Cross-region SAGE + per-state VLM observation + merge
# ----------------------------------------------------------------------------
if [ "${FULL_SKIP_KB:-0}" != "1" ]; then
  step "3b. Cross-region SAGE pipeline + per-state VLM observation"
  python3 -m pathome_kb \
    --csv "$USABLE_CSV" \
    --out "$OUT" \
    --regional \
    --resume-from extraction \
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
n_canon = sum(1 for p in profiles if (p.get('canonical') or {}).get('summary'))
n_reg   = sum(1 for p in profiles if p.get('regional_observations'))
n_blocks = sum(len(p.get('regional_observations') or {}) for p in profiles)
n_with_treat = sum(1 for p in profiles if (p.get('canonical') or {}).get('treatments'))
n_text = n_image = 0
n_variations = 0
for p in profiles:
    for f, cits in ((p.get('canonical') or {}).get('sources') or {}).items():
        for c in cits:
            n_text += 1 if c.get('grounding','text') == 'text' else 0
    for state, obs in (p.get('regional_observations') or {}).items():
        n_variations += len(obs.get('variations_from_canonical') or [])
        for f, cits in (obs.get('sources') or {}).items():
            for c in cits:
                n_image += 1 if c.get('grounding') == 'image' else 0
print(f'profiles total                   : {len(profiles)}')
print(f'profiles w/ canonical summary    : {n_canon}')
print(f'profiles w/ canonical treatments : {n_with_treat}')
print(f'profiles w/ regional observations: {n_reg}')
print(f'total per-state blocks           : {n_blocks}')
print(f'total variations bullets         : {n_variations}')
print(f'text-grounded citations (canonical): {n_text}')
print(f'image-grounded citations (regional): {n_image}')
"

echo
echo "Push the seed to GitHub:"
echo "  git add -f $OUT \\"
echo "             $USABLE_CSV \\"
echo "             artifacts/pathome_kb/Tomato/{discovery_results,final_registry,regional_observations}.json \\"
echo "             artifacts/pathome_kb/Soybean/{discovery_results,final_registry,regional_observations}.json"
echo "  git commit -m 'smoke: regenerate state-aware KB'"
echo "  git push origin main"
echo
echo "Then on Nova:"
echo "  ssh tirtho@hpc-login.iastate.edu"
echo "  cd /work/mech-ai-scratch/tirtho/PlantSwarm && git pull origin main"
echo "  sbatch smoke/submit_smoke.sh"
