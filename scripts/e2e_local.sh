#!/bin/bash
# ============================================================================
# scripts/e2e_local.sh
# ============================================================================
# Local part of the end-to-end pipeline:
#   1. Filter the raw Bugwood CSV (Setup)
#   2. Top up the image cache
#   3. Run Phase 0 canonical KB build via Claude
#   4. git add / commit / push canonical artefacts
#
# Prerequisites:
#   - `claude` CLI authed (`claude auth login`)
#   - ANTHROPIC_API_KEY in env or .env at repo root (optional but faster)
#   - Git working tree clean (or you accept the auto-commit)
#
# Knobs (env vars; safe defaults shown):
#   PATHOME_RAW_CSV       BugWood_Diseases.csv
#   PATHOME_USABLE_CSV    BugWood_Diseases_usable.csv
#   PATHOME_ONLY_CROPS    (empty = all crops)
#   PATHOME_SEED_QUICK    0  (set to 1 for fast smoke iteration)
#   PATHOME_SEED_FILE     artifacts/pathome_seed/symptoms_seed.json
#   PATHOME_GIT_REMOTE    origin
#   PATHOME_GIT_BRANCH    main
#   PATHOME_SKIP_PUSH     0  (set to 1 to commit but skip the push)
# ============================================================================
set -euo pipefail

PATHOME_REPO="${PATHOME_REPO:-$(pwd)}"
cd "$PATHOME_REPO"

RAW_CSV="${PATHOME_RAW_CSV:-BugWood_Diseases.csv}"
USABLE_CSV="${PATHOME_USABLE_CSV:-BugWood_Diseases_usable.csv}"
SEED="${PATHOME_SEED_FILE:-artifacts/pathome_seed/symptoms_seed.json}"
GIT_REMOTE="${PATHOME_GIT_REMOTE:-origin}"
GIT_BRANCH="${PATHOME_GIT_BRANCH:-main}"

echo "================================================================="
echo "  e2e_local : Setup + Image Cache + Phase 0 canonical + git push"
echo "================================================================="
echo "  raw csv:    $RAW_CSV"
echo "  usable csv: $USABLE_CSV"
echo "  seed:       $SEED"
echo

# ---- 1. Setup -------------------------------------------------------------
echo
echo "[1/4] Setup — filter CSV"
if [ ! -f "$RAW_CSV" ]; then
  echo "  ERROR: raw CSV not found at $RAW_CSV"
  exit 1
fi
python3 scripts/filter_bugwood_csv.py \
  --input  "$RAW_CSV" \
  --output "$USABLE_CSV" \
  --threshold "${PATHOME_THRESHOLD:-10}" \
  --report bugwood_classes_report.tsv

# ---- 2. Image cache -------------------------------------------------------
echo
echo "[2/4] Image cache top-up"
bash scripts/setup_image_cache.sh

# ---- 3. Phase 0 canonical KB (Claude) -------------------------------------
echo
echo "[3/4] Phase 0 canonical KB (Claude)"
bash scripts/run_phase0_local.sh

# ---- 4. Git push ----------------------------------------------------------
echo
echo "[4/4] Git push canonical artefacts"
git add -f "$SEED" "$USABLE_CSV" bugwood_classes_report.tsv \
            artifacts/pathome_kb/*/final_registry.json \
            artifacts/pathome_kb/*/discovery_results.json 2>/dev/null || true
if git diff --cached --quiet; then
  echo "  no canonical artefacts changed; skipping commit"
else
  git commit -m "Phase 0: canonical KB ($(date -u +%Y-%m-%dT%H:%MZ))"
fi
if [ "${PATHOME_SKIP_PUSH:-0}" = "1" ]; then
  echo "  PATHOME_SKIP_PUSH=1 — committed but not pushing"
else
  git push "$GIT_REMOTE" "$GIT_BRANCH"
fi

echo
echo "e2e_local complete."
echo "Next step: run scripts/e2e_nova.sh on the GPU host (Nova)."
