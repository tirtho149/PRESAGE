#!/bin/bash
# ============================================================================
# scripts/sh_01_phase0_local.sh           STEP 1 — LOCAL
# ============================================================================
# Phase 0 — canonical KB build via Claude (LOCAL, OAuth-authenticated CLI),
# then git push canonical artifacts to GitHub so step 2 on Nova can pull.
#
# What Phase 0 produces
#   For each crop in $CROPS:
#     artifacts/pathome_kb/<Crop>/discovery_results.json   per-disease URL list
#     artifacts/pathome_kb/<Crop>/raw_extractions.json     per-source quotes
#     artifacts/pathome_kb/<Crop>/final_registry.json      canonical KB
#                                                          (NON-visual fields;
#                                                          regional_observations
#                                                          will be empty here)
#
# Pre-reqs
#   - `claude` CLI authenticated (run `claude` once interactively)
#   - optional ANTHROPIC_API_KEY in .env (speeds up the Anthropic-SDK
#     reconciliation step)
#   - BugWood_Diseases_usable.csv present
#
# Knobs (env vars)
#   CROPS                  comma-separated crop allowlist
#                          "smoke" = "Soybean,Tomato"
#                          "all"   = no filter (production, ~16-24 h, ~$60-180)
#                          default "smoke" — safer to bake in
#   PATHOME_USABLE_CSV     default BugWood_Diseases_usable.csv
#   PATHOME_SKIP_PUSH      set 1 to commit but not push (e.g. on plane)
#   GIT_REMOTE             default origin
#   GIT_BRANCH             default main
# ============================================================================
set -euo pipefail

REPO_ROOT="${PATHOME_REPO:-$(pwd)}"
cd "$REPO_ROOT"

CROPS="${CROPS:-smoke}"
PATHOME_USABLE_CSV="${PATHOME_USABLE_CSV:-BugWood_Diseases_usable.csv}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"

# Resolve crop allowlist.
case "$CROPS" in
  smoke) PATHOME_ONLY_CROPS="Soybean,Tomato";;
  all)   PATHOME_ONLY_CROPS="";;
  *)     PATHOME_ONLY_CROPS="$CROPS";;
esac

echo "================================================================="
echo " STEP 1 — Phase 0 canonical KB (LOCAL, Claude)"
echo "================================================================="
echo "  CROPS                : ${PATHOME_ONLY_CROPS:-(all 197)}"
echo "  PATHOME_USABLE_CSV   : $PATHOME_USABLE_CSV"
echo "  GIT_REMOTE           : $GIT_REMOTE"
echo "  GIT_BRANCH           : $GIT_BRANCH"
echo

# Sanity check Claude CLI before burning time.
if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' CLI not found on PATH. Install + run 'claude' once"
  echo "       interactively to authenticate, then retry."
  exit 2
fi

# Run Phase 0.
echo "[1/3] Running Phase 0 (Claude discovery -> extraction -> reconciliation)"
if [ -n "$PATHOME_ONLY_CROPS" ]; then
  PATHOME_ONLY_CROPS="$PATHOME_ONLY_CROPS" \
    python -m pathome_kb --csv "$PATHOME_USABLE_CSV"
else
  python -m pathome_kb --csv "$PATHOME_USABLE_CSV"
fi

# Stage + commit.
echo
echo "[2/3] git add canonical artifacts"
git add -f artifacts/pathome_kb/*/final_registry.json \
           artifacts/pathome_kb/*/discovery_results.json \
           2>/dev/null || true

if git diff --cached --quiet; then
  echo "  no changes to commit"
else
  git commit -m "Phase 0 (LOCAL): canonical KB for ${PATHOME_ONLY_CROPS:-all crops}"
fi

# Push.
echo
echo "[3/3] git push to $GIT_REMOTE $GIT_BRANCH"
if [ "${PATHOME_SKIP_PUSH:-0}" = "1" ]; then
  echo "  PATHOME_SKIP_PUSH=1 — committed but not pushing"
else
  git push "$GIT_REMOTE" "$GIT_BRANCH"
fi

echo
echo "STEP 1 done."
echo "  Next: ssh to Nova, then run scripts/sh_02_swarm_nova.sh"
