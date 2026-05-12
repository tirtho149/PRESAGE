#!/bin/bash
# ============================================================================
# scripts/setup_image_cache.sh
# ============================================================================
# Phase 0R input prep — download one Bugwood photograph per
# (crop, disease, state) tuple referenced in the filtered CSV. Idempotent:
# previously-cached images are skipped.
#
# Inputs:
#   $PATHOME_USABLE_CSV    (default: BugWood_Diseases_usable.csv)
#   $PATHOME_IMAGE_CACHE_DIR (default: .bugwood_cache)
#
# Output:
#   <cache-dir>/<image_number>.{jpg|png|webp}
#
# Documented in README under "Script reference".
# ============================================================================
set -euo pipefail

PATHOME_REPO="${PATHOME_REPO:-$(pwd)}"
cd "$PATHOME_REPO"

CSV="${PATHOME_USABLE_CSV:-BugWood_Diseases_usable.csv}"
CACHE_DIR="${PATHOME_IMAGE_CACHE_DIR:-.bugwood_cache}"

if [ ! -f "$CSV" ]; then
  echo "ERROR: filtered CSV not found at $CSV"
  echo "Run scripts/submit_pathome_setup_filter.sh first (or its python equivalent)"
  exit 1
fi

echo "================================================================="
echo "  Image cache top-up"
echo "================================================================="
echo "  csv:        $CSV"
echo "  cache dir:  $CACHE_DIR"
echo

python3 scripts/ensure_state_image_cache.py \
  --csv "$CSV" \
  --cache-dir "$CACHE_DIR"

echo
echo "Done."
