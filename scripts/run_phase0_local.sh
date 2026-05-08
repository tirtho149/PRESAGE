#!/bin/bash
# ============================================================================
# scripts/run_phase0_local.sh
# ============================================================================
# Phase 0 of the Pathome pipeline — run THIS on your local machine, NOT on
# Nova. Nova's compute nodes don't allow the outbound HTTPS / OAuth login
# flow that the `claude` CLI needs.
#
#   Local                          GitHub                       Nova
#   ─────                          ──────                       ────
#   bash run_phase0_local.sh   →   git push  →    git pull  →   sbatch all
#                                  symptoms_seed.json
#                                  pathome_kb/<Crop>/...
#
# Prerequisites on your local machine
#   - python venv with pathome dependencies (`pip install -r requirements.txt`)
#   - `claude` CLI installed + auth'd: `curl -fsSL https://claude.ai/install.sh | bash`
#                                       `claude auth login`
#   - ANTHROPIC_API_KEY in env or .env at repo root (Anthropic SDK is
#     used for the extraction + reconciliation stages)
#
# Output
#   artifacts/pathome_seed/symptoms_seed.json     (consumed by Phase 1 on Nova)
#   artifacts/pathome_kb/<Crop>/...               (per-crop provenance)
#
# After this finishes you'll be reminded to:
#   git add -f artifacts/pathome_seed/symptoms_seed.json
#   git add -f artifacts/pathome_kb/                       # optional, for audit
#   git commit -m "Phase 0: seeded KB"
#   git push origin main
# ============================================================================

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CSV="${PATHOME_USABLE_CSV:-BugWood_Diseases_usable.csv}"
OUT="${PATHOME_SEED_FILE:-artifacts/pathome_seed/symptoms_seed.json}"

if [ ! -f "$CSV" ]; then
  echo "ERROR: filtered CSV not found at $CSV"
  echo "Generate it locally first:"
  echo "  python scripts/filter_bugwood_csv.py --threshold 10"
  exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' CLI not on PATH"
  echo "Install: curl -fsSL https://claude.ai/install.sh | bash"
  echo "Then:    claude auth login"
  exit 1
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ ! -f .env ]; then
  echo "ERROR: ANTHROPIC_API_KEY not set and no .env at repo root."
  echo "Set it: export ANTHROPIC_API_KEY=sk-ant-..."
  exit 1
fi

ARGS=("--csv" "$CSV" "--out" "$OUT")
if [ "${PATHOME_SEED_QUICK:-0}" = "1" ]; then ARGS+=("--quick"); fi
if [ -n "${PATHOME_SEED_LIMIT:-}" ];      then ARGS+=("--limit-crops" "$PATHOME_SEED_LIMIT"); fi
if [ -n "${PATHOME_SEED_ONLY_CROPS:-}" ]; then ARGS+=("--only-crops"  "$PATHOME_SEED_ONLY_CROPS"); fi
if [ -n "${PATHOME_SEED_RESUME:-}" ];     then ARGS+=("--resume-from" "$PATHOME_SEED_RESUME"); fi
if [ "${PATHOME_SEED_NO_CACHE:-0}" = "1" ]; then ARGS+=("--no-cache"); fi

echo "================================================================="
echo "  Phase 0 — pathome_kb (LOCAL machine)"
echo "================================================================="
echo "  csv:   $CSV"
echo "  out:   $OUT"
echo "  args:  ${ARGS[*]}"
echo "================================================================="
python -m pathome_kb "${ARGS[@]}"

echo
echo "================================================================="
echo "  Phase 0 complete. Push the seeded KB to GitHub:"
echo "================================================================="
echo
echo "  # The seed file is the input Nova needs for Phase 1."
echo "  git add -f $OUT"
echo
echo "  # OPTIONAL: include the per-crop provenance dump for paper audit."
echo "  # ~50-100 MB across 197 crops; skip this if you'd rather rsync."
echo "  git add -f artifacts/pathome_kb/"
echo
echo "  git commit -m 'Phase 0: seeded KB ($(date -u +%Y-%m-%dT%H:%MZ))'"
echo "  git push origin main"
echo
echo "Then on Nova:"
echo "  cd /work/mech-ai-scratch/tirtho/PlantSwarm && git pull"
echo "  bash scripts/submit_pathome_all.sh    # runs setup → Phase 1 → … → Phase 5"
