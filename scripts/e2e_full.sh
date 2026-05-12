#!/bin/bash
# ============================================================================
# scripts/e2e_full.sh
# ============================================================================
# THE end-to-end pipeline. Runs:
#
#   LOCAL   e2e_local.sh         (Setup, image cache, Phase 0, git push)
#       |
#       v   ssh PATHOME_NOVA_HOST
#   NOVA    e2e_nova.sh          (pull, Phase 0R, OBSERVE train + eval, push)
#       |
#       v   back on LOCAL
#   LOCAL   e2e_visualize.sh     (pull, viz, paper)
#
# Required env vars for the SSH leg:
#   PATHOME_NOVA_HOST     e.g.  user@hpc-login.iastate.edu
#   PATHOME_NOVA_REPO     repo path on the Nova side (sets PATHOME_REPO there)
#
# Skip any leg by setting:
#   PATHOME_SKIP_LOCAL=1
#   PATHOME_SKIP_NOVA=1
#   PATHOME_SKIP_VIZ=1
#
# When PATHOME_NOVA_HOST is unset, the Nova leg is skipped (so this script
# also acts as a "local-only smoke" runner when you already have results
# on disk).
# ============================================================================
set -euo pipefail

PATHOME_REPO="${PATHOME_REPO:-$(pwd)}"
cd "$PATHOME_REPO"

NOVA_HOST="${PATHOME_NOVA_HOST:-}"
NOVA_REPO="${PATHOME_NOVA_REPO:-}"

echo "================================================================="
echo "  e2e_full : LOCAL -> NOVA -> LOCAL"
echo "================================================================="
echo "  LOCAL repo: $PATHOME_REPO"
echo "  NOVA host:  ${NOVA_HOST:-(skipped)}"
echo "  NOVA repo:  ${NOVA_REPO:-(n/a)}"
echo

# ---- LOCAL part 1 ---------------------------------------------------------
if [ "${PATHOME_SKIP_LOCAL:-0}" != "1" ]; then
  echo
  echo "===== leg 1/3 : e2e_local.sh ====="
  bash scripts/e2e_local.sh
else
  echo "[skip] PATHOME_SKIP_LOCAL=1"
fi

# ---- NOVA part ------------------------------------------------------------
if [ "${PATHOME_SKIP_NOVA:-0}" != "1" ]; then
  if [ -z "$NOVA_HOST" ]; then
    echo
    echo "===== leg 2/3 : e2e_nova.sh  ===== (NO PATHOME_NOVA_HOST — skipped)"
  else
    echo
    echo "===== leg 2/3 : e2e_nova.sh on $NOVA_HOST ====="
    SSH_CMD="cd '$NOVA_REPO' && PATHOME_REPO='$NOVA_REPO' bash scripts/e2e_nova.sh"
    ssh "$NOVA_HOST" "$SSH_CMD"
  fi
else
  echo "[skip] PATHOME_SKIP_NOVA=1"
fi

# ---- LOCAL part 2 ---------------------------------------------------------
if [ "${PATHOME_SKIP_VIZ:-0}" != "1" ]; then
  echo
  echo "===== leg 3/3 : e2e_visualize.sh ====="
  bash scripts/e2e_visualize.sh
else
  echo "[skip] PATHOME_SKIP_VIZ=1"
fi

echo
echo "e2e_full complete."
echo
echo "Inspect:"
echo "  results/figures/                   PNGs ready for the paper"
echo "  plantswarm/latex/auto_*.tex        \\input{...} from the paper main"
echo "  plantswarm/latex/acl_latex.pdf     compiled PDF (if available)"
echo "  artifacts/pathome_seed/symptoms_seed.json   final KB"
echo "  observe/checkpoints/observe_best.pt         trained student"
echo "  results/observe_eval.json                   held-out metrics"
