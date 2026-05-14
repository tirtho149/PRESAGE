#!/bin/bash
# ============================================================================
# scripts/sh_push_to_github.sh             LOCAL or NOVA → GitHub
# ============================================================================
# Explicit hand-off: stage relevant artifacts, commit, push.
#
# Use this between step scripts when you want to push manually instead
# of letting the step's built-in `git push` at the end run. Useful when
# a step crashes mid-run, or when you want to push a partial KB / set
# of features without re-running the whole step.
#
# What it stages by default:
#   artifacts/pathome_kb/*/final_registry.json    KB (canonical + deltas)
#   artifacts/pathome_kb/*/discovery_results.json discovery cache
#   artifacts/bugwood_judgement.json              step-0 judge report
#   results/pathomeood_report.md                  step-5 master report
#   results/tables/*.md                           step-5 paper tables
#   results/pathomeood_eval/*/*.json              step-5 TabPFN JSONs
#   results/figures/                              step-5 Grad-CAM PNGs
#   BugWood_Diseases_usable.csv                   step-0 cleaned CSV
#
# Override via PATHOME_PUSH_PATHS (space-separated glob list).
#
# Knobs
#   PATHOME_PUSH_PATHS   space-separated glob list (default: see above)
#   COMMIT_MSG           commit message (default: timestamped "manual push")
#   GIT_REMOTE           default origin
#   GIT_BRANCH           default main
#   PATHOME_DRY_RUN      set 1 to print plan without pushing
# ============================================================================
set -euo pipefail

REPO_ROOT="${PATHOME_REPO:-$(pwd)}"
cd "$REPO_ROOT"

GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"
DEFAULT_PATHS=(
  "artifacts/pathome_kb/*/final_registry.json"
  "artifacts/pathome_kb/*/discovery_results.json"
  "artifacts/bugwood_judgement.json"
  "results/pathomeood_report.md"
  "results/tables/*.md"
  "results/pathomeood_eval/*/*.json"
  "results/figures/"
  "BugWood_Diseases_usable.csv"
)
PATHS_STR="${PATHOME_PUSH_PATHS:-${DEFAULT_PATHS[*]}}"
read -ra PATHS <<<"$PATHS_STR"

DEFAULT_MSG="manual push ($(date -u +%Y-%m-%dT%H:%MZ))"
COMMIT_MSG="${COMMIT_MSG:-$DEFAULT_MSG}"

echo "================================================================="
echo " push to GitHub"
echo "================================================================="
echo "  GIT_REMOTE   : $GIT_REMOTE"
echo "  GIT_BRANCH   : $GIT_BRANCH"
echo "  COMMIT_MSG   : $COMMIT_MSG"
echo "  PATHS        : ${PATHS[*]}"
echo

if [ "${PATHOME_DRY_RUN:-0}" = "1" ]; then
  echo "PATHOME_DRY_RUN=1 — would run:"
  echo "  git add -f ${PATHS[*]}"
  echo "  git commit -m \"$COMMIT_MSG\""
  echo "  git push $GIT_REMOTE $GIT_BRANCH"
  exit 0
fi

# Stage. -f so .gitignore'd large artifacts (e.g. data caches) still
# go through when explicitly requested.
echo "[1/3] git add"
git add -f -- ${PATHS[@]} 2>/dev/null || true

# Commit (no-op if nothing changed).
echo
echo "[2/3] git commit"
if git diff --cached --quiet; then
  echo "  nothing to commit"
  exit 0
fi
git commit -m "$COMMIT_MSG"

# Push.
echo
echo "[3/3] git push $GIT_REMOTE $GIT_BRANCH"
git push "$GIT_REMOTE" "$GIT_BRANCH"

echo
echo "push done. HEAD: $(git rev-parse --short HEAD)"
