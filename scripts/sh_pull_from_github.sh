#!/bin/bash
# ============================================================================
# scripts/sh_pull_from_github.sh           GitHub → LOCAL or NOVA
# ============================================================================
# Explicit hand-off: fast-forward pull the latest state from GitHub.
# Use this between step scripts when you want to pull manually instead
# of letting the next step's built-in `git pull --ff-only` at its start
# run. Also useful on Nova right after a LOCAL push (steps 0/1/3) to
# refresh before starting an sbatch job.
#
# It REFUSES to overwrite local changes. If you have uncommitted edits
# (other than untracked files), the script aborts and tells you what
# to do.
#
# Knobs
#   GIT_REMOTE   default origin
#   GIT_BRANCH   default main
# ============================================================================
set -euo pipefail

REPO_ROOT="${PATHOME_REPO:-$(pwd)}"
cd "$REPO_ROOT"

GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"

echo "================================================================="
echo " pull from GitHub"
echo "================================================================="
echo "  GIT_REMOTE   : $GIT_REMOTE"
echo "  GIT_BRANCH   : $GIT_BRANCH"
echo "  HEAD before  : $(git rev-parse --short HEAD)"
echo

# Refuse if there are unstaged or staged changes on tracked files.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: you have local modifications to tracked files."
  echo "       Either commit them, or 'git stash' them, before pulling."
  echo
  echo "Local diff (top 20 paths):"
  git status --porcelain | head -n 20
  exit 2
fi

echo "[1/2] git fetch $GIT_REMOTE $GIT_BRANCH"
git fetch "$GIT_REMOTE" "$GIT_BRANCH"

echo
echo "[2/2] git pull --ff-only $GIT_REMOTE $GIT_BRANCH"
git pull --ff-only "$GIT_REMOTE" "$GIT_BRANCH"

echo
echo "pull done. HEAD now: $(git rev-parse --short HEAD)"
