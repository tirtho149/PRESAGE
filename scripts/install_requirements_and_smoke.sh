#!/usr/bin/env bash
# Install PlantSwarm dependencies into the default GPU micromamba prefix, then run smoke_test.sh.
#
# Default prefix (ISU Nova): /work/mech-ai-scratch/tirtho/gpu_env
# Override:
#   PLANTSWARM_GPU_ENV=/other/prefix bash scripts/install_requirements_and_smoke.sh
#
# Optional:
#   INSTALL_TFDS=1   — also pip install -r requirements-tfds.txt if present
#   SKIP_INSTALL=1   — only run smoke (deps assumed installed)
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shellcheck source=slurm/env_gpu_defaults.sh
source "${ROOT}/scripts/slurm/env_gpu_defaults.sh"

PY="${PYTHON_BIN}"
echo "[install_requirements_and_smoke] ENV_PATH=${ENV_PATH}"
echo "[install_requirements_and_smoke] using: $PY ($("$PY" --version 2>&1))"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  echo "[install] upgrading pip…"
  "$PY" -m pip install -U pip wheel

  echo "[install] requirements.txt…"
  "$PY" -m pip install -r "${ROOT}/requirements.txt"

  if [[ "${INSTALL_TFDS:-0}" == "1" ]] && [[ -f "${ROOT}/requirements-tfds.txt" ]]; then
    echo "[install] requirements-tfds.txt (INSTALL_TFDS=1)…"
    "$PY" -m pip install -r "${ROOT}/requirements-tfds.txt"
  fi
else
  echo "[install] SKIP_INSTALL=1 — skipping pip install"
fi

export PYTHON_BIN="$PY"

echo "[smoke] scripts/smoke_test.sh with PYTHON_BIN=${PYTHON_BIN}"
bash "${ROOT}/scripts/smoke_test.sh"

echo "[install_requirements_and_smoke] OK"
