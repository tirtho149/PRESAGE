#!/usr/bin/env bash
# Single source of truth for the micromamba / prefix GPU environment on ISU Nova.
# Sourced by every *.slurm driver, scripts/use_gpu_env.sh, and install_requirements_and_smoke.sh.
#
# Override for a different machine:
#   export PLANTSWARM_GPU_ENV=/path/to/prefix   # or ENV_PATH
# before sourcing this file or submitting Slurm.

: "${PLANTSWARM_GPU_ENV_DEFAULT:=/work/mech-ai-scratch/tirtho/gpu_env}"
export PLANTSWARM_GPU_ENV="${PLANTSWARM_GPU_ENV:-${ENV_PATH:-$PLANTSWARM_GPU_ENV_DEFAULT}}"
export ENV_PATH="${ENV_PATH:-$PLANTSWARM_GPU_ENV}"

if [[ -f "${ENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ENV_PATH}/bin/activate"
fi
if [[ ! -x "${ENV_PATH}/bin/python" ]]; then
  echo "[FATAL] PlantSwarm GPU env: expected python at ${ENV_PATH}/bin/python (set PLANTSWARM_GPU_ENV or ENV_PATH if different)." >&2
  exit 1
fi
export PYTHON_BIN="${ENV_PATH}/bin/python"
export PATH="${ENV_PATH}/bin:${PATH}"
