#!/usr/bin/env bash
# Source this file to put the micromamba / prefix env on PATH (same as all Slurm jobs).
#
#   source scripts/use_gpu_env.sh
#   python scripts/run_plantswarm.py --config configs/default.yaml --subset 5
#
# Override the prefix:
#   export PLANTSWARM_GPU_ENV=/path/to/env && source scripts/use_gpu_env.sh

_U="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=slurm/env_gpu_defaults.sh
source "${_U}/slurm/env_gpu_defaults.sh"
