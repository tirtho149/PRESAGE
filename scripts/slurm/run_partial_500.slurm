#!/bin/bash
# 500-image partial run + LaTeX sync.
#SBATCH --job-name=plantswarm_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --output=/work/mech-ai-scratch/tirtho/CyAg/logs/plantswarm-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/CyAg/logs/plantswarm-%j.err
#SBATCH --chdir=/work/mech-ai-scratch/tirtho/CyAg/PlantSwarm
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
mkdir -p /work/mech-ai-scratch/tirtho/CyAg/logs
export SUBSET="${SUBSET:-500}"
export ORCHESTRATOR="${ORCHESTRATOR:-autogen_swarm}"
export CONFIG_PATH="${CONFIG_PATH:-configs/cyag_directory.yaml}"
WORKDIR="${WORKDIR:-/work/mech-ai-scratch/tirtho/CyAg/PlantSwarm}"
ENV_PATH="${ENV_PATH:-/work/mech-ai-scratch/tirtho/gpu_env}"
export CYAG_IMAGE_ROOT="${CYAG_IMAGE_ROOT:-/work/mech-ai-scratch/tirtho/CyAg/Curated_Dataset/Images}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
export RESULTS_DIR="${RESULTS_DIR:-${WORKDIR}/results/partial500_${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${WORKDIR}/logs" "${RESULTS_DIR}"
cd "${WORKDIR}"
export CYAG_IMAGE_ROOT

if [[ -f "${ENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ENV_PATH}/bin/activate"
  PYTHON_BIN="${ENV_PATH}/bin/python"
elif [[ -x "${ENV_PATH}/bin/python" ]]; then
  PYTHON_BIN="${ENV_PATH}/bin/python"
fi
export PYTHON_BIN

echo "PARTIAL-500  SUBSET=${SUBSET}  ORCHESTRATOR=${ORCHESTRATOR}  RESULTS_DIR=${RESULTS_DIR}"
export ROUTING_SUBSET="${ROUTING_SUBSET:-${SUBSET}}"
bash "${WORKDIR}/scripts/run_experiment_bundle.sh"
