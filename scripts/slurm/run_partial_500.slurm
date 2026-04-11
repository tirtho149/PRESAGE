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
_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env_gpu_defaults.sh
source "${_SLURM_DIR}/env_gpu_defaults.sh"

export SUBSET="${SUBSET:-500}"
export ORCHESTRATOR="${ORCHESTRATOR:-autogen_swarm}"
export CONFIG_PATH="${CONFIG_PATH:-configs/cyag_directory.yaml}"
WORKDIR="${WORKDIR:-/work/mech-ai-scratch/tirtho/CyAg/PlantSwarm}"
export CYAG_IMAGE_ROOT="${CYAG_IMAGE_ROOT:-/work/mech-ai-scratch/tirtho/CyAg/Curated_Dataset/Images}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
export RESULTS_DIR="${RESULTS_DIR:-${WORKDIR}/results/partial500_${RUN_ID}}"

mkdir -p "${WORKDIR}/logs" "${RESULTS_DIR}"
cd "${WORKDIR}"
export CYAG_IMAGE_ROOT

echo "PARTIAL-500  ENV_PATH=${ENV_PATH}  PYTHON_BIN=${PYTHON_BIN}  SUBSET=${SUBSET}  ORCHESTRATOR=${ORCHESTRATOR}  RESULTS_DIR=${RESULTS_DIR}"
export ROUTING_SUBSET="${ROUTING_SUBSET:-${SUBSET}}"
bash "${WORKDIR}/scripts/run_experiment_bundle.sh"
