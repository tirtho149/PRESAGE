#!/bin/bash
# Array job: AutoGen Swarm × (100 | 500) images (two subset sizes).
# Submit: sbatch scripts/slurm/run_matrix_array.slurm
# Override: sbatch --export=ALL,CONFIG_PATH=configs/cyag_directory.yaml scripts/slurm/run_matrix_array.slurm
#
#SBATCH --job-name=plantswarm_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --output=/work/mech-ai-scratch/tirtho/CyAg/logs/plantswarm-%A_%a.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/CyAg/logs/plantswarm-%A_%a.err
#SBATCH --chdir=/work/mech-ai-scratch/tirtho/CyAg/PlantSwarm
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-1

set -euo pipefail
mkdir -p /work/mech-ai-scratch/tirtho/CyAg/logs
_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env_gpu_defaults.sh
source "${_SLURM_DIR}/env_gpu_defaults.sh"

WORKDIR="${WORKDIR:-/work/mech-ai-scratch/tirtho/CyAg/PlantSwarm}"
export CONFIG_PATH="${CONFIG_PATH:-configs/cyag_directory.yaml}"
export CYAG_IMAGE_ROOT="${CYAG_IMAGE_ROOT:-/work/mech-ai-scratch/tirtho/CyAg/Curated_Dataset/Images}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) export ORCHESTRATOR="autogen_swarm"; export SUBSET="100" ;;
  1) export ORCHESTRATOR="autogen_swarm"; export SUBSET="500" ;;
  *)
    echo "[FATAL] Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

RUN_ID="$(date +%Y%m%d_%H%M%S)_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export RESULTS_DIR="${WORKDIR}/results/matrix_autogen_${SUBSET}_${RUN_ID}"
export ROUTING_SUBSET="${SUBSET}"

mkdir -p "${WORKDIR}/logs" "${RESULTS_DIR}"
cd "${WORKDIR}"
export CYAG_IMAGE_ROOT

echo "MATRIX task=${SLURM_ARRAY_TASK_ID} ENV_PATH=${ENV_PATH} PYTHON_BIN=${PYTHON_BIN} ORCHESTRATOR=${ORCHESTRATOR} SUBSET=${SUBSET}"
echo "RESULTS_DIR=${RESULTS_DIR}"
bash "${WORKDIR}/scripts/run_experiment_bundle.sh"
