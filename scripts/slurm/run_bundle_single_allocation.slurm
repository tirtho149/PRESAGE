#!/bin/bash
# One GPU allocation: start vLLM on-node, then full experiment bundle (same node).
# Edit paths, mail, and VLLM_MODEL to match model.backbone in CONFIG_PATH.
#
# Submit:
#   sbatch scripts/slurm/run_bundle_single_allocation.slurm
#
# Override at submit time:
#   sbatch --export=ALL,SUBSET=100,VLLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct scripts/slurm/run_bundle_single_allocation.slurm

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

WORKDIR="${WORKDIR:-/work/mech-ai-scratch/tirtho/CyAg/PlantSwarm}"
export CONFIG_PATH="${CONFIG_PATH:-configs/cyag_directory.yaml}"
export SUBSET="${SUBSET:-}"
export ORCHESTRATOR="${ORCHESTRATOR:-autogen_swarm}"
export ROUTING_SUBSET="${ROUTING_SUBSET:-${SUBSET:-500}}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
export RESULTS_DIR="${RESULTS_DIR:-${WORKDIR}/results/single_alloc_${RUN_ID}}"
# Must match model.backbone in CONFIG_PATH (HF id passed to vllm serve)
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"

cd "${WORKDIR}"

export CYAG_IMAGE_ROOT="${CYAG_IMAGE_ROOT:-/work/mech-ai-scratch/tirtho/CyAg/Curated_Dataset/Images}"

echo "SINGLE ALLOCATION  ENV_PATH=${ENV_PATH}  PYTHON_BIN=${PYTHON_BIN}  VLLM_MODEL=${VLLM_MODEL}  CONFIG_PATH=${CONFIG_PATH}  RESULTS_DIR=${RESULTS_DIR}"
bash scripts/run_single_allocation.sh
