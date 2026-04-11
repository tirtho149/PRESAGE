#!/bin/bash
#SBATCH --job-name=plantswarm-all-tests
#SBATCH --output=/work/mech-ai-scratch/tirtho/CyAg/PlantSwarm/logs/%x-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/CyAg/PlantSwarm/logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=nova
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --chdir=/work/mech-ai-scratch/tirtho/CyAg/PlantSwarm

set -euo pipefail

# -------------------------------
# CyAg / scratch layout (override with env if needed)
# -------------------------------
WORKDIR="${WORKDIR:-/work/mech-ai-scratch/tirtho/CyAg/PlantSwarm}"
ENV_PATH="${ENV_PATH:-/work/mech-ai-scratch/tirtho/gpu_env}"
# Root for leaf images (parquet image_path values are often relative to this)
CYAG_IMAGE_ROOT="${CYAG_IMAGE_ROOT:-/work/mech-ai-scratch/tirtho/CyAg/Curated_Dataset/Images}"

CONFIG_PATH="${CONFIG_PATH:-configs/cyag_directory.yaml}"
ORCHESTRATOR="${ORCHESTRATOR:-autogen_swarm}"   # must be autogen_swarm (classic rejected in run_plantswarm)
SUBSET="${SUBSET:-20}"                           # smoke-friendly subset
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-results/slurm_${RUN_ID}}"

mkdir -p "${WORKDIR}/logs" "${RESULTS_DIR}"
cd "${WORKDIR}"

export CYAG_IMAGE_ROOT

# Keep caches local to this workspace.
export XDG_CACHE_HOME="${WORKDIR}/.cache"
export HF_HOME="${WORKDIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${WORKDIR}/.cache/huggingface/transformers"
export HUGGINGFACE_HUB_CACHE="${WORKDIR}/.cache/huggingface/hub"
export TORCH_HOME="${WORKDIR}/.cache/torch"
export PIP_CACHE_DIR="${WORKDIR}/.cache/pip"
export WANDB_DIR="${WORKDIR}/.cache/wandb"
export TMPDIR="${WORKDIR}/.cache/tmp"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" \
         "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${WANDB_DIR}"

echo "==============================="
echo "PlantSwarm Slurm full test run"
echo "==============================="
echo "HOSTNAME         : ${HOSTNAME:-unknown}"
echo "JOB_ID           : ${SLURM_JOB_ID:-local}"
echo "WORKDIR          : ${WORKDIR}"
echo "ENV_PATH         : ${ENV_PATH}"
echo "CYAG_IMAGE_ROOT  : ${CYAG_IMAGE_ROOT}"
echo "CONFIG_PATH      : ${CONFIG_PATH}"
echo "ORCHESTRATOR     : ${ORCHESTRATOR}"
echo "SUBSET           : ${SUBSET}"
echo "RESULTS_DIR      : ${RESULTS_DIR}"
echo

if [[ -f "${ENV_PATH}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ENV_PATH}/bin/activate"
  PYTHON_BIN="${ENV_PATH}/bin/python"
elif [[ -x "${ENV_PATH}/bin/python" ]]; then
  PYTHON_BIN="${ENV_PATH}/bin/python"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[FATAL] Python binary not found: ${PYTHON_BIN}"
  exit 1
fi

# Optional server health check (non-fatal)
if command -v curl >/dev/null 2>&1; then
  echo
  echo "Checking vLLM endpoint from config (best-effort)..."
  export CONFIG_PATH
  BASE_URL="$(${PYTHON_BIN} - <<'PY'
import os
import yaml
cfg = yaml.safe_load(open(os.environ["CONFIG_PATH"]))
print(cfg["model"]["vllm_base_url"].rstrip("/"))
PY
)"
  if curl -sS -m 5 "${BASE_URL}/models" >/dev/null 2>&1; then
    echo "vLLM endpoint reachable: ${BASE_URL}"
  else
    echo "[WARN] vLLM endpoint not reachable: ${BASE_URL}"
    echo "       Jobs that require model inference may fail."
  fi
fi

export ORCHESTRATOR SUBSET CONFIG_PATH RESULTS_DIR PYTHON_BIN
bash "${WORKDIR}/scripts/run_experiment_bundle.sh"

echo
echo "All steps completed successfully."
echo "Artifacts: ${RESULTS_DIR}"
