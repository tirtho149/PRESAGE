#!/bin/bash
#SBATCH --job-name=pathome_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/work/mech-ai-scratch/tirtho/PlantSwarm
#SBATCH --output=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_pipeline-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_pipeline-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Pathome end-to-end pipeline (paper pathome_final, EMNLP 2026)
#   1) Build PathomeDB from Bugwood records
#   2) Generate 5,460 PlantSwarm routing traces (30 runs/image x 182 images)
#   3) Train OBSERVE: Phase A Decision Transformer, then Phase B GRPO
#   4) Evaluate on full PlantVillage and full PlantWild
#
# Each stage appends to disk with fsync, so a SLURM walltime kill is safe.
# Re-running the script picks up where it left off.
# ============================================================================

set -e
echo "================================"
echo "Pathome pipeline"
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
nvidia-smi || true
echo "================================"

module load python cuda/11.8
source /work/mech-ai-scratch/tirtho/PlantSwarm/.venv/bin/activate
mkdir -p logs results artifacts/pathome_v1 observe/checkpoints

CONFIG=configs/bugwood_pathome.yaml
PV_CONFIG=configs/plantvillage_full_eval.yaml
PW_CONFIG=configs/plantwild_full_eval.yaml

# Boot vLLM in-job for the swarm (Qwen2.5-VL-7B)
echo "Booting vLLM..."
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --trust-remote-code \
  > logs/vllm-${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT
for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "vLLM ready (after ${i}*5s)"; break
  fi
  sleep 5
done

# Stage 1 — PathomeDB
echo
echo "[1/4] Building PathomeDB..."
python scripts/build_pathome.py --config $CONFIG

# Stage 2 — PlantSwarm traces on Bugwood (30 runs x 182 images)
echo
echo "[2/4] Generating PlantSwarm traces..."
python scripts/run_pathome_traces.py --config $CONFIG \
  --pathome-dir artifacts/pathome_v1

# Stage 3 — OBSERVE training (Phase A then Phase B)
echo
echo "[3/4] Training OBSERVE..."
python scripts/train_observe_pathome.py --config $CONFIG --phase both

# Stage 4 — held-out evaluation on full PlantVillage and full PlantWild
echo
echo "[4/4] Evaluating on full PV..."
python scripts/evaluate_pathome.py --config $PV_CONFIG \
  --observe-ckpt observe/checkpoints/observe_grpo_epoch_10.pt \
  --unseen-classes "" || echo "  PV eval failed (continuing)"

echo
echo "[4/4] Evaluating on full PW..."
python scripts/evaluate_pathome.py --config $PW_CONFIG \
  --observe-ckpt observe/checkpoints/observe_grpo_epoch_10.pt || echo "  PW eval failed"

echo
echo "Pipeline complete: $(date)"
