#!/bin/bash
#SBATCH --job-name=pathome_phase2_traces
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/work/mech-ai-scratch/tirtho/PlantSwarm
#SBATCH --output=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_phase2_traces-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_phase2_traces-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Phase 2: Generate PlantSwarm routing traces against the seeded PathomeDB
# ============================================================================
# A100 + vLLM (Qwen2.5-VL-7B). Each trace is appended with fsync, so a
# walltime kill is recoverable — re-running picks up where it left off.
#
# Volume:
#   3,388 trace seeds × 30 runs = 101,640 traces
#   Wall: ~36-50 h on a single A100 with autogen_swarm + vLLM batching.
#
# Override at submit time:
#   PATHOME_DB_DIR=artifacts/pathome_v1_seed sbatch scripts/submit_pathome_phase2_traces.sh
#   PATHOME_OUT_DIR=results/bugwood_seed     sbatch scripts/submit_pathome_phase2_traces.sh
#
# Output:
#   $PATHOME_OUT_DIR/traces/plantswarm_traces.jsonl   (with bugwood_meta per row)
# ============================================================================

set -e
echo "================================"
echo "Phase 2: PlantSwarm traces"
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
nvidia-smi || true
echo "================================"

module load python cuda/11.8
source /work/mech-ai-scratch/tirtho/PlantSwarm/.venv/bin/activate

CONFIG="${PATHOME_CONFIG:-configs/bugwood_pathome.yaml}"
DB_DIR="${PATHOME_DB_DIR:-artifacts/pathome_v1_seed}"
OUT_DIR="${PATHOME_OUT_DIR:-results/bugwood_seed}"

mkdir -p logs "$OUT_DIR/traces"

# ----------------------------------------------------------------------------
# Boot vLLM in-job (Qwen2.5-VL-7B)
# ----------------------------------------------------------------------------
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
    echo "vLLM ready after ${i}*5s"; break
  fi
  sleep 5
done

# ----------------------------------------------------------------------------
# Patch config so output goes under $OUT_DIR (lets us split seed vs enhanced runs)
# ----------------------------------------------------------------------------
python -c "
import sys, yaml, tempfile, subprocess, os
cfg = yaml.safe_load(open(sys.argv[1]))
cfg['output']['results_dir'] = sys.argv[2]
cfg['output']['traces_dir']  = os.path.join(sys.argv[2], 'traces')
cfg.setdefault('pathome', {})['load_dir'] = sys.argv[3]
with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as tf:
    yaml.safe_dump(cfg, tf); patched = tf.name
ret = subprocess.call([
    'python', 'scripts/run_pathome_traces.py',
    '--config', patched,
    '--orchestrator', 'autogen_swarm',
    '--pathome-dir', sys.argv[3],
])
os.unlink(patched); sys.exit(ret)
" "$CONFIG" "$OUT_DIR" "$DB_DIR"

echo
echo "Phase 2 complete: $(date)"
echo "Output: $OUT_DIR/traces/plantswarm_traces.jsonl"
