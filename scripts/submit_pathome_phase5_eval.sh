#!/bin/bash
#SBATCH --job-name=pathome_phase5_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=nova
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/work/mech-ai-scratch/tirtho/PlantSwarm
#SBATCH --output=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_phase5_eval-%j.out
#SBATCH --error=/work/mech-ai-scratch/tirtho/PlantSwarm/logs/pathome_phase5_eval-%j.err
#SBATCH --mail-user=tirtho@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# ============================================================================
# Phase 5: Held-out evaluation (PV + PW) for both OBSERVE checkpoints, then
# write the seed-vs-enhanced comparison artifacts (markdown + LaTeX).
# ============================================================================
# Step 5a: evaluate seed-trained OBSERVE on full PV (with --unseen-classes)
# Step 5b: evaluate enhanced-trained OBSERVE on full PV
# Step 5c: evaluate both checkpoints on full PW
# Step 5d: compare_pathome_versions.py emits comparison.{json,md,tex}
#
# All four steps run inside one SLURM allocation. Each eval reuses the same
# vLLM instance booted at the start of the job.
# ============================================================================

set -e
echo "================================"
echo "Phase 5: Eval + Compare"
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
nvidia-smi || true
echo "================================"

module load python cuda/11.8
source /work/mech-ai-scratch/tirtho/PlantSwarm/.venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

mkdir -p logs

PV_CONFIG="${PATHOME_PV_CONFIG:-configs/plantvillage_full_eval.yaml}"
PW_CONFIG="${PATHOME_PW_CONFIG:-configs/plantwild_full_eval.yaml}"
SEED_CKPT="${PATHOME_SEED_CKPT:-observe/checkpoints/seed/observe_grpo_epoch_10.pt}"
ENH_CKPT="${PATHOME_ENHANCED_CKPT:-observe/checkpoints/enhanced/observe_grpo_epoch_10.pt}"
SEED_TRACES="${PATHOME_SEED_TRACES:-results/bugwood_seed/traces/plantswarm_traces.jsonl}"
ENH_TRACES="${PATHOME_ENHANCED_TRACES:-results/bugwood_enhanced/traces/plantswarm_traces.jsonl}"
UNSEEN="${PATHOME_UNSEEN_CLASSES:-}"
RESULTS_BASE="${PATHOME_RESULTS_BASE:-results/pathome_compare}"

# ----------------------------------------------------------------------------
# Boot vLLM once and reuse across all four evaluations.
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

eval_one() {
  local tag="$1" cfg="$2" ckpt="$3" out_subdir="$4"
  local out="$RESULTS_BASE/$out_subdir"
  mkdir -p "$out"
  echo
  echo "── [$tag]  cfg=$cfg  ckpt=$ckpt  → $out ──"
  python -c "
import sys, yaml, tempfile, subprocess, os
cfg = yaml.safe_load(open(sys.argv[1]))
cfg.setdefault('output', {})['results_dir'] = sys.argv[2]
with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as tf:
    yaml.safe_dump(cfg, tf); patched = tf.name
args = ['python', 'scripts/evaluate_pathome.py',
        '--config', patched, '--observe-ckpt', sys.argv[3]]
if sys.argv[4]:
    args += ['--unseen-classes', sys.argv[4]]
ret = subprocess.call(args); os.unlink(patched); sys.exit(ret)
" "$cfg" "$out" "$ckpt" "$UNSEEN"
}

# 5a + 5b: full PlantVillage with seen/unseen slice for both checkpoints
eval_one "seed_PV"     "$PV_CONFIG" "$SEED_CKPT" "seed/pv"
eval_one "enhanced_PV" "$PV_CONFIG" "$ENH_CKPT"  "enhanced/pv"

# 5c: full PlantWild for both checkpoints
eval_one "seed_PW"     "$PW_CONFIG" "$SEED_CKPT" "seed/pw"
eval_one "enhanced_PW" "$PW_CONFIG" "$ENH_CKPT"  "enhanced/pw"

# ----------------------------------------------------------------------------
# 5d: comparison artifact (markdown + LaTeX + json)
# ----------------------------------------------------------------------------
# Use PV eval (richer slices) for the headline before/after table; trace
# inputs come from Phase 2 outputs (seed and enhanced runs).
echo
echo "── building before/after comparison ──"
python scripts/compare_pathome_versions.py \
  --seed-eval     "$RESULTS_BASE/seed/pv/pathome_eval.json" \
  --enhanced-eval "$RESULTS_BASE/enhanced/pv/pathome_eval.json" \
  ${SEED_TRACES:+--seed-traces "$SEED_TRACES"} \
  ${ENH_TRACES:+--enhanced-traces "$ENH_TRACES"} \
  --out-dir       "$RESULTS_BASE"

echo
echo "Phase 5 complete: $(date)"
echo "comparison: $RESULTS_BASE/comparison.{json,md,tex}"
