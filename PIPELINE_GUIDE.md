# Full Pipeline Execution Guide

This guide walks through running the complete PlantSwarm + OBSERVE pipeline and syncing results to the paper.

## Prerequisites

1. **vLLM Server Running**
```bash
# Terminal 1: Start vLLM (leave running)
python -m vllm.entrypoints.openai_api_server \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --port 8000
# Verify: curl http://localhost:8000/v1/models
```

2. **Dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-tfds.txt
```

---

## Quick Start (5 Images)

For testing/smoke tests:
```bash
# Everything in one command (all phases)
bash scripts/run_full_pipeline.sh --subset 5

# Or individual phases:
python scripts/run_plantswarm.py --config configs/plant_village_tfds.yaml --subset 5
python scripts/train_observe.py --traces results/plant_village_tfds/traces/plantswarm_traces.jsonl --epochs 2 --output observe/checkpoints/observe_smoke.pt
python scripts/sync_latex_metrics.py --results-dir results/plant_village_tfds --latex-dir plantswarm/latex
```

**Output:**
- Routing traces: `results/plant_village_tfds/traces/plantswarm_traces.jsonl`
- OBSERVE model: `observe/checkpoints/observe_smoke.pt`
- LaTeX metrics: `plantswarm/latex/auto_*.tex`

---

## Full Pipeline (10,000 Images)

### Phase 1: Generate PlantSwarm Routing Traces (12-18 hours)

```bash
python scripts/run_plantswarm.py \
  --config configs/plant_village_tfds.yaml
  # --subset 10000 (optional, default is full dataset)
```

**What it does:**
- Runs 5-agent VLM swarm on 10,000 PlantVillage images
- Generates routing traces with agent paths, predictions, confidence levels
- Computes metrics: accuracy, ECE, TPCP, routing statistics

**Output files:**
```
results/plant_village_tfds/
├── plantswarm_metrics.json        # Main results (T1-T5, ECE, TPCP)
├── plantswarm_predictions.jsonl   # Per-image predictions
├── traces/
│   └── plantswarm_traces.jsonl    # Routing traces (for OBSERVE training)
```

**Key metrics to check:**
```bash
cat results/plant_village_tfds/plantswarm_metrics.json | jq '.T3'
# Expected: macro_f1 ≈ 0.89, ece ≈ 0.08, tpcp ≈ 650
```

---

### Phase 2: Run Experimental Comparisons (2-3 hours)

All comparison scripts can run in parallel:

```bash
# Option A: Run all at once
bash scripts/run_full_pipeline.sh --subset 100  # Or omit --subset for full run

# Option B: Run individually
python scripts/run_baselines.py --config configs/plant_village_tfds.yaml
python scripts/run_ablations.py --config configs/plant_village_tfds.yaml
python scripts/run_calibration.py --config configs/plant_village_tfds.yaml \
  --predictions results/plant_village_tfds/plantswarm_predictions.jsonl
python scripts/run_routing_analysis.py --config configs/plant_village_tfds.yaml
```

**Output files:**
```
results/plant_village_tfds/
├── baseline_results.json          # 8 baseline method comparisons
├── ablation_metrics_*.json        # 6 architectural ablations
├── calibration_report.json        # ECE, temp scaling, conformal
├── routing_analysis.json          # P1-P4 falsifiable predictions
```

---

### Phase 3: Train OBSERVE (4-6 hours on A100)

```bash
python scripts/train_observe.py \
  --traces results/plant_village_tfds/traces/plantswarm_traces.jsonl \
  --output observe/checkpoints/observe_final.pt \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4
```

**What it does:**
- Loads 8,000 routing traces from Phase 1
- Fine-tunes Qwen2.5-VL-3B with LoRA (r=16, α=32)
- Trains for 50 epochs with multi-task loss
- Saves best model by validation ECE

**Output files:**
```
observe/checkpoints/
├── observe_final.pt               # Model weights (trained LoRA + heads)
└── training_history.json          # Loss curves, metrics per epoch
```

**Monitor training:**
```bash
# Watch loss curves in real-time
tail -f observe/checkpoints/training_history.json | jq '.train_loss[-5:]'
```

---

### Phase 4: Evaluate OBSERVE on PlantWild (OOD)

PlantWild requires separate setup (18,000 wild images):

```bash
# Generate PlantWild traces first
python scripts/run_plantswarm.py --config configs/plantwild_hf.yaml

# Evaluate OBSERVE on OOD data
python scripts/evaluate_observe.py \
  --model observe/checkpoints/observe_final.pt \
  --traces results/plantwild/traces/plantswarm_traces.jsonl \
  --output results/plantwild/observe_evaluation.json
```

**Expected results:**
- OOD ECE: ~0.16 (52% improvement over PlantSwarm's 0.33)
- Overconfidence detection F1: ~0.81
- Escalation prediction F1: ~0.84

---

### Phase 5: Sync Metrics to LaTeX Paper

**Automatic (included in full pipeline):**
```bash
python scripts/sync_latex_metrics.py \
  --results-dir results/plant_village_tfds \
  --latex-dir plantswarm/latex
```

**What it does:**
- Reads all JSON results files
- Generates auto_*.tex fragments with actual metrics
- Fills in Table 1 (PlantSwarm results)
- Fills in Table 2 (OBSERVE results)
- Fills in inline macros for paper text

**Output files:**
```
plantswarm/latex/auto/
├── auto_metrics.tex                    # Inline \newcommand{}{} macros
├── auto_table_main_results.tex         # Table 1: PlantSwarm vs baselines
├── auto_table_observe_results.tex      # Table 2: OBSERVE results
├── auto_table_ablation_results.tex     # Table 3: Ablation study
├── auto_table_predictions.tex          # Table 4: P1-P4 predictions
└── auto_table_mechanisms.tex           # Table 5: Context mechanisms
```

---

## Compile Paper with Synced Metrics

After Phase 5, compile the paper:

```bash
# Using latexmk (recommended)
cd plantswarm/latex
latexmk -pdf acl_latex.tex

# Or pdflatex directly
pdflatex -interaction=nonstopmode acl_latex.tex
pdflatex -interaction=nonstopmode acl_latex.tex  # Run twice for TOC
```

**Output:** `plantswarm/latex/acl_latex.pdf` (with all metrics synced)

---

## Practical Workflows

### Smoke Test Everything (15 minutes)
```bash
bash scripts/run_full_pipeline.sh --subset 5
# Outputs: paper with placeholder metrics, OBSERVE trained on 5 traces
```

### Fast Iteration (Reuse PlantSwarm Traces)

If you have `plantswarm_traces.jsonl` already:

```bash
# Skip Phase 1, just train OBSERVE
python scripts/train_observe.py \
  --traces results/plant_village_tfds/traces/plantswarm_traces.jsonl \
  --epochs 10 --batch-size 4  # Fast test

# Sync and recompile
python scripts/sync_latex_metrics.py --results-dir results/plant_village_tfds --latex-dir plantswarm/latex
cd plantswarm/latex && latexmk -pdf acl_latex.tex
```

### Partial Experiments (Debug Mode)

Run only the expensive Phase 1, skip Phase 2:

```bash
python scripts/run_plantswarm.py --config configs/plant_village_tfds.yaml --subset 100
# No baselines/ablations, just traces

python scripts/train_observe.py --traces results/plant_village_tfds/traces/plantswarm_traces.jsonl --epochs 5
python scripts/sync_latex_metrics.py --results-dir results/plant_village_tfds --latex-dir plantswarm/latex
```

---

## Troubleshooting

### vLLM Server Not Responding
```bash
# Check if running
curl http://localhost:8000/v1/models

# Restart
pkill -f vllm
python -m vllm.entrypoints.openai_api_server \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --gpu-memory-utilization 0.8 --port 8000
```

### OBSERVE Training OOM
```bash
# Reduce batch size
python scripts/train_observe.py \
  --traces results/plant_village_tfds/traces/plantswarm_traces.jsonl \
  --batch-size 4  # Default is 8
```

### Missing Routing Traces
```bash
# Check file
ls -lh results/plant_village_tfds/traces/plantswarm_traces.jsonl

# If missing, run Phase 1
python scripts/run_plantswarm.py --config configs/plant_village_tfds.yaml
```

### LaTeX Compilation Error
```bash
# Ensure auto files exist
ls plantswarm/latex/auto/

# If missing, run sync
python scripts/sync_latex_metrics.py --results-dir results/plant_village_tfds --latex-dir plantswarm/latex

# Clean and rebuild
cd plantswarm/latex && latexmk -C && latexmk -pdf acl_latex.tex
```

---

## Expected Results (Paper Targets)

### PlantVillage (Controlled)
- T3 F1: **89.2%** (Easy), **84.3%** (Hard)
- ECE: **0.08** (before temp scaling)
- TPCP: **650 tokens/correct prediction**

### OBSERVE vs PlantSwarm
| Metric | Seen (PlantVillage) | OOD (PlantWild) |
|--------|---|---|
| **ECE** | 0.11 | 0.16 |
| **F1** | 0.89 | 0.84 |
| **Inference tokens** | ~700 | ~700 |
| **Reduction** | 6× vs PlantSwarm | 6× vs PlantSwarm |

### Falsifiable Predictions
- **P1**: Path length ↔ entropy ρ ≈ +0.48
- **P2**: Backtrack improves PathogenAgent +9 F1
- **P3**: Early termination +12 F1
- **P4**: OOD ECE 0.16 (52% vs baselines' 0.33)

---

## Detailed Phase Breakdown

### Phase 1: Understanding `plantswarm_traces.jsonl`

Each trace is a line-delimited JSON object:
```json
{
  "image_id": "plantvillage_00042",
  "path": ["MorphologyAgent", "SymptomAgent", "PathogenAgent", "SeverityAgent", "DiagnosisAgent"],
  "path_length": 5,
  "backtrack_count": 0,
  "backtrack_resolved": false,
  "early_terminated": false,
  "contradiction_resolved": false,
  "total_tokens": 2847,
  "final_predictions": {
    "T1": "symptom_complex",
    "T2": "Fungal",
    "T3": "Late Blight",
    "T4": "Moderate",
    "T5": "Tomato"
  },
  "ground_truth": {
    "T1": "symptom_complex",
    "T2": "Fungal",
    "T3": "Late Blight",
    "T4": "Moderate",
    "T5": "Tomato"
  },
  "context_buffer": [...]  // Full reasoning history
}
```

OBSERVE training extracts:
- `path_length` → epistemic uncertainty heuristic
- `backtrack_resolved` → aleatoric difficulty
- `contradiction_resolved` → confidence calibration
- Next agent from `path[1]`

### Phase 3: Training Loop Details

LoRA fine-tuning parameters:
```python
LoRA:
  r = 16
  alpha = 32
  target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
  dropout = 0.05

Optimizer:
  AdamW, lr = 1e-4
  weight_decay = 0.01
  warmup_steps = 500
  decay = cosine

Loss weights (Eq. 2 in paper):
  routing = 1.0
  calibration = 0.4
  consistency = 0.2
  belief = 0.2
```

---

## Advanced: Custom Configs

To run with different parameters, edit/create config YAML:

```yaml
# configs/custom.yaml
data:
  tfds_name: "plant_village"
  tfds_split: "train"
  tfds_max_examples: 5000  # Custom size
  image_col: "image_bytes"

model:
  backbone: "Qwen/Qwen2.5-VL-3B-Instruct"
  vllm_base_url: "http://localhost:8000/v1"
  temperature: 0.0

routing:
  orchestrator: "autogen_swarm"
  Tmax: 15
  allow_backtrack: true
  # ... rest of config
```

Then run:
```bash
python scripts/run_plantswarm.py --config configs/custom.yaml
```

---

## Reproducibility

All results should be reproducible with fixed seeds:

```bash
# In scripts, seed is set at start
export PYTHONHASHSEED=42
python scripts/run_plantswarm.py --config configs/plant_village_tfds.yaml

# Check it's in logs
grep "seed" results/plant_village_tfds/plantswarm_metrics.json
```

---

## Citation

If you use this pipeline:

```bibtex
@inproceedings{observe2026,
  title={Why Ask When You Can Observe? A Vision-Language-Action Model for Epistemic Action Selection in Multi-Agent Crop Disease Diagnosis},
  author={[Authors]},
  booktitle={Proceedings of EMNLP 2026},
  year={2026}
}
```

---

**Last Updated:** 2026-05-01  
**Tested on:** Python 3.10+, CUDA 11.8+, vLLM 0.4+
