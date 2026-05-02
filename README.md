# PlantSwarm: Multi-Agent VLM Swarm for Plant Disease Diagnosis

**Paper:** *Why Ask When You Can Observe? A Vision-Language-Action Model for Epistemic Action Selection in Multi-Agent Crop Disease Diagnosis* (EMNLP 2026)

**Core Contribution:** PlantSwarm establishes that **routing behavior** (path length, backtrack decisions, contradiction events) predicts correctness far better than **self-declared confidence** in multi-agent VLM systems. OBSERVE operationalizes this as the first Vision-Language-Action model trained on routing traces, achieving 52% calibration improvement under domain shift with 6× lower inference cost.

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.10+
- NVIDIA GPU (for vLLM inference)
- 50GB disk (for TFDS Plant Village cache)

### Install & Test
```bash
# 1. Clone and setup
cd ObservePlantSwarm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Install TFDS support (for Plant Village)
pip install -r requirements-tfds.txt

# 3. Verify imports
python -c "from agents import *; from plantswarm import *; print('✓ Ready')"
```

---

## 📋 Complete Workflow (5 Phases)

1. **Phase 1:** Generate PlantSwarm routing traces on PlantVillage
2. **Phase 2:** Run experimental comparisons (baselines, ablations, calibration, bias)
3. **Phase 3:** Train OBSERVE model on routing traces
4. **Phase 4:** Evaluate on PlantWild (OOD)
5. **Phase 5:** Build paper with auto-synced metrics

---

### Phase 1: Generate Routing Traces (Training Data)

**Goal:** Run PlantSwarm on PlantVillage (~10,000 images) to generate routing traces for OBSERVE training.

#### Step 1a: Start vLLM Server
On a GPU machine with vLLM installed:
```bash
# Start vLLM serving Qwen3-VL-8B
python -m vllm.entrypoints.openai_api_server \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --port 8000
# Server ready at http://localhost:8000/v1
```

**On Mac/Windows:** Use SSH port forwarding to remote GPU:
```bash
ssh -L 8000:localhost:8000 gpu_machine
# Then use http://localhost:8000/v1 in config
```

#### Step 1b: Run PlantSwarm Training
```bash
# Smoke test (5 images, ~1 min)
python scripts/run_plantswarm.py \
  --config configs/qwen25_vl_3b_smoke.yaml \
  --subset 5

# Full training (10,000 images, ~12-18 hours on 1x A100)
python scripts/run_plantswarm.py \
  --config configs/plant_village_tfds.yaml
```

**Output:** `results/plant_village_tfds/`
- `plantswarm_metrics.json` — accuracy, ECE, TPCP metrics
- `plantswarm_predictions.jsonl` — per-image predictions
- `traces/plantswarm_traces.jsonl` — routing traces (training data for OBSERVE)

---

### Phase 2: Run Experimental Comparisons

#### Step 2a: Baselines (Single-agent, Fixed Chain, Debate, etc.)
```bash
python scripts/run_baselines.py --config configs/plant_village_tfds.yaml
```
Compares PlantSwarm against 8 baselines:
- Zero-shot single VLM
- Chain-of-thought
- Fixed chain (no routing)
- DeeR (two-stage exit)
- Multi-agent debate
- Random, Majority class

**Output:** `results/plant_village_tfds/baseline_results.json`

#### Step 2b: Ablations (Factorial Study)
```bash
python scripts/run_ablations.py --config configs/plant_village_tfds.yaml
```
Tests contribution of routing components (Table 3):
- Fixed Chain (baseline)
- +Context buffer
- +Free routing (no confidence gate)
- +Backtracking
- 3-agent swarm
- Full PlantSwarm

**Output:** `results/plant_village_tfds/ablation_metrics_*.json`

#### Step 2c: Calibration Analysis
```bash
python scripts/run_calibration.py \
  --config configs/plant_village_tfds.yaml \
  --predictions results/plant_village_tfds/plantswarm_predictions.jsonl
```
Analyzes uncertainty quantification:
- ECE before/after temperature scaling
- Reliability diagrams
- Split conformal prediction
- κ calibration (confidence vs. correctness)

**Output:** `results/plant_village_tfds/calibration_report.json`

#### Step 2d: Routing Analysis
```bash
python scripts/run_routing_analysis.py --config configs/plant_village_tfds.yaml
```
Tests falsifiable predictions (P1-P4):
- P1: Path length ↔ entropy correlation
- P2: Backtrack improves confidence
- P3: Early termination accuracy
- P4: OOD behavioral transfer

**Output:** `results/plant_village_tfds/routing_analysis.json`

#### Step 2e: Bias Analysis
```bash
python scripts/run_bias_analysis.py --config configs/plant_village_tfds.yaml
```
Examines demographic parity and confounding effects.

**Output:** `results/plant_village_tfds/bias_analysis.json`

---

### Phase 3: Train OBSERVE (Vision-Language-Action Model)

**Goal:** Train a lightweight epistemic action selector on PlantSwarm routing traces for 6× lower inference cost with 52% better calibration under domain shift.

#### Step 3a: Prepare Training Traces
Ensure you have routing traces from Phase 1:
```bash
# Check traces exist
ls -lh results/plant_village_tfds/traces/plantswarm_traces.jsonl
# Should contain 8,000-10,000 routing traces
```

#### Step 3b: Train OBSERVE Model
```bash
# Smoke test (train on first 100 traces)
python scripts/train_observe.py \
  --traces results/plant_village_tfds/traces/plantswarm_traces.jsonl \
  --output observe/checkpoints/observe_smoke.pt \
  --epochs 2 \
  --batch-size 8

# Full training (~4-6 hours on single A100)
python scripts/train_observe.py \
  --traces results/plant_village_tfds/traces/plantswarm_traces.jsonl \
  --output observe/checkpoints/observe_final.pt \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4
```

**Training Details:**
- **Architecture:** Qwen2.5-VL-3B with LoRA (r=16, α=32, ~56M trainable params)
- **Data:** 8,000 routing traces from PlantSwarm with epistemic/aleatoric labels
- **Loss:** Weighted combination of routing (1.0) + calibration (0.4) + consistency (0.2) + belief (0.2)
- **Output:** `observe/checkpoints/observe_final.pt` (model weights + training history)

**Output:**
- `observe/checkpoints/observe_final.pt` — trained model weights
- `observe/checkpoints/training_history.json` — loss curves and metrics

#### Step 3c: Evaluate OBSERVE on PlantVillage (ID)
```bash
python scripts/evaluate_observe.py \
  --model observe/checkpoints/observe_final.pt \
  --traces results/plant_village_tfds/traces/plantswarm_traces.jsonl \
  --output results/plant_village_tfds/observe_evaluation.json
```

#### Step 3d: Inference with OBSERVE
```python
from observe import OBSERVEInference
from PIL import Image

# Load model
inference = OBSERVEInference("observe/checkpoints/observe_final.pt")

# Single image
image = Image.open("crop.jpg")
action = inference.predict(image, context_text="Prior observations: healthy leaf")

print(f"Next agent: {action.next_agent}")
print(f"Backtrack: {action.backtrack}")
print(f"Epistemic uncertainty: {action.epistemic_uncertainty:.3f}")
print(f"Aleatoric uncertainty: {action.aleatoric_uncertainty:.3f}")
print(f"Confidence: {action.confidence:.3f}")

# Get uncertainty decomposition with recommendations
decomp = inference.get_uncertainty_decomposition(action)
print(f"\nEpistemic: {decomp['epistemic']['recommendation']}")
print(f"Aleatoric: {decomp['aleatoric']['recommendation']}")

# Batch inference
images = [Image.open(f"crop_{i}.jpg") for i in range(10)]
actions = inference.predict_batch(images, batch_size=4)
```

---

### Phase 4: OOD Evaluation (PlantWild)

**Goal:** Evaluate PlantSwarm on wild (uncontrolled) images for domain shift assessment.

```bash
# Smoke test (5 images)
python scripts/run_plantswarm.py \
  --config configs/plantwild_hf.yaml \
  --subset 5

# Full OOD evaluation (~18,000 images)
python scripts/run_plantswarm.py \
  --config configs/plantwild_hf.yaml
```

**Output:** `results/plantwild/`
- `plantswarm_metrics.json` — OOD accuracy, ECE (should be worse than PlantVillage)
- Validates robustness to controlled→wild domain shift

#### Step 4 (Optional): Evaluate OBSERVE on PlantWild (OOD)
```bash
python scripts/evaluate_observe.py \
  --model observe/checkpoints/observe_final.pt \
  --traces results/plantwild/traces/plantswarm_traces.jsonl \
  --output results/plantwild/observe_evaluation.json
```

Should show 52% ECE improvement over prompt-based baselines under domain shift.

---

### Phase 5: Build the Paper

#### Step 5a: Sync Metrics to LaTeX
```bash
python scripts/sync_latex_metrics.py \
  --results-dir results/plant_village_tfds/ \
  --latex-dir plantswarm/latex/ \
  --subset-hint full
```

Converts JSON metrics → TeX table fragments:
- `auto_metrics.tex` — inline macro definitions (ECE, F1, etc.)
- `auto_table_main_results.tex` — Table 4 (PlantSwarm vs baselines)
- `auto_table_ablation_results.tex` — Table 3 (ablations)
- `auto_table_mechanisms.tex` — context buffer mechanisms (RQ5)

#### Step 5b: Compile PDF
```bash
bash scripts/build_latex_pdf.sh \
  --latex-dir plantswarm/latex/ \
  --main-tex acl_latex.tex \
  --results-dir results/plant_village_tfds/
```

Produces:
- `plantswarm/latex/acl_latex.pdf` — paper with latest metrics
- `results/plant_village_tfds/paper_acl_latex.pdf` — copy for results dir

---

## 🔧 Configuration Guide

### configs/plant_village_tfds.yaml (Training)
```yaml
data:
  tfds_name: "plant_village"      # TensorFlow Datasets
  tfds_split: "train"
  tfds_max_examples: 10000        # ~54k available; use subset for testing
  image_col: "image_bytes"        # TFDS provides JPEG bytes

model:
  backbone: "Qwen/Qwen3-VL-8B-Instruct"
  vllm_base_url: "http://localhost:8000/v1"
  temperature: 0.0                # Deterministic routing

routing:
  orchestrator: "autogen_swarm"   # Microsoft AutoGen Swarm
  Tmax: 15                        # Max agents per image
  allow_backtrack: true

output:
  results_dir: "results/plant_village_tfds/"
  save_traces: true              # Required for OBSERVE training
```

### configs/plantwild_hf.yaml (OOD Evaluation)
```yaml
data:
  hf_dataset_id: "rashikahura/plantWild"  # HuggingFace dataset
  image_col: "image_bytes"
  n_images: 18000                 # Full wild dataset
```

---

## 📊 Understanding Results

### plantswarm_metrics.json
```json
{
  "T1": {"macro_f1": 87.5, "ece": 0.11, "tpcp": 720},
  "T2": {"macro_f1": 92.3, "ece": 0.08, "tpcp": 650},
  ...
  "by_benchmark": {
    "plantvillage": {"T2": {"macro_f1": 94.1}, "T3": {"macro_f1": 88.9}},
    "plantwild": {...}  // OOD results
  }
}
```

**Key Metrics:**
- `macro_f1`: F1 score (main accuracy metric)
- `ece`: Expected Calibration Error (0.0=perfect, 1.0=worst)
- `tpcp`: Tokens-per-correct-prediction (efficiency)

### plantswarm_traces.jsonl
Per-image routing trace (training data for OBSERVE):
```json
{
  "image_id": "plantvillage_00042",
  "path": ["MorphologyAgent", "SymptomAgent", "PathogenAgent", "SeverityAgent", "DiagnosisAgent"],
  "path_length": 5,
  "backtrack_count": 0,
  "early_terminated": false,
  "total_tokens": 2847,
  "final_predictions": {"T1": "Blight", "T2": "Fungal", "T3": "Late Blight", ...},
  "ground_truth": {"T1": "Blight", "T2": "Fungal", "T3": "Late Blight", ...}
}
```

---

## 🏗️ Architecture Overview

### 5-Agent Swarm (Routing Strategy)
```
MorphologyAgent (visual grounding only)
         ↓
SymptomAgent (T1: symptom classification)
         ↓
PathogenAgent (T2: pathogen, T3: disease name)
         ↓
SeverityAgent (T4: severity, T5: crop species)
         ↓
DiagnosisAgent (synthesis + final JSON)
```

**Routing Decisions (Algorithm 1):**
- **Low confidence + no backtrack:** → MorphologyAgent (regrounding)
- **High confidence + all tasks complete:** → DiagnosisAgent (early terminate)
- **Medium confidence or pending tasks:** → forward to next agent

### OBSERVE: Vision-Language-Action Model

**Architecture:**
```
Input: Image + Context Text
  ↓
Qwen2.5-VL-3B (frozen, 2.95B params)
  ↓
LoRA Adapter (r=16, α=32, ~50M trainable params)
  ↓
Shared Head (512-dim)
  ├→ Routing Head (5-class softmax)
  ├→ Backtrack Head (binary sigmoid)
  ├→ Epistemic Head (scalar ∈ [0,1])
  ├→ Aleatoric Head (scalar ∈ [0,1])
  ├→ Confidence Head (scalar ∈ [0,1])
  └→ Belief Text (autoregressive from decoder)
```

**Key Outputs:**
- **next_agent:** Which of 5 agents to route to next
- **backtrack:** Whether to backtrack to MorphologyAgent
- **epistemic_uncertainty:** Resolvable ambiguity (improved by more evidence)
- **aleatoric_uncertainty:** Irreducible difficulty (escalate to human)
- **confidence:** Calibrated confidence in prediction [0, 1]
- **belief_state:** Natural language belief about current situation

**Training:**
- **Data:** 8,000-10,000 routing traces from PlantSwarm
- **Loss:** Weighted multi-task (routing 1.0 + calibration 0.4 + consistency 0.2 + belief 0.2)
- **Optimizer:** AdamW with lr=1e-4
- **Time:** ~4-6 hours on single A100 GPU for 50 epochs
- **Hyperparams:** batch_size=8, weight_decay=0.01

**Performance:**
- **Cost:** 700 tokens vs 4,200 for full PlantSwarm (6× reduction)
- **ID Accuracy:** 92% on PlantVillage
- **OOD Calibration:** ECE 0.16 vs 0.33 for baselines (52% improvement)
- **Uncertainty Decomposition:** Actionable epistemic/aleatoric split with human escalation guidance

---

## 📦 Directory Structure

```
PlantSwarm/
├── agents/
│   ├── base_agent.py           # ABC for all agents
│   ├── morphology_agent.py     # Visual grounding
│   ├── symptom_agent.py        # T1
│   ├── pathogen_agent.py       # T2, T3
│   ├── severity_agent.py       # T4, T5
│   └── diagnosis_agent.py      # Synthesis
│
├── observe/                    # Vision-Language-Action model
│   ├── __init__.py             # Module exports
│   ├── model.py                # OBSERVE architecture + LoRA
│   ├── trainer.py              # Training pipeline
│   ├── inference.py            # Deployment/evaluation
│   └── checkpoints/            # Trained model weights (after training)
│
├── plantswarm/
│   ├── pipeline.py             # Core κ-routing orchestrator
│   ├── autogen_pipeline.py     # AutoGen Swarm runtime (default)
│   ├── entropy_pipeline.py     # Entropy-driven routing variant
│   └── latex/                  # Paper source + auto-generated tables
│
├── calibration/
│   ├── ensemble.py             # Confidence-weighted aggregation
│   ├── ece.py                  # Expected Calibration Error
│   ├── temperature_scaling.py  # Post-hoc calibration
│   └── conformal.py            # Prediction sets
│
├── utils/
│   ├── vllm_client.py          # OpenAI-compatible HTTP client
│   ├── metrics.py              # F1, TPCP, McNemar's test
│   ├── routing_trace.py        # Trace I/O & analysis
│   ├── sequence_entropy.py     # Token-level entropy
│   └── hedge_lexicon.py        # Uncertainty signals
│
├── data/
│   ├── loader.py               # Unified PlantDiagBenchLoader
│   ├── tfds_plant_village.py   # TFDS Plant Village backend
│   ├── plantwild_hf.py         # HuggingFace PlantWild backend
│   ├── directory_index.py      # Folder tree backend
│   └── stratifier.py           # Train/cal/test splits
│
├── baselines/                  # 8 baseline implementations
├── ablations/                  # 6 factorial ablation variants
├── bias/                       # Demographic parity analysis
├── scripts/
│   ├── run_plantswarm.py       # Main entry point
│   ├── run_baselines.py
│   ├── run_ablations.py
│   ├── run_calibration.py
│   ├── run_routing_analysis.py
│   ├── run_bias_analysis.py
│   ├── sync_latex_metrics.py   # JSON → LaTeX
│   └── build_latex_pdf.sh      # LaTeX → PDF
│
├── configs/                    # YAML experiment configs
├── setup.py
├── requirements.txt
└── README.md (this file)
```

---

## 🤖 OBSERVE Quick Reference

### Training a New OBSERVE Model
```bash
# Full training (10,000 traces, 50 epochs)
python scripts/train_observe.py \
  --traces results/plant_village_tfds/traces/plantswarm_traces.jsonl \
  --output observe/checkpoints/observe_final.pt \
  --epochs 50 --batch-size 8

# Check training history
cat observe/checkpoints/training_history.json
```

### Using OBSERVE for Inference
```python
from observe import OBSERVEInference
from PIL import Image

# Load trained model
inference = OBSERVEInference("observe/checkpoints/observe_final.pt")

# Single image prediction
image = Image.open("plant_crop.jpg")
context = "Symptoms: lesions on leaf"
action = inference.predict(image, context)

# Inspect results
print(f"Next agent: {action.next_agent}")
print(f"Confidence: {action.confidence:.3f}")
print(f"Epistemic uncertainty: {action.epistemic_uncertainty:.3f}")
print(f"Aleatoric uncertainty: {action.aleatoric_uncertainty:.3f}")

# Get actionable recommendations
decomp = inference.get_uncertainty_decomposition(action)
print(decomp["epistemic"]["recommendation"])
print(decomp["aleatoric"]["recommendation"])
```

### Batch Inference (Faster)
```python
from observe import OBSERVEInference
from PIL import Image

inference = OBSERVEInference("observe/checkpoints/observe_final.pt")

# Load multiple images
images = [Image.open(f"crop_{i}.jpg") for i in range(100)]

# Batch predict (4 images at a time)
actions = inference.predict_batch(images, batch_size=4)

# Process results
for i, action in enumerate(actions):
    print(f"Image {i}: {action.next_agent}, conf={action.confidence:.3f}")
```

### Evaluate OBSERVE on Benchmark
```bash
# ID evaluation (PlantVillage)
python scripts/evaluate_observe.py \
  --model observe/checkpoints/observe_final.pt \
  --traces results/plant_village_tfds/traces/plantswarm_traces.jsonl \
  --output results/plant_village_tfds/observe_eval.json

# OOD evaluation (PlantWild)
python scripts/evaluate_observe.py \
  --model observe/checkpoints/observe_final.pt \
  --traces results/plantwild/traces/plantswarm_traces.jsonl \
  --output results/plantwild/observe_eval.json
```

---

## 🧪 Troubleshooting

### vLLM Server Issues
```bash
# Check server is reachable
curl http://localhost:8000/v1/models

# Verify Qwen model loaded
# Output should include: "Qwen/Qwen3-VL-8B-Instruct"

# If memory error: reduce batch size or use smaller model
# In config: adjust image_size or use Qwen2.5-VL-7B
```

### TFDS Download Stuck
```bash
# Clear stale cache
rm -rf ~/tensorflow_datasets/plant_village

# Retry with subset
python scripts/run_plantswarm.py --config configs/plant_village_tfds.yaml --subset 100
```

### LaTeX PDF Build Fails
```bash
# Install TeX Live (macOS)
brew install texlive

# Or use TinyTeX
curl -fsSL https://yihui.org/tinytex/install-bin-unix.sh | sh

# Then try build again
bash scripts/build_latex_pdf.sh --latex-dir plantswarm/latex/
```

### Out of Memory
```bash
# Run on smaller subset first
python scripts/run_plantswarm.py --config configs/plant_village_tfds.yaml --subset 100

# Reduce calibration split
# In config: calibration_split_size: 100 (default 500)
```

---

## 📈 Performance Targets (Paper Results)

### PlantVillage (Controlled)
| Metric | T2 | T3 |
|--------|----|----|
| Macro-F1 | 92.3% | 88.9% |
| ECE | 0.08 | 0.11 |
| TPCP | 650 | 720 |

### PlantWild (OOD)
| Metric | T2 | T3 |
|--------|----|----|
| Macro-F1 | 85.1% | 79.4% |
| ECE | 0.16 | 0.19 |
| (52% ECE improvement vs. prompt-based baselines) |

---

## 📚 Citation

If you use this code, cite the paper:

```bibtex
@inproceedings{observe2026,
  title={Why Ask When You Can Observe? A Vision-Language-Action Model for Epistemic Action Selection in Multi-Agent Crop Disease Diagnosis},
  author={[Authors]},
  booktitle={Proceedings of EMNLP 2026},
  year={2026}
}
```

---

## 📝 License

This project is released under the MIT License. See LICENSE file for details.

---

## 🤝 Contributing

For bug reports, feature requests, or questions:
1. Check existing issues
2. Open a new issue with reproducible steps
3. For contributions, open a pull request

---

## 📞 Support

For questions about:
- **Paper/methodology:** See Section 2-6 of `plantswarm/latex/acl_latex.tex`
- **Code:** Check docstrings in respective modules
- **Results:** Refer to `results/*/` directories after running experiments

---

**Last Updated:** May 2026  
**Tested on:** Python 3.10+, CUDA 11.8+, vLLM 0.4+
