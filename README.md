# PlantSwarm + PathomeDB + OBSERVE — Train on the Wild

**Paper:** *Train on the Wild: Geospatial Multi-Agent Routing for Cross-Crop Plant Disease Diagnosis from Ten Field Images* (EMNLP 2026, anonymous submission). Source: `plantswarm/latex/acl_latex.tex`.

**Core argument.** Train on the field; deploy on the field. We seed a knowledge base with what diseases *look like*, generate diverse routing traces with a multi-agent VLM swarm, then enhance the KB with what the swarm actually saw — and measure how much that enhancement is worth.

**The pipeline in one diagram:**

```
  ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
  │  Phase 0 SEED  │  →   │ Phase 1 BUILD  │  →   │ Phase 2 TRACES │
  │ Claude headless│      │ symptoms.json +│      │ Qwen2.5-VL-7B  │
  │ writes 484     │      │ state/AEZ geo +│      │ × 5 agents     │
  │ VisualSymptom  │      │ 1,452 refs from│      │ × 30 runs      │
  │ blocks         │      │ Bugwood CSV    │      │ = 101k traces  │
  └────────────────┘      └────────────────┘      └────────┬───────┘
                                                            │
  ┌────────────────┐      ┌────────────────┐      ┌────────▼───────┐
  │ Phase 5 COMPARE│  ←   │ Phase 4 TRAIN  │  ←   │ Phase 3 ENHANCE│
  │ before / after │      │ OBSERVE × 2    │      │ mine traces →  │
  │ ΔT3 F1, ΔECE,  │      │ (seed DB,      │      │ SwarmObserva-  │
  │ ΔPathLen, …    │      │  enhanced DB)  │      │ tions per class│
  └────────────────┘      └────────────────┘      └────────────────┘
```

**Source of truth** is `BugWood_Diseases_usable.csv` (11,513 rows, 484 classes; produced by `scripts/filter_bugwood_csv.py`). Geo signal is at US-state-centroid resolution (paper §6.3 monthly AEZ grid is reduced to a state histogram — see [`MIGRATION.md`](MIGRATION.md) caveats).

**PathomeDB v2** (current) is two stores:

- `db.symptoms` — `SymptomLibrary` of `SymptomProfile` per (crop, disease). Each profile carries a `VisualSymptom` block (plant parts, color, shape/margin/texture, sporulation, distinctive signs, progression, confusion diseases, notes), per-state and per-AEZ observation counts, and an optional `swarm_observations` block populated after Phase 3.
- `db.refs` — `ReferenceLibrary` over the 1,452 held-out Bugwood references with CLIP embeddings + FAISS retrieval (`0.7·cos + 0.3·ClimSim`).

The older 5-layer split (mechanistic pathway, cross-crop manifestation, regional epidemiology, decision graph, reference library) was collapsed into the symptom-centric design once it became clear that the Bugwood CSV cannot feed the mechanistic / decision-graph layers. Old layer modules remain on disk for users who curated content against them.

> **Status:** mid-migration. See [`MIGRATION.md`](MIGRATION.md) for what's done, what's stubbed, and the dependency order.

---

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.10+
- NVIDIA GPU (for vLLM inference)
- 50GB disk (for TFDS Plant Village cache)

### Install & Test
```bash
# 1. Clone and setup
cd PlantSwarm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Install TFDS support (for Plant Village)
pip install -r requirements-tfds.txt

# 3. Verify imports
python -c "from agents import *; from plantswarm import *; print('✓ Ready')"
```

---

## 📋 Pathome Pipeline (6 Phases — current workflow)

| Phase | Compute | Wall | What it does |
|---|---|---|---|
| **0 SEED** | CPU, no GPU | ~15 min | `claude -p` writes a `VisualSymptom` block per (crop, disease) into `artifacts/pathome_seed/symptoms_seed.json` |
| **1 BUILD** | CPU + outbound HTTPS | ~30 min | Layers Bugwood state/AEZ counts and 1,452 reference IDs onto the Claude seed → `artifacts/pathome_v1_seed/` |
| **2 TRACES** | A100 + vLLM | ~36–50 h | 3,388 trace seeds × 30 runs = **101,640 PlantSwarm traces** (resumable) |
| **3 ENHANCE** | CPU | ~5 min | Mines traces into per-profile `SwarmObservations` → `artifacts/pathome_v1_enhanced/` |
| **4 TRAIN** | A100 | ~20–24 h | Trains OBSERVE × 2 (Phase A DT + Phase B GRPO) — once on seed DB, once on enhanced DB |
| **5 EVAL+COMPARE** | A100 + CPU | ~6–8 h | Held-out eval on full PV (with seen/unseen slice) and full PW for both checkpoints, then writes `comparison.{json,md,tex}` |

The legacy "PlantSwarm-on-PlantVillage" workflow (5 different phases, pre-Pathome) is documented further down under [Legacy: PlantVillage workflow](#legacy-plantvillage-workflow) and is no longer the recommended path.

---

## ☁️ Google Colab Setup (Free GPU)

Run on Google Colab for free GPU access (T4 15GB) or upgrade to Pro for faster GPUs (V100/A100).

### Quick Start
```bash
# Option 1: Direct Colab Link
# Open in browser:
https://colab.research.google.com/github/tirtho149/PlantSwarm/blob/main/notebooks/plantswarm_colab.ipynb

# Option 2: VS Code + Colab Extension (Recommended)
# 1. Install: Extensions → Search "Colab" → Install "Colab" by Google
# 2. Open notebook in VS Code
# 3. Click "Open in Colab" button
# 4. Edit in VS Code, execute in Colab browser tab
```

**See [COLAB_SETUP.md](COLAB_SETUP.md) for full details, free vs Pro comparison, and troubleshooting.**

---

## 🖥️ First-Time Nova HPC Setup

### Step 1: SSH Access
```bash
# On your local machine
ssh tirtho@hpc-login.iastate.edu

# You'll be prompted for your ISU credentials
# If this is your first time, contact HPC support for account activation
```

### Step 2: Clone Repository
```bash
# On Nova login node
cd /work/mech-ai-scratch/tirtho/
git clone https://github.com/tirtho149/PlantSwarm.git
cd PlantSwarm
```

### Step 3: Load Modules & Create Virtual Environment
```bash
# Load required modules
module load python cuda/11.8

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Verify Python
python --version  # Should be 3.10+
```

### Step 4: Install Dependencies
```bash
# Install core dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Install TFDS support for PlantVillage
pip install -r requirements-tfds.txt

# Verify installation
python -c "from agents import *; from plantswarm import *; print('✓ Ready')"
```

### Step 5: Create Required Directories
```bash
# Create directory structure
mkdir -p logs data results observe/checkpoints

# Set proper permissions
chmod -R 755 logs data results observe
```

### Step 6: Configure Git (One-time)
```bash
# Set up Git credentials for syncing
git config --global user.name "Your Name"
git config --global user.email "your.email@iastate.edu"

# Verify
git config --global --list
```

### Verification Checklist
```bash
# Verify all setup
python -c "import tensorflow_datasets; print('✓ TFDS')"
python -c "import torch; print('✓ PyTorch')"
python -c "from observe import OBSERVE; print('✓ OBSERVE')"
sbatch --version  # Should show SLURM version
```

---

## 🚀 Running the Pathome Pipeline on Nova HPC

All six phases are individual SLURM scripts under `scripts/submit_pathome_phase*.sh` plus a master chain `scripts/submit_pathome_all.sh` that wires sbatch dependencies for you.

### One-time prerequisites

```bash
# 1. Working tree on Nova
cd /work/mech-ai-scratch/tirtho/PlantSwarm
git pull origin main

# 2. The Claude Code CLI must be on PATH and authenticated for Phase 0
#    (it shells out to `claude -p` × 484 calls in the seeding step)
curl -fsSL https://claude.ai/install.sh | bash
claude auth login
claude --version          # confirm

# 3. Filtered Bugwood CSV (11,513 rows, 484 classes)
python scripts/filter_bugwood_csv.py --threshold 10
ls -lh BugWood_Diseases_usable.csv
```

### Quick start: chain everything

```bash
bash scripts/submit_pathome_all.sh

# squeue -u $USER will show six queued jobs:
#   pathome_phase0_seed         → pathome_phase1_build      (afterok)
#   pathome_phase1_build        → pathome_phase2_traces     (afterok)
#   pathome_phase2_traces       → pathome_phase3_enhance    (afterok)
#   pathome_phase3_enhance      → pathome_phase4_train      (afterok)
#   pathome_phase4_train        → pathome_phase5_eval       (afterok)

# Skip phases that are already done:
PATHOME_SKIP="0,1" bash scripts/submit_pathome_all.sh    # seed + build cached
PATHOME_FROM_PHASE=4 bash scripts/submit_pathome_all.sh  # start at training
```

### Individual phase scripts

#### Phase 0 — Seed visual symptom blocks (`submit_pathome_phase0_seed.sh`)
```bash
# Default: 4 parallel `claude -p` workers, sonnet
sbatch scripts/submit_pathome_phase0_seed.sh

# Tune at submit time:
PATHOME_SEED_WORKERS=8 PATHOME_SEED_MODEL=opus \
  sbatch scripts/submit_pathome_phase0_seed.sh

# Resumable. Failed profiles land in artifacts/pathome_seed/failed.jsonl;
# rerun to pick them up.
```

#### Phase 1 — Build PathomeDB v1_seed (`submit_pathome_phase1_build.sh`)
```bash
sbatch scripts/submit_pathome_phase1_build.sh
# Outputs:
#   artifacts/pathome_v1_seed/symptoms.json  (Claude visual + auto geo + ref_ids)
#   artifacts/pathome_v1_seed/refs/          (held-out reference image registry)
#   artifacts/pathome_v1_seed/build_summary.json
# First run downloads ~600 MB of Bugwood thumbnails to .bugwood_cache/
```

#### Phase 2 — PlantSwarm trace generation (`submit_pathome_phase2_traces.sh`)
```bash
sbatch scripts/submit_pathome_phase2_traces.sh
# A100 + vLLM (Qwen2.5-VL-7B). Walltime 72 h, traces appended with fsync.
# A SIGKILL is recoverable — resubmit and already-done image_ids are skipped.

# Override DB / output dir at submit time:
PATHOME_DB_DIR=artifacts/pathome_v1_seed \
PATHOME_OUT_DIR=results/bugwood_seed \
  sbatch scripts/submit_pathome_phase2_traces.sh
```

#### Phase 3 — Enhance DB from traces (`submit_pathome_phase3_enhance.sh`)
```bash
sbatch scripts/submit_pathome_phase3_enhance.sh
# Outputs:
#   artifacts/pathome_v1_enhanced/symptoms.json
#     (each profile.swarm_observations:
#        n_traces, avg_path_length, backtrack_rate,
#        high_confidence_rate, confusion_targets)
#   artifacts/pathome_v1_enhanced/enhancement_summary.json
```

#### Phase 4 — Train OBSERVE × 2 (`submit_pathome_phase4_train.sh`)
```bash
sbatch scripts/submit_pathome_phase4_train.sh
# A100, 24 h walltime. Trains seed-DB OBSERVE first, then enhanced-DB OBSERVE.
# Both runs do Phase A (Decision Transformer) + Phase B (GRPO).
# Outputs:
#   observe/checkpoints/seed/observe_grpo_epoch_*.pt
#   observe/checkpoints/enhanced/observe_grpo_epoch_*.pt
```

#### Phase 5 — Eval × 4 + before/after comparison (`submit_pathome_phase5_eval.sh`)
```bash
sbatch scripts/submit_pathome_phase5_eval.sh
# Boots vLLM once, evaluates seed and enhanced checkpoints on:
#   - full PlantVillage (54,306 images, with seen/unseen slice)
#   - full PlantWild (HF dataset)
# Then runs scripts/compare_pathome_versions.py to emit:
#   results/pathome_compare/comparison.json   (machine-readable deltas)
#   results/pathome_compare/comparison.md     (PR-ready table)
#   results/pathome_compare/comparison.tex    (\PathomeDelta* macros)
```

### Monitoring

```bash
squeue -u $USER                           # all queued jobs
tail -f logs/pathome_phase2_traces-*.out  # live trace progress
tail -f logs/vllm-*.log                   # vLLM server log (phases 2 + 5)
sacct -j <JOBID>                          # post-mortem
```

### Output locations (final)

| File | Phase | Purpose |
|---|---|---|
| `artifacts/pathome_seed/symptoms_seed.json` | 0 | Claude visual blocks |
| `artifacts/pathome_v1_seed/` | 1 | Seed PathomeDB (visual + auto geo + refs) |
| `results/bugwood_seed/traces/plantswarm_traces.jsonl` | 2 | 101,640 routing traces |
| `artifacts/pathome_v1_enhanced/` | 3 | Enhanced PathomeDB (+ swarm_observations) |
| `observe/checkpoints/{seed,enhanced}/observe_grpo_epoch_*.pt` | 4 | Two trained OBSERVE checkpoints |
| `results/pathome_compare/comparison.{json,md,tex}` | 5 | Headline before/after artifact |

<a id="legacy-plantvillage-workflow"></a>
### Legacy: PlantVillage workflow (pre-Pathome)

> The five `submit_phase{1..5}_*.sh` scripts below predate the symptom-centric Pathome pipeline. They train PlantSwarm on PlantVillage directly and do **not** consume the seed/enhance loop. Kept for reference and reproducibility of older results — do not use these for new Pathome runs.

```bash
sbatch scripts/submit_phase1_plantswarm.sh        # PV trace generation
sbatch scripts/submit_phase2_experiments.sh        # baselines + ablations
sbatch scripts/submit_phase3_observe_training.sh  # one-shot OBSERVE training
sbatch scripts/submit_phase4_ood_evaluation.sh    # PlantWild eval
sbatch scripts/submit_phase5_latex_sync.sh        # auto_*.tex
bash   scripts/submit_all_phases.sh               # chain the above
```

### Monitoring Jobs

```bash
# Check all your jobs
squeue -u $USER

# Monitor specific phase output (live)
tail -f logs/pathome_phase2_traces-*.out
tail -f logs/pathome_phase4_train-*.out

# Check for errors
tail -f logs/pathome_phase*-*.err

# See completed job info
sacct -j <JOBID>
```

### Output & Syncing Results

After jobs complete, sync results back to GitHub:

```bash
# On Nova HPC
git add artifacts/pathome_seed/ artifacts/pathome_v1_seed/ artifacts/pathome_v1_enhanced/ \
        results/bugwood_seed/ results/pathome_compare/ \
        observe/checkpoints/seed/ observe/checkpoints/enhanced/ \
        logs/
git commit -m "Pathome pipeline results (seed vs enhanced)"
git push origin main

# On local machine
git pull origin main
cat results/pathome_compare/comparison.md   # the headline before/after table
```

---

## 🔄 Two-Way Sync Workflow (Local ↔ GitHub ↔ Nova)

### Workflow Overview
```
Local Machine ──→ GitHub (code) ──→ Nova HPC (run jobs)
Local Machine ←── GitHub (results) ←── Nova HPC (push results)
```

### Step 1: Push Code Changes (Local → GitHub)

After making code changes locally:

```bash
# On local machine
git status                          # See changes
git add <file1> <file2> ...        # Stage specific files
git commit -m "Description of changes"
git push origin main               # Push to GitHub
```

### Step 2: Pull Code on Nova (GitHub → Nova)

Before running jobs on Nova:

```bash
# On Nova HPC
cd /work/mech-ai-scratch/tirtho/PlantSwarm
git fetch origin                   # Get latest from GitHub
git pull origin main               # Update local clone
```

### Step 3: Submit Jobs and Wait

```bash
# On Nova HPC
bash scripts/submit_all_phases.sh  # Start all 5 phases
squeue -u $USER                    # Monitor jobs
tail -f logs/phase1_plantswarm-*.out  # Watch progress
```

### Step 4: Push Results Back (Nova → GitHub)

After jobs complete:

```bash
# On Nova HPC (in PlantSwarm directory)
git add results/ observe/checkpoints/ plantswarm/latex/auto_* logs/
git commit -m "Results: Phase 1-5 pipeline (X hours, Y% accuracy)"
git push origin main               # Push results to GitHub
```

### Step 5: Pull Results Locally (GitHub → Local)

To get results on your local machine:

```bash
# On local machine
git fetch origin                   # Get latest from GitHub
git pull origin main               # Update with results
ls -lh results/plant_village_tfds/ # Check results downloaded
```

### Full Daily Workflow Example

```bash
# DAY 1: Local Development
# ========================
# On local machine (/work/mech-ai-scratch/tirtho/PlantSwarm)
git add configs/plant_village_tfds.yaml
git commit -m "Adjust temperature scaling parameters"
git push origin main

# DAY 1: Nova Setup & Submit
# ===========================
# SSH to Nova login node
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/PlantSwarm
git pull origin main               # Get latest code
sbatch scripts/submit_setup_plantwild.sh  # One-time dataset download
# Wait 2-4 hours...

# DAY 2: Nova Pipeline Submission
# ================================
bash scripts/submit_all_phases.sh  # Submit all 5 phases (~30 hours)
squeue -u $USER                    # Track progress

# DAY 3-4: Nova Results Sync
# ==========================
# After jobs finish (30 hours later)
git add results/ observe/checkpoints/ plantswarm/latex/auto_* logs/
git commit -m "Full pipeline: 92.3% F1, 0.08 ECE on PlantVillage"
git push origin main

# DAY 4: Local Results Retrieval
# ===============================
# On local machine
git pull origin main
# Review results in results/plant_village_tfds/plantswarm_metrics.json
cat results/plant_village_tfds/plantswarm_metrics.json | python -m json.tool
```

### Common Sync Issues

**Issue:** "Your branch is ahead of origin"
```bash
# You have commits locally that aren't pushed
git push origin main
```

**Issue:** "Your branch is behind origin"
```bash
# Nova has pushed results you haven't pulled
git pull origin main
```

**Issue:** Merge conflict after pulling
```bash
# Edit the conflicted files manually
git add <resolved_files>
git commit -m "Resolve merge conflict"
git push origin main
```

**Issue:** Want to discard local changes and use GitHub version
```bash
git fetch origin
git reset --hard origin/main
```

### Best Practices

✅ **Do:**
- Commit frequently with clear messages
- Push after each significant change
- Pull before starting new work
- Include job logs (.out, .err) in results commits

❌ **Don't:**
- Push large binary files directly (except trained models)
- Force push to main (`git push --force`)
- Edit files in parallel on local + Nova without syncing
- Keep uncommitted changes for >1 day

---

### Phase 1: Generate Routing Traces (Training Data)

**Goal:** Run PlantSwarm on PlantVillage (~10,000 images) to generate routing traces for OBSERVE training.

#### Step 1: Smoke test (always do this first)
```bash
# 50 images, ~10 min on A100 — verifies model loads and traces hit disk
salloc --gres=gpu:a100:1 --time=00:30:00 --mem=64G
source /work/mech-ai-scratch/tirtho/PlantSwarm/.venv/bin/activate
python scripts/run_plantswarm.py \
  --config configs/plant_village_tfds.yaml \
  --orchestrator hf_direct \
  --subset 50

# Confirm traces file is being written
wc -l results/plant_village_tfds/traces/plantswarm_traces.jsonl
```

#### Step 2: Submit Full Phase 1 Job (Nova HPC)
```bash
# hf_direct (default — simpler, slower)
sbatch scripts/submit_phase1_plantswarm.sh

# OR autogen_swarm (boots vLLM in-job — recommended for 10K images)
PLANTSWARM_MODE=autogen_swarm sbatch scripts/submit_phase1_plantswarm.sh

# Monitor progress
tail -f logs/phase1_plantswarm-*.out
watch -n 30 'wc -l results/plant_village_tfds/traces/plantswarm_traces.jsonl'
```

#### Step 3: If walltime kills the job — just resubmit
Each trace is appended to `plantswarm_traces.jsonl` with `fsync` immediately
after it's produced, so SLURM termination, OOM, or crash leaves a usable
partial file. Resubmitting skips already-done image_ids automatically:
```bash
sbatch scripts/submit_phase1_plantswarm.sh   # picks up where it left off
```

**Time:** 12-18 hours on single A100 GPU (autogen_swarm). V100 with
`hf_direct` is roughly 10× slower and not viable for the full 10K.

**Output:** `results/plant_village_tfds/`
- `plantswarm_metrics.json` — accuracy, ECE, TPCP metrics (computed on the most recent run's subset)
- `plantswarm_predictions.jsonl` — per-image predictions (appended)
- `traces/plantswarm_traces.jsonl` — routing traces, appended; training data for OBSERVE

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
Included in Phase 2 experiments script. Analyzes uncertainty quantification:
- ECE before/after temperature scaling
- Reliability diagrams
- Split conformal prediction
- κ calibration (confidence vs. correctness)

**Output:** `results/plant_village_tfds/calibration_report.json`

#### Step 2d: Routing Analysis
Included in Phase 2 experiments script. Tests falsifiable predictions (P1-P4):
- P1: Path length ↔ entropy correlation
- P2: Backtrack improves confidence
- P3: Early termination accuracy
- P4: OOD behavioral transfer

**Output:** `results/plant_village_tfds/routing_analysis.json`

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

#### Step 3b: Train OBSERVE Model (Nova HPC)
```bash
# Submit OBSERVE training job
sbatch scripts/submit_phase3_observe_training.sh

# Monitor progress
tail -f logs/phase3_observe_training-*.out
```

**Training Details:**
- **Time:** 4-6 hours on single A100 GPU
- **Architecture:** Qwen2.5-VL-3B with LoRA (r=16, α=32, ~56M trainable params)
- **Data:** 8,000-10,000 routing traces from PlantSwarm
- **Loss:** Weighted multi-task (routing 1.0 + calibration 0.4 + consistency 0.2 + belief 0.2)
- **Optimizer:** AdamW with lr=1e-4, warmup 500 steps, cosine decay

**Output:**
- `observe/checkpoints/observe_final.pt` — trained model weights
- `observe/checkpoints/training_history.json` — loss curves and metrics

#### Step 3c: Evaluate OBSERVE on PlantVillage (ID)
Evaluation runs automatically after training completes. Check results:
```bash
cat observe/checkpoints/training_history.json
cat results/plant_village_tfds/observe_evaluation.json
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
sbatch scripts/submit_phase4_ood_evaluation.sh
```

**Time:** 2-3 hours on single A100 GPU  
**Output:** `results/plantwild/`
- `plantswarm_metrics.json` — OOD accuracy, ECE (should be worse than PlantVillage)
- `traces/plantswarm_traces.jsonl` — routing traces for OBSERVE evaluation
- Validates robustness to controlled→wild domain shift

Evaluation automatically evaluates OBSERVE on PlantWild after PlantSwarm completes.  
Should show 52% ECE improvement over prompt-based baselines under domain shift.

---

### Phase 5: LaTeX Metrics Sync

**Goal:** Auto-sync all metrics to paper LaTeX files.

```bash
sbatch scripts/submit_phase5_latex_sync.sh
```

**Time:** <1 minute  
**Output:** Auto-generated TeX files synced to `plantswarm/latex/auto_*.tex`
- `auto_metrics.tex` — inline macro definitions (ECE, F1, etc.)
- `auto_table_main_results.tex` — Table 4 (PlantSwarm vs baselines)
- `auto_table_ablation_results.tex` — Table 3 (ablations)
- `auto_table_mechanisms.tex` — context buffer mechanisms

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

## 📦 Advanced: DataLoader.py (30+ Datasets)

For comprehensive dataset curation across 30+ plant disease sources (Kaggle, Zenodo, HuggingFace):

**See [DATALOADER_GUIDE.md](DATALOADER_GUIDE.md)** for:
- Support for SBRD, MangoLeaf, BananaLeaf, Cucumber, PlantDoc, LeafNet, and 24+ more datasets
- Interactive sampling per class with stratification
- Excel multi-sheet reporting with validation
- Crop/disease name normalization across sources

**Note:** For PlantSwarm pipeline, use modular loaders:
- Training: `data/tfds_plant_village.py` (PlantVillage)
- OOD Eval: `data/plantwild_hf.py` (PlantWild)
- General: `data/loader.py` (unified dispatcher)

DataLoader.py is a legacy research tool for exploration and custom dataset integration.

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

## 🤖 OBSERVE Model Usage

### Training OBSERVE
```bash
# Submit training job (Phase 3)
sbatch scripts/submit_phase3_observe_training.sh

# Monitor progress
tail -f logs/phase3_observe_training-*.out

# Check training history after completion
cat observe/checkpoints/training_history.json
```

### Evaluating OBSERVE
```bash
# Evaluate on PlantVillage (ID) or PlantWild (OOD)
sbatch scripts/submit_evaluate_observe.sh

# Check results
cat results/plant_village_tfds/observe_evaluation.json
```

### Using OBSERVE for Inference (Python)
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

# Get actionable recommendations
decomp = inference.get_uncertainty_decomposition(action)
print(decomp["epistemic"]["recommendation"])

# Batch inference
images = [Image.open(f"crop_{i}.jpg") for i in range(100)]
actions = inference.predict_batch(images, batch_size=4)
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

### Phase 1 job killed by SLURM walltime
No action needed. Traces are appended to `plantswarm_traces.jsonl` with
`fsync` after every image, so the partial file is safe. Just resubmit:
```bash
sbatch scripts/submit_phase1_plantswarm.sh   # auto-skips already-done image_ids
wc -l results/plant_village_tfds/traces/plantswarm_traces.jsonl  # progress check
```

### Throughput much slower than 12-18 h estimate
Check the GPU and the orchestrator:
```bash
# In the .out log:
nvidia-smi      # should show A100, not V100
grep orchestrator logs/phase1_plantswarm-*.out
```
If on V100 + `hf_direct`, you'll see ~10 min/image. Switch to A100 (edit
`--gres=gpu:a100:1` in the SLURM script) or `PLANTSWARM_MODE=autogen_swarm`.

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
