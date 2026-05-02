# SLURM Job Submission Guide (Nova HPC)

Complete guide for running PlantSwarm + OBSERVE on ISU Nova HPC cluster.

---

## Quick Start

### 1. Clone to Nova & Setup (One-time)
```bash
# On Nova login node
cd /work/mech-ai/tirtho/

# Clone from GitHub
git clone https://github.com/yourusername/ObservePlantSwarm.git
cd ObservePlantSwarm

# Create logs directory
mkdir -p logs

# Setup virtual environment
module load python cuda/11.8
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-tfds.txt

# Verify setup
python -c "from observe import OBSERVE; print('✓ Ready')"
```

### 2. Make SLURM Scripts Executable
```bash
chmod +x scripts/submit_*.sh
```

### 3. Run Pipeline (Choose one)

**Option A: Submit all phases sequentially (auto-chainable)**
```bash
# Submit Phase 1 (generates dependency for Phase 2)
sbatch scripts/submit_phase1_plantswarm.sh

# Note: Will create output showing job ID, e.g., "Submitted batch job 12345"
# Use that to chain Phase 2:
sbatch --dependency=afterok:12345 scripts/submit_phase2_experiments.sh
sbatch --dependency=afterok:12346 scripts/submit_phase3_observe_training.sh
# etc.
```

**Option B: Submit all at once (manual chaining)**
```bash
# Check job status
squeue -u $USER

# Submit Phase 1
JOB1=$(sbatch scripts/submit_phase1_plantswarm.sh | awk '{print $NF}')
echo "Phase 1 Job ID: $JOB1"

# Submit Phase 2 (wait for Phase 1)
JOB2=$(sbatch --dependency=afterok:$JOB1 scripts/submit_phase2_experiments.sh | awk '{print $NF}')
echo "Phase 2 Job ID: $JOB2"

# Submit Phase 3 (wait for Phase 2)
JOB3=$(sbatch --dependency=afterok:$JOB2 scripts/submit_phase3_observe_training.sh | awk '{print $NF}')
echo "Phase 3 Job ID: $JOB3"

# Optional: Phase 4 (OOD evaluation)
JOB4=$(sbatch --dependency=afterok:$JOB3 scripts/submit_phase4_ood_evaluation.sh | awk '{print $NF}')
echo "Phase 4 Job ID: $JOB4"

# Phase 5 (LaTeX sync, can run independently)
sbatch --dependency=afterok:$JOB3 scripts/submit_phase5_latex_sync.sh
```

**Option C: Manual step-by-step (safest for first time)**
```bash
# Phase 1: Submit and wait
sbatch scripts/submit_phase1_plantswarm.sh
# Monitor: squeue -u $USER
# Wait until COMPLETED

# Phase 2: Submit and wait
sbatch scripts/submit_phase2_experiments.sh
# Wait until COMPLETED

# Phase 3: Submit and wait
sbatch scripts/submit_phase3_observe_training.sh
# Wait until COMPLETED

# Phase 5: LaTeX sync (fast)
sbatch scripts/submit_phase5_latex_sync.sh
```

---

## SLURM Script Details

### Phase 1: PlantSwarm Routing Traces (submit_phase1_plantswarm.sh)
**Duration:** 12-18 hours  
**GPU:** 1x (A100 preferred)  
**Memory:** 48 GB  
**Input:** PlantVillage dataset (auto-downloaded via TFDS)  
**Output:** `results/plant_village_tfds/plantswarm_metrics.json`, `traces/plantswarm_traces.jsonl`

```bash
sbatch scripts/submit_phase1_plantswarm.sh
# Output: "Submitted batch job 12345"
```

**Monitor:**
```bash
squeue -j 12345  # Check job status
tail -f logs/phase1_plantswarm-12345.out  # Watch output
```

---

### Phase 2: Experiments (submit_phase2_experiments.sh)
**Duration:** 2-3 hours  
**GPU:** 1x  
**Memory:** 48 GB  
**Input:** plantswarm_predictions.jsonl from Phase 1  
**Output:** 
- `baseline_results.json`
- `ablation_metrics_*.json`
- `calibration_report.json`
- `routing_analysis.json`

Runs sequentially:
- 2a: Baselines (8 methods, 30-45 min)
- 2b: Ablations (6 variants, 45 min)
- 2c: Calibration (15-30 min)
- 2d: Routing analysis (P1-P4, 15-30 min)

```bash
sbatch --dependency=afterok:12345 scripts/submit_phase2_experiments.sh
```

---

### Phase 3: OBSERVE Training (submit_phase3_observe_training.sh)
**Duration:** 4-6 hours (A100)  
**GPU:** 1x (40GB+ VRAM needed)  
**Memory:** 64 GB  
**Input:** `traces/plantswarm_traces.jsonl` from Phase 1  
**Output:** 
- `observe/checkpoints/observe_final.pt` (trained model)
- `observe/checkpoints/training_history.json` (curves)

This is the most critical computation. Trains LoRA on Qwen2.5-VL-3B.

```bash
sbatch --dependency=afterok:12346 scripts/submit_phase3_observe_training.sh

# Monitor training loss
tail -f logs/phase3_observe_training-*.out | grep "Epoch"
```

---

### Phase 4: OOD Evaluation (submit_phase4_ood_evaluation.sh)
**Duration:** 2-3 hours  
**GPU:** 1x  
**Memory:** 48 GB  
**Input:** `observe_final.pt` from Phase 3  
**Output:** `results/plantwild/observe_evaluation.json` (OOD metrics)

Evaluates OBSERVE on wild (uncontrolled) PlantWild dataset.

```bash
sbatch --dependency=afterok:12347 scripts/submit_phase4_ood_evaluation.sh
```

---

### Phase 5: LaTeX Sync (submit_phase5_latex_sync.sh)
**Duration:** <1 minute  
**GPU:** None (CPU only)  
**Memory:** 16 GB  
**Input:** All JSON results files from Phases 1-2  
**Output:** 
- `plantswarm/latex/auto/auto_metrics.tex`
- `plantswarm/latex/auto/auto_table_*.tex` (6 files)

Generates LaTeX table fragments and inline macros for paper.

```bash
sbatch --dependency=afterok:12347 scripts/submit_phase5_latex_sync.sh

# Copy synced paper for compilation
cp plantswarm/latex/acl_latex.tex plantswarm/latex/acl_latex_synced.tex
```

---

## Monitoring Jobs

### Check Status
```bash
# All your jobs
squeue -u $USER

# Specific job
squeue -j 12345

# Detailed info
scontrol show job 12345
```

### View Output/Logs
```bash
# Real-time log
tail -f logs/phase1_plantswarm-12345.out

# Last 50 lines
tail -50 logs/phase1_plantswarm-12345.out

# Check for errors
grep -i "error" logs/phase1_plantswarm-12345.err
```

### Cancel Jobs
```bash
# Cancel specific job
scancel 12345

# Cancel all your jobs
scancel -u $USER
```

---

## Troubleshooting

### "Module not found: python"
```bash
module load python cuda/11.8  # Load before running scripts
```

### "VENV not found"
```bash
# Create virtual environment first
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### "GPU out of memory"
**Solution:** Reduce batch size in Phase 3 script
```bash
# Edit submit_phase3_observe_training.sh
# Change: --batch-size 8
# To:     --batch-size 4
```

### "Job timed out"
**Increase time limit in .sh file:**
```bash
# In submit_phase3_observe_training.sh
# Change: --time=08:00:00
# To:     --time=12:00:00
```

### "Traces not found" in Phase 2
**Ensure Phase 1 completed successfully:**
```bash
ls results/plant_village_tfds/traces/plantswarm_traces.jsonl
# If not found, Phase 1 failed - check its logs
```

---

## Tips & Best Practices

### 1. Always Check Available GPUs
```bash
sinfo -p nova -o "%20N %10c %10m %20G"  # GPU availability
```

### 2. Estimate Time Before Submitting
```bash
# Phase 1: ~12-18h
# Phase 2: ~2-3h
# Phase 3: ~4-6h (A100), ~8-12h (V100)
# Phase 4: ~2-3h
# Total: ~20-32 hours (with good GPU)
```

### 3. Chain Jobs Automatically
```bash
# Submit all with dependencies in one go
JOB1=$(sbatch scripts/submit_phase1_plantswarm.sh | awk '{print $NF}')
JOB2=$(sbatch --dependency=afterok:$JOB1 scripts/submit_phase2_experiments.sh | awk '{print $NF}')
JOB3=$(sbatch --dependency=afterok:$JOB2 scripts/submit_phase3_observe_training.sh | awk '{print $NF}')
JOB5=$(sbatch --dependency=afterok:$JOB3 scripts/submit_phase5_latex_sync.sh | awk '{print $NF}')

# All 5 jobs queued, will run automatically in sequence
squeue -u $USER  # See full pipeline
```

### 4. Keep Results Backed Up
```bash
# After each phase, copy results to backup
cp -r results/ results_backup_$(date +%Y%m%d)/
```

### 5. Push Results Back to GitHub
```bash
# After Phase 5 completes
git add results/ observe/checkpoints/ plantswarm/latex/auto/
git commit -m "Results from Phase 1-5 pipeline - $(date)"
git push origin main
```

---

## Running Locally vs Nova

### Local Machine (Mac/Linux)
```bash
python scripts/train_observe.py --device cuda  # Uses GPU if available
```

### Nova HPC
```bash
sbatch scripts/submit_phase3_observe_training.sh  # Uses SLURM + GPU
```

Key difference: Nova uses SLURM job scheduler; local runs directly.

---

## Complete End-to-End Example

```bash
# 1. Clone and setup (once)
cd /work/mech-ai/tirtho/
git clone https://github.com/yourusername/ObservePlantSwarm.git
cd ObservePlantSwarm
module load python cuda/11.8
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p logs

# 2. Submit full pipeline
chmod +x scripts/submit_*.sh
JOB1=$(sbatch scripts/submit_phase1_plantswarm.sh | awk '{print $NF}')
echo "Submitted Phase 1: $JOB1"

JOB2=$(sbatch --dependency=afterok:$JOB1 scripts/submit_phase2_experiments.sh | awk '{print $NF}')
echo "Submitted Phase 2: $JOB2"

JOB3=$(sbatch --dependency=afterok:$JOB2 scripts/submit_phase3_observe_training.sh | awk '{print $NF}')
echo "Submitted Phase 3: $JOB3"

JOB5=$(sbatch --dependency=afterok:$JOB3 scripts/submit_phase5_latex_sync.sh | awk '{print $NF}')
echo "Submitted Phase 5: $JOB5"

# 3. Monitor
squeue -u $USER  # Track all jobs

# 4. When done, push results
git add results/ observe/checkpoints/ plantswarm/latex/auto/
git commit -m "Pipeline complete"
git push origin main

# 5. On local: Pull results and compile paper
git pull origin main
cd plantswarm/latex && latexmk -pdf acl_latex.tex
```

---

## Email Notifications

Each script sends emails on:
- **BEGIN:** Job started
- **END:** Job completed successfully
- **FAIL:** Job failed (check logs!)

Set email in script header:
```bash
#SBATCH --mail-user=your.email@iastate.edu
```

---

## Support

**Check logs first:**
```bash
# Most common issues appear in .out or .err files
cat logs/phase*.out
cat logs/phase*.err
```

**Common errors:**
- "ModuleNotFound" → Load modules first
- "CUDA out of memory" → Reduce batch size
- "File not found" → Check previous phase completed

---

**Last Updated:** 2026-05-01  
**Tested on:** Nova HPC (SLURM)
