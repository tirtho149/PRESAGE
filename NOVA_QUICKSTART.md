# Nova HPC Quick Start Guide

**For ISU Nova users running PlantSwarm + OBSERVE**

---

## 🚀 TL;DR (5 minutes)

```bash
# 1. Clone to Nova
cd /work/mech-ai/tirtho/
git clone https://github.com/yourusername/ObservePlantSwarm.git
cd ObservePlantSwarm

# 2. One-time setup
module load python cuda/11.8
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p logs

# 3. Submit entire pipeline
bash scripts/submit_all_phases.sh

# 4. Monitor (in another terminal)
squeue -u $USER

# 5. When done, push results
git add results/ observe/checkpoints/ plantswarm/latex/auto/
git commit -m "Pipeline complete"
git push origin main
```

**That's it!** All 5 phases will run automatically.

---

## Step-by-Step Setup

### 1. Clone Repository to Nova
```bash
# SSH to Nova login node
ssh tirtho@hpc-login.iastate.edu

# Navigate to work directory
cd /work/mech-ai/tirtho/

# Clone from GitHub (public repo)
git clone https://github.com/yourusername/ObservePlantSwarm.git
cd ObservePlantSwarm

# Verify files
ls -la scripts/submit_*.sh  # Should list 6 SLURM scripts
ls -la observe/            # Should list model.py, trainer.py, inference.py
```

### 2. Initialize Python Environment
```bash
# Load required modules
module load python cuda/11.8

# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-tfds.txt

# Verify
python -c "from observe import OBSERVE; print('✓ Setup complete')"
```

### 3. Create Directories
```bash
# Create logs directory for SLURM output
mkdir -p logs results/plant_village_tfds results/plantwild

# Verify
ls -d logs results/
```

---

## Running the Pipeline

### Option A: Full Pipeline (Recommended)
**Submits all 5 phases with automatic dependencies**

```bash
bash scripts/submit_all_phases.sh
```

**Output:**
```
✓ Phase 1 submitted: Job ID 12345
✓ Phase 2 submitted: Job ID 12346 (depends on 12345)
✓ Phase 3 submitted: Job ID 12347 (depends on 12346)
✓ Phase 4 submitted: Job ID 12348 (depends on 12347)
✓ Phase 5 submitted: Job ID 12349 (depends on 12347)

Monitor: squeue -u $USER
```

All jobs will run sequentially automatically. Total time: ~25-35 hours.

---

### Option B: Manual Phase by Phase
**For more control or debugging**

```bash
# Phase 1 (12-18 hours)
sbatch scripts/submit_phase1_plantswarm.sh
# Wait for completion, then:

# Phase 2 (2-3 hours)
sbatch scripts/submit_phase2_experiments.sh
# Wait for completion, then:

# Phase 3 (4-6 hours)
sbatch scripts/submit_phase3_observe_training.sh
# Wait for completion, then:

# Phase 5 (fast)
sbatch scripts/submit_phase5_latex_sync.sh
```

---

### Option C: Chain with Dependencies
**For unattended execution**

```bash
# Get Job ID from Phase 1
JOB1=$(sbatch scripts/submit_phase1_plantswarm.sh | awk '{print $NF}')
echo "Phase 1: $JOB1"

# Chain Phase 2 to Phase 1
JOB2=$(sbatch --dependency=afterok:$JOB1 scripts/submit_phase2_experiments.sh | awk '{print $NF}')
echo "Phase 2: $JOB2"

# Chain Phase 3 to Phase 2
JOB3=$(sbatch --dependency=afterok:$JOB2 scripts/submit_phase3_observe_training.sh | awk '{print $NF}')
echo "Phase 3: $JOB3"

# Chain Phase 5 to Phase 3 (no GPU needed)
JOB5=$(sbatch --dependency=afterok:$JOB3 scripts/submit_phase5_latex_sync.sh | awk '{print $NF}')
echo "Phase 5: $JOB5"

# Now everything is queued and will run automatically!
```

---

## Monitoring Jobs

### Check All Your Jobs
```bash
squeue -u $USER
```

**Output example:**
```
JOBID PARTITION      NAME     USER ST       TIME  NODES CPUS
12345      nova     plantswarm tirtho  R   2:15:30      1    8
12346      nova     experiments tirtho  PD  0:00       1    8  (waiting for 12345)
```

### Monitor Specific Job
```bash
squeue -j 12345
scontrol show job 12345  # More detailed info
```

### Watch Output in Real-Time
```bash
tail -f logs/phase1_plantswarm-12345.out
# Press Ctrl+C to stop watching
```

### Check for Errors
```bash
grep -i error logs/phase1_plantswarm-12345.err
cat logs/phase1_plantswarm-12345.err  # See all error messages
```

### Cancel a Job
```bash
scancel 12345           # Cancel job 12345
scancel -u $USER        # Cancel all your jobs
```

---

## After Pipeline Completes

### 1. Verify All Outputs Generated
```bash
# Check Phase 1 outputs
ls -lh results/plant_village_tfds/plantswarm_metrics.json
ls -lh results/plant_village_tfds/traces/plantswarm_traces.jsonl

# Check Phase 2 outputs
ls -lh results/plant_village_tfds/baseline_results.json
ls -lh results/plant_village_tfds/ablation_metrics_*.json
ls -lh results/plant_village_tfds/calibration_report.json
ls -lh results/plant_village_tfds/routing_analysis.json

# Check Phase 3 outputs
ls -lh observe/checkpoints/observe_final.pt
ls -lh observe/checkpoints/training_history.json

# Check Phase 5 outputs
ls -lh plantswarm/latex/auto/auto_*.tex
```

### 2. Push Results to GitHub
```bash
# Stage all results
git add results/
git add observe/checkpoints/
git add plantswarm/latex/auto/

# Commit
git commit -m "Pipeline complete - $(date '+%Y-%m-%d')'
git config user.email "tirtho@iastate.edu"
git config user.name "Tirtho Roy"
git commit -m "Full pipeline results"

# Push to GitHub
git push origin main
```

### 3. Pull on Local Machine
```bash
# On your Mac/local
cd ~/Desktop/ObservePlantSwarm
git pull origin main
ls results/  # Should now have all results

# Compile paper with synced metrics
cd plantswarm/latex
latexmk -pdf acl_latex.tex
# Opens: acl_latex.pdf with all metrics filled in!
```

---

## Troubleshooting

### "Module load: python not found"
```bash
module avail python  # See available Python versions
module load python/3.10  # Load specific version
```

### "VENV permission denied"
```bash
# Try absolute path
/usr/bin/python3 -m venv /work/mech-ai/tirtho/ObservePlantSwarm/.venv
```

### "Job failed: CUDA out of memory"
Edit `scripts/submit_phase3_observe_training.sh`, change:
```bash
# From:
--batch-size 8

# To:
--batch-size 4
```

Then resubmit.

### "traces not found" in Phase 2
Phase 1 didn't complete successfully. Check:
```bash
tail -100 logs/phase1_plantswarm-*.out
tail -50 logs/phase1_plantswarm-*.err
```

### "Job time limit exceeded"
Phase ran longer than allocated time. Edit .sh file, change:
```bash
# From:
#SBATCH --time=08:00:00

# To:
#SBATCH --time=12:00:00
```

### "Permission denied" running scripts
Make executable:
```bash
chmod +x scripts/submit_*.sh
bash scripts/submit_all_phases.sh
```

---

## File Transfer: Local ↔ Nova

### Push Local Code to Nova
```bash
# On your Mac
cd ~/Desktop/ObservePlantSwarm
git add -A
git commit -m "New code changes"
git push origin main

# On Nova
cd /work/mech-ai/tirtho/ObservePlantSwarm
git pull origin main
```

### Pull Nova Results to Local
```bash
# On Nova
cd /work/mech-ai/tirtho/ObservePlantSwarm
git add results/ observe/checkpoints/ plantswarm/latex/auto/
git commit -m "Results from pipeline"
git push origin main

# On your Mac
cd ~/Desktop/ObservePlantSwarm
git pull origin main
ls results/  # Results now on your machine!
```

---

## Expected Results

After successful pipeline completion:

### PlantVillage (Controlled)
- T3 Macro F1: **89-92%**
- ECE: **0.08-0.09**
- TPCP: **600-700 tokens/image**

### OBSERVE Performance
- Cost reduction: **6× (700 vs 4,200 tokens)**
- OOD ECE: **0.16 (52% improvement)**
- Overconfidence F1: **0.81**
- Escalation F1: **0.84**

### Paper (Phase 5)
- All tables auto-filled with metrics
- PDF: `plantswarm/latex/acl_latex.pdf`

---

## Quick Commands Reference

```bash
# Navigation
cd /work/mech-ai/tirtho/ObservePlantSwarm
source .venv/bin/activate

# Submission
bash scripts/submit_all_phases.sh          # All phases
sbatch scripts/submit_phase3_observe_training.sh  # Single phase

# Monitoring
squeue -u $USER                             # All your jobs
tail -f logs/phase*.out                     # Watch output
scancel 12345                               # Cancel job

# Git (push to GitHub for transfer)
git add .
git commit -m "message"
git push origin main                        # Send to GitHub
git pull origin main                        # Get from GitHub

# Compilation
cd plantswarm/latex && latexmk -pdf acl_latex.tex
```

---

## Contact & Support

**Questions about Nova?**
- Contact HPC support: hpc-support@iastate.edu
- Or ask Tirtho: tirtho@iastate.edu

**Issues with pipeline?**
- Check logs: `cat logs/phase*.err`
- Read SLURM_README.md for detailed troubleshooting
- Review specific phase's .sh script

---

## Next Steps

1. ✅ Clone to Nova
2. ✅ Setup Python environment
3. ✅ Run `bash scripts/submit_all_phases.sh`
4. ✅ Monitor with `squeue -u $USER`
5. ✅ Push results to GitHub
6. ✅ Pull on local + compile paper

**Total time:** ~1 hour setup + ~30 hours computation = 31 hours wall-clock time

---

**Last Updated:** 2026-05-01  
**Tested on:** ISU Nova HPC (SLURM scheduler)
