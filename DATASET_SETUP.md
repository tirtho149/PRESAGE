# Dataset Setup Guide

Instructions for setting up **PlantVillage** (training) and **PlantWild** (OOD evaluation) datasets.

---

## PlantVillage (Training Dataset)

**Type:** TFDS (TensorFlow Datasets)  
**Size:** ~54,000 images, 50GB  
**Auto-download:** Yes (handled by `tf.datasets`)  
**Setup:** Automatic (no manual download needed)

### How it works:
- First time you run `scripts/run_plantswarm.py --config configs/plant_village_tfds.yaml`
- TensorFlow automatically downloads to: `~/tensorflow_datasets/plant_village/`
- Cached for future runs

### If download is slow:
```bash
# Set cache directory to faster location
export TFDS_DATA_DIR=/work/mech-ai/tirtho/ObservePlantSwarm/data/tfds_cache

# Then run pipeline
python scripts/run_plantswarm.py --config configs/plant_village_tfds.yaml
```

---

## PlantWild (OOD Evaluation Dataset)

**Type:** HuggingFace Dataset (Cloned with git-xet)  
**Size:** ~18,000 images, ~50GB  
**Location:** `data/PlantWild/` (local clone)  
**Download:** Manual (requires HF CLI or git-xet)

### Setup on Nova

#### Option A: Automated (Recommended)
```bash
# Submit SLURM job to download
sbatch scripts/submit_setup_plantwild.sh

# Monitor download
tail -f logs/setup_plantwild-12345.out

# Takes 2-4 hours depending on bandwidth
```

#### Option B: Manual Download
```bash
# Install tools (first time)
curl -LsSf https://hf.co/cli/install.sh | bash
# OR
brew install git-xet && git xet install

# Download dataset
mkdir -p data
cd data
hf download uqtwei2/PlantWild --repo-type=dataset --local-dir ./PlantWild

# Verify
ls -lh PlantWild/
# Should show ~18,000 image files
```

### Dataset Structure
```
data/PlantWild/
├── tomato___early_blight.jpg
├── tomato___late_blight.jpg
├── potato___late_blight.jpg
├── ...
└── (18,000 images total)
```

Image filenames follow format: `<crop>___<disease>.jpg`

---

## Local Development (Mac/Linux)

### Quick Setup
```bash
# PlantVillage (automatic)
# Just run: python scripts/run_plantswarm.py --config configs/plant_village_tfds.yaml

# PlantWild (if you want local OOD eval)
mkdir -p data
cd data

# Option 1: HF CLI (easiest)
brew install hf-hub
hf download uqtwei2/PlantWild --repo-type=dataset --local-dir ./PlantWild

# Option 2: git-xet
brew install git-xet
git xet install
git clone https://huggingface.co/datasets/uqtwei2/PlantWild

# Option 3: Skip download, use API
# scripts/run_plantswarm.py --config configs/plantwild_hf.yaml
# (downloads on-the-fly during execution)
```

---

## Nova HPC Setup

### Recommended Workflow

#### Step 0: Setup PlantWild (one-time, before running pipeline)
```bash
# On Nova login node
cd /work/mech-ai/tirtho/ObservePlantSwarm

# Submit dataset download
sbatch scripts/submit_setup_plantwild.sh

# Takes 2-4 hours. Monitor:
tail -f logs/setup_plantwild-12345.out

# Verify when done
ls -lh data/PlantWild/ | head -20
```

#### Step 1: After PlantWild Downloaded
```bash
# Now submit full pipeline
bash scripts/submit_all_phases.sh

# Phase 1 uses PlantVillage (auto-downloads via TFDS)
# Phase 4 uses local PlantWild (from data/PlantWild/)
```

---

## Checking Dataset Availability

### Check PlantVillage (TFDS)
```bash
python -c "
import tensorflow_datasets as tfds
ds = tfds.load('plant_village', split='train')
print(f'PlantVillage: {len(list(ds))} images')
"
```

### Check PlantWild (Local)
```bash
python -c "
from data.plantwild_local import build_plantwild_local_dataframe
df = build_plantwild_local_dataframe('data/PlantWild')
print(f'PlantWild: {len(df)} images')
print(df.head())
"
```

---

## Troubleshooting

### PlantVillage: "Dataset not found"
```bash
# Check TFDS cache
ls ~/tensorflow_datasets/plant_village/

# If missing, it will auto-download on first run
python scripts/run_plantswarm.py --config configs/plant_village_tfds.yaml
```

### PlantWild: "data/PlantWild not found"
```bash
# Did you download it?
sbatch scripts/submit_setup_plantwild.sh  # Submit download job
# OR
hf download uqtwei2/PlantWild --repo-type=dataset --local-dir ./data/PlantWild
```

### PlantWild: "HF CLI not found"
```bash
# Install
curl -LsSf https://hf.co/cli/install.sh | bash

# Or git-xet
brew install git-xet
git xet install
```

### Download too slow
```bash
# Check bandwidth
iperf3 -c hpc.iastate.edu  # Test connection

# Try git-xet instead of HF CLI (sometimes faster with sparse clone)
git clone https://huggingface.co/datasets/uqtwei2/PlantWild data/PlantWild
```

### Disk space issues
```bash
# Check available space
df -h /work/mech-ai/tirtho/

# PlantWild needs ~50GB
# PlantVillage uses TFDS cache in ~/tensorflow_datasets/ (also ~50GB)
# Total: ~100GB needed

# If low on space, use smaller subset
python scripts/run_plantswarm.py --config configs/plant_village_tfds.yaml --subset 1000
```

---

## Dataset Usage in Code

### Using PlantVillage
```python
# Automatic via TFDS
df = build_plantvillage_dataframe(
    tfds_name="plant_village",
    tfds_split="train",
    tfds_max_examples=10000
)
```

### Using PlantWild (Local)
```python
# From local clone
from data.plantwild_local import build_plantwild_local_dataframe

df = build_plantwild_local_dataframe(
    data_dir="data/PlantWild",
    max_examples=None
)
```

### Using PlantWild (HF API, on-the-fly)
```python
# Download during execution (slower but no pre-download)
from data.plantwild_hf import build_plantwild_dataframe

df = build_plantwild_dataframe(
    hf_dataset_id="uqtwei2/PlantWild",
    split="train"
)
```

---

## Data Specifications

### PlantVillage
- **Images:** ~54,000
- **Crops:** 14 (Tomato, Potato, Grape, Apple, etc.)
- **Diseases:** 26+
- **T1 (Symptoms):** 8 classes (healthy, early blight, late blight, etc.)
- **T2 (Pathogen):** 5 classes (Fungal, Bacterial, Viral, Nutrient, Pest)
- **T3 (Disease Name):** 26+ classes (Late Blight, Early Blight, etc.)
- **T4 (Severity):** 4 classes (Healthy, Mild, Moderate, Severe)
- **T5 (Crop):** 14 classes (Tomato, Potato, Grape, etc.)
- **Image Size:** ~512×512
- **Format:** JPG
- **Source:** https://huggingface.co/datasets/vihanb/PlantVillage

### PlantWild
- **Images:** ~18,000
- **Crops:** 10 (subset of PlantVillage)
- **T3 (Disease Name):** From filename (tomato___early_blight.jpg → "early_blight", "tomato")
- **T5 (Crop):** From filename
- **Conditions:** Wild/uncontrolled (non-laboratory)
- **Image Size:** Variable
- **Format:** JPG
- **Source:** https://huggingface.co/datasets/uqtwei2/PlantWild

---

## Storage Estimates

| Dataset | Compressed | Extracted | Cache | Total |
|---------|-----------|-----------|-------|-------|
| PlantVillage (TFDS) | ~20GB | ~50GB | - | ~50GB |
| PlantWild (HF) | ~15GB | ~50GB | - | ~50GB |
| **Total** | **~35GB** | **~100GB** | - | **~100GB** |

**Recommendation:** Have 150GB available on `/work/mech-ai/tirtho/` to be safe.

---

## Summary

| Dataset | Setup | Download | Time | Auto? |
|---------|-------|----------|------|-------|
| **PlantVillage** | TFDS | Auto | ~30 min | ✅ Yes |
| **PlantWild** | Local (HF) | Manual | ~2-4 hours | ❌ No (one-time) |

**Quick checklist:**
- [ ] PlantVillage: Happens automatically on first run
- [ ] PlantWild: Run `sbatch scripts/submit_setup_plantwild.sh` before Phase 1
- [ ] Check: `ls data/PlantWild/ | wc -l` should show ~18000 files
- [ ] Ready: Run `bash scripts/submit_all_phases.sh`

---

**Last Updated:** 2026-05-01
