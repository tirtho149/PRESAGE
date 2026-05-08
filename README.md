# PlantSwarm + PathomeDB + OBSERVE — Train on the Wild

**Paper:** *Train on the Wild: Geospatial Multi-Agent Routing for Cross-Crop Plant Disease Diagnosis from Ten Field Images* (EMNLP 2026, anonymous submission). Source: `plantswarm/latex/acl_latex.tex`.

---

## TL;DR

A three-step loop that decouples *what a disease looks like* from *what a multi-agent VLM swarm sees when routed against it*:

1. **Seed** the visual content of the knowledge base by running the SAGE-ported `pathome_kb` pipeline (Claude headless web discovery → URL extraction with verbatim quotes → reconciliation per-source). Provenance-tracked: every visual fact carries `{value, url, quote}`.
2. **Trace** with PlantSwarm — 5 agents over Qwen2.5-VL-7B, 30 stochastic runs per Bugwood image — against the seeded KB. ~101k routing traces.
3. **Enhance** the KB by mining those traces into per-class `SwarmObservations` (path length, backtrack rate, confusion targets).

Then train OBSERVE twice — once on the seed-only KB, once on the enhanced KB — and report the seed→enhanced delta on the full PlantVillage and PlantWild benchmarks. **The headline result is the delta.**

```
  ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
  │  Phase 0 SEED  │  →   │ Phase 1 BUILD  │  →   │ Phase 2 TRACES │
  │ Claude headless│      │ symptoms.json +│      │ Qwen2.5-VL-7B  │
  │ writes 484     │      │ state/AEZ geo +│      │ × 5 agents     │
  │ VisualSymptom  │      │ 1,452 refs from│      │ × 30 runs      │
  │ blocks (LOCAL) │      │ Bugwood CSV    │      │ = 101k traces  │
  └────────────────┘      └────────────────┘      └────────┬───────┘
                                                            │
  ┌────────────────┐      ┌────────────────┐      ┌────────▼───────┐
  │ Phase 5 COMPARE│  ←   │ Phase 4 TRAIN  │  ←   │ Phase 3 ENHANCE│
  │ before / after │      │ OBSERVE × 2    │      │ mine traces →  │
  │ ΔT3 F1, ΔECE,  │      │ (seed DB,      │      │ SwarmObserva-  │
  │ ΔPathLen, …    │      │  enhanced DB)  │      │ tions per class│
  └────────────────┘      └────────────────┘      └────────────────┘
```

PathomeDB is two stores: `db.symptoms` (`SymptomLibrary`) and `db.refs` (`ReferenceLibrary`). The earlier 5-layer split (mechanistic pathway / cross-crop manifestation / regional epidemiology / decision graph / references) was retired in the post-CSV migration — see [`MIGRATION.md`](MIGRATION.md).

---

## Where each phase runs

The pipeline splits across two machines:

```
   ┌──────────────────────────┐                 ┌────────────────────────┐
   │       LOCAL machine      │     GitHub      │     Nova compute       │
   │  (laptop / workstation)  │      git        │  (SLURM-scheduled GPU) │
   └──────────────────────────┘                 └────────────────────────┘

   Phase 0  Claude-headless    ────push──→     git pull
   KB build (~30 min – 20 h)                       ↓
       ↓                                       Setup    Filter Bugwood CSV
   symptoms_seed.json                          Phase 1  Build PathomeDB
                                               Phase 2  PlantSwarm traces  (A100)
                                               Phase 3  Enhance from traces
                                               Phase 4  Train OBSERVE × 2  (A100)
                                               Phase 5  Eval × 4 + compare
```

**Why the split.** Phase 0 needs the `claude` CLI's OAuth login flow, which Nova compute nodes don't allow. Everything else is pure compute (Python + Qwen2.5-VL-7B) and runs as ordinary SLURM jobs.

**Handoff.** Phase 0 produces a single JSON file (`artifacts/pathome_seed/symptoms_seed.json`, a few MB). You `git add -f` + push it from your laptop and `git pull` it on Nova. The chain script bails out with a clear error if that file isn't present, so you can't accidentally start Phase 1 without the seed.

---

## Repository layout

```
PlantSwarm/
├── BugWood_Diseases.csv                  raw IPMNet export (committed)
├── BugWood_Diseases_usable.csv           filtered subset (committed; regenerable via Setup)
├── bugwood_classes_report.tsv            per-class candidate counts
│
├── configs/
│   ├── bugwood_pathome.yaml              training config (single source of truth)
│   ├── plantvillage_full_eval.yaml       held-out PV eval
│   └── plantwild_full_eval.yaml          held-out PW eval
│
├── pathome_kb/                           Phase 0 — SAGE-ported KB build (LOCAL)
│   ├── pipeline.py                       per-crop orchestrator + seed merge
│   ├── internet_pipeline.py              discovery → extraction → reconciliation
│   ├── shared.py                         Anthropic SDK + claude -p wrapper
│   ├── symptoms_adapter.py               SAGE registry → SymptomProfile JSON
│   ├── prompts/                          discovery / extraction / reconciliation
│   ├── utils.py, config.py
│   └── __main__.py                       python -m pathome_kb …
│
├── pathome/                              Phase 1+ — PathomeDB stores
│   ├── database.py                       PathomeDB orchestrator
│   ├── symptoms.py                       SymptomLibrary, SymptomProfile,
│   │                                     VisualSymptom, Citation, SwarmObservations
│   └── layer5_references.py              ReferenceLibrary (CLIP + FAISS)
│
├── data/bugwood_loader.py                CSV → BugwoodRecord stream
├── plantswarm/                           multi-agent routing pipelines
├── observe/                              OBSERVE student (Qwen2.5-VL-7B + LoRA + DT + GRPO)
├── agents/                               5 routing agents
├── utils/                                geo (state centroid + AEZ), trace I/O, vLLM/HF
├── calibration/                          ECE, temperature scaling, conformal
│
├── scripts/
│   ├── filter_bugwood_csv.py             setup: CSV → filtered usable CSV
│   ├── seed_pathome_with_claude.py       legacy schema-driven seeder (optional fallback)
│   ├── build_pathome.py                  Phase 1 — build PathomeDB
│   ├── run_pathome_traces.py             Phase 2 — PlantSwarm trace generation
│   ├── enhance_pathome_from_traces.py    Phase 3 — trace mining
│   ├── train_observe_pathome.py          Phase 4 — DT + GRPO
│   ├── evaluate_pathome.py               Phase 5a — held-out eval
│   ├── compare_pathome_versions.py       Phase 5b — comparison.{json,md,tex}
│   ├── sync_pathome_metrics.py           LaTeX macro emitter
│   ├── run_phase0_local.sh               LOCAL: Phase 0 wrapper
│   ├── submit_pathome_setup_filter.sh    NOVA: filter CSV (~30 s, CPU)
│   ├── submit_pathome_phase1_build.sh    NOVA: build DB (~30 min, CPU+net)
│   ├── submit_pathome_phase2_traces.sh   NOVA: traces (~36–50 h, A100+vLLM)
│   ├── submit_pathome_phase3_enhance.sh  NOVA: enhance (~5 min, CPU)
│   ├── submit_pathome_phase4_train.sh    NOVA: OBSERVE × 2 (~24 h, A100)
│   ├── submit_pathome_phase5_eval.sh     NOVA: eval+compare (~6–8 h, A100)
│   └── submit_pathome_all.sh             NOVA: chain Setup + Phases 1–5
│
├── smoke/                                end-to-end smoke (2 crops; see smoke/README.md)
│
├── artifacts/                            pipeline outputs (gitignored; seed pushed via -f)
├── results/                              eval JSONs + comparison artefacts (gitignored)
│
├── plantswarm/latex/acl_latex.tex        the paper
├── MIGRATION.md                          what changed across the symptom-centric refactor
└── README.md                             (this file)
```

---

## One-time prerequisites

### On your local machine

```bash
git clone https://github.com/tirtho149/PlantSwarm.git
cd PlantSwarm

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

For Phase 0 you also need:

```bash
# Claude Code CLI (used for the discovery WebSearch stage)
curl -fsSL https://claude.ai/install.sh | bash
claude auth login        # OAuth in browser

# Anthropic SDK key (used for the extraction + reconciliation stages)
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

The Bugwood IPMNet CSV (`BugWood_Diseases.csv`) is committed. If you ever pull a fresh export, replace the file at the repo root and re-run the Setup step on Nova.

### On Nova

```bash
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/
git clone https://github.com/tirtho149/PlantSwarm.git    # first time only
cd PlantSwarm
mkdir -p logs

module load python cuda/11.8
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r requirements-tfds.txt        # for the held-out PV eval
```

Optional: get the FAO GAEZ shapefile to upgrade Layer 3 from the 2-zone coarse fallback to the full ~17-zone resolution. Add to `~/.bashrc` so SLURM sees it:

```bash
export PATHOME_AEZ_SHAPEFILE=/path/to/FAO_AEZv4_50K.shp
```

---

## Smoke test first (recommended)

Before kicking off the multi-day full run, validate every code path on a 2-crop / ~25-class subset (~60–90 min on a single A100). Same local→Nova split as production:

```bash
# === LOCAL (laptop), ~5 min ===
bash smoke/run_phase0_local.sh
git add -f smoke/artifacts/pathome_seed/symptoms_seed.json \
           smoke/BugWood_Diseases_smoke_usable.csv
git commit -m "smoke phase 0 seed" && git push origin main

# === NOVA, single A100 job, ~60-90 min ===
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/PlantSwarm && git pull origin main
sbatch smoke/submit_smoke.sh
tail -f logs/pathome_smoke-*.out
```

See [`smoke/README.md`](smoke/README.md) for what's downscaled, skip/resume knobs, and the expected outputs.

---

## Quick start (full pipeline)

```bash
# === LOCAL ===
bash scripts/run_phase0_local.sh
# 12-20 h full / ~30 min --quick. See Phase 0 section for cost + flags.

git add -f artifacts/pathome_seed/symptoms_seed.json
git add -f artifacts/pathome_kb/                    # optional audit trail
git commit -m "phase 0 seed" && git push origin main

# === NOVA ===
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/PlantSwarm && git pull origin main
bash scripts/submit_pathome_all.sh
# Setup → Phase 1 → 2 → 3 → 4 → 5 (sbatch dependency chain)
```

Skip steps that are already done:
```bash
PATHOME_SKIP="setup"     bash scripts/submit_pathome_all.sh   # CSV already filtered
PATHOME_FROM_PHASE=4     bash scripts/submit_pathome_all.sh   # restart at training
```

Monitor:
```bash
squeue -u $USER
tail -f logs/pathome_*-*.out
```

Final output drops at `results/pathome_compare/comparison.md`.

---

## Step-by-step

### Phase 0 — Build the seed PathomeDB knowledge base (LOCAL only)

`scripts/run_phase0_local.sh` → `python -m pathome_kb`

> ⚠ **Runs on your local machine, not on Nova.** Nova compute nodes block the OAuth login flow that `claude` headless needs.

Three stages per crop:

```
discovery       claude -p WebSearch per disease (parallel)  →  candidate URLs
                          │
                          ▼
extraction      fetch each URL  →  claude -p extracts disease records with
                VERBATIM QUOTES from the page text (never invents content)
                          │
                          ▼
reconciliation  merge per-source records into a canonical entry per disease.
                Every field stored as {value, url, quote}, so each visual
                fact in the KB is traceable to the exact sentence on the
                exact source page that supports it.
```

The orchestrator groups the 484 classes by crop, runs the internet track once per crop (so each discovery search focuses on one crop's disease catalogue), and merges per-crop registries into a single `symptoms_seed.json`.

| | |
|---|---|
| **Where it runs** | LOCAL machine |
| **Compute** | CPU; outbound HTTPS for `api.anthropic.com` + per-source page fetches |
| **Walltime** | Quick mode (3 sources/crop): ~30 min. Full run (197 crops × ~5–15 sources each): 12–20 h |
| **Inputs** | `BugWood_Diseases_usable.csv`, authenticated `claude` CLI, `ANTHROPIC_API_KEY` |
| **Outputs (local disk)** | `artifacts/pathome_kb/<Crop>/{discovery_results,raw_extractions,final_registry}.json` + `registry.md` + `internet.xlsx`; merged `artifacts/pathome_seed/symptoms_seed.json` |
| **Handoff** | `git add -f artifacts/pathome_seed/symptoms_seed.json && git commit && git push` |
| **Knobs** | `PATHOME_SEED_QUICK=1`, `PATHOME_SEED_LIMIT=N`, `PATHOME_SEED_ONLY_CROPS="Tomato,Soybean"`, `PATHOME_SEED_RESUME=discovery\|extraction\|reconciliation`, `PATHOME_SEED_NO_CACHE=1` |
| **Resume** | Two levels. (a) per-crop: any crop with an existing `final_registry.json` is skipped on rerun (override with `PATHOME_SEED_NO_CACHE=1`). (b) per-stage within a crop: `--resume-from extraction` reuses `discovery_results.json` already on disk. |
| **Cost** | ~$50–150 in Anthropic API spend for a full run; ~$5 for quick. |

```bash
# Quick smoke (~30 min, ~$5) on Tomato + Soybean + Corn
PATHOME_SEED_QUICK=1 PATHOME_SEED_ONLY_CROPS="Tomato,Soybean,Corn" \
  bash scripts/run_phase0_local.sh

# Full run (12-20 h, ~$50-150)
bash scripts/run_phase0_local.sh

# Resume only the reconciliation stage (assumes raw_extractions.json present)
PATHOME_SEED_RESUME=reconciliation bash scripts/run_phase0_local.sh

# Force every crop to re-run from scratch
PATHOME_SEED_NO_CACHE=1 bash scripts/run_phase0_local.sh
```

The script prints exact `git add -f` / `commit` / `push` commands when finished — copy-paste them.

### Setup — Filter Bugwood CSV (Nova)

`scripts/submit_pathome_setup_filter.sh`

| | |
|---|---|
| **Purpose** | Normalise the raw IPMNet export into the per-class-thresholded subset the pipeline trains on. |
| **Compute** | 2 CPUs, 4 GB RAM, no GPU |
| **Walltime** | ~30 s |
| **Inputs** | `BugWood_Diseases.csv` |
| **Outputs** | `BugWood_Diseases_usable.csv` (~11,513 rows / 484 classes), `bugwood_classes_report.tsv` |
| **Knobs** | `PATHOME_THRESHOLD` (default 10 rows/class; `15`→263 classes, `5`→982) |

```bash
sbatch scripts/submit_pathome_setup_filter.sh
PATHOME_THRESHOLD=15 sbatch scripts/submit_pathome_setup_filter.sh
```

### Phase 1 — Build PathomeDB v1_seed (Nova)

`scripts/submit_pathome_phase1_build.sh`

| | |
|---|---|
| **Purpose** | Layer the Claude seed JSON over the filtered CSV. Produces `SymptomLibrary` (visual + per-state + per-AEZ counts + reference IDs) and `ReferenceLibrary` (1,452 held-out images, lazily CLIP-indexed on first retrieval). |
| **Compute** | 8 CPUs, 32 GB RAM, no GPU, network for first-time Bugwood image downloads |
| **Walltime** | 6 h budget; ~30 min on first run, instant on subsequent (cache hit) |
| **Inputs** | `configs/bugwood_pathome.yaml`, `BugWood_Diseases_usable.csv`, `artifacts/pathome_seed/symptoms_seed.json` |
| **Outputs** | `artifacts/pathome_v1_seed/{symptoms.json, refs/, version.txt, build_summary.json}` |
| **Knobs** | `PATHOME_CONFIG`, `PATHOME_SEED_FILE`, `PATHOME_OUT_DIR` |

```bash
sbatch scripts/submit_pathome_phase1_build.sh
```

### Phase 2 — Generate PlantSwarm traces (Nova)

`scripts/submit_pathome_phase2_traces.sh`

| | |
|---|---|
| **Purpose** | Run the 5-agent swarm over Qwen2.5-VL-7B against the seeded PathomeDB. 3,388 trace seeds × 30 stochastic runs at T=0.9 = **101,640 traces**. |
| **Compute** | 1× A100-80GB, 8 CPUs, 64 GB RAM; vLLM booted in-job |
| **Walltime** | 72 h budget; typical ~36–50 h |
| **Inputs** | `artifacts/pathome_v1_seed/`, `BugWood_Diseases_usable.csv`, Qwen weights (HF cache) |
| **Outputs** | `results/bugwood_seed/traces/plantswarm_traces.jsonl` (one JSON per trace, fsynced) |
| **Knobs** | `PATHOME_DB_DIR`, `PATHOME_OUT_DIR` |
| **Resume** | Yes — already-persisted `image_id`s are skipped on resubmit. Walltime kill is recoverable. |

```bash
sbatch scripts/submit_pathome_phase2_traces.sh
```

If vLLM fails to boot, the loader falls back to `hf_direct` mode automatically (slower but memory-safe after the recent allocator fix).

### Phase 3 — Enhance DB from traces (Nova)

`scripts/submit_pathome_phase3_enhance.sh`

| | |
|---|---|
| **Purpose** | Mine the traces into per-class `SwarmObservations` (n_traces, avg_path_length, backtrack_rate, high_confidence_rate, confusion_targets) attached to the matching `SymptomProfile`. Visual blocks left untouched — enhancement is strictly additive. |
| **Compute** | 4 CPUs, 16 GB RAM, no GPU |
| **Walltime** | 1 h budget; ~5 min in practice |
| **Inputs** | `artifacts/pathome_v1_seed/`, `results/bugwood_seed/traces/plantswarm_traces.jsonl` |
| **Outputs** | `artifacts/pathome_v1_enhanced/{symptoms.json, refs/, enhancement_summary.json}` |

```bash
sbatch scripts/submit_pathome_phase3_enhance.sh
```

### Phase 4 — Train OBSERVE × 2 (Nova)

`scripts/submit_pathome_phase4_train.sh`

| | |
|---|---|
| **Purpose** | Train OBSERVE twice on the same trace set, differing only in which PathomeDB the agents read from at training time. Each run does Phase A (Decision Transformer) + Phase B (GRPO). |
| **Compute** | 1× A100-80GB, 8 CPUs, 128 GB RAM |
| **Walltime** | 24 h budget; ~10–14 h DT + ~6–8 h GRPO per checkpoint, sequential |
| **Inputs** | `artifacts/pathome_v1_seed/`, `artifacts/pathome_v1_enhanced/`, traces from Phase 2, `configs/bugwood_pathome.yaml` |
| **Outputs** | `observe/checkpoints/seed/observe_grpo_epoch_*.pt`, `observe/checkpoints/enhanced/observe_grpo_epoch_*.pt` |

```bash
sbatch scripts/submit_pathome_phase4_train.sh
```

### Phase 5 — Eval + before/after compare (Nova)

`scripts/submit_pathome_phase5_eval.sh`

| | |
|---|---|
| **Purpose** | Evaluate both checkpoints on full PV (with seen/unseen slice) and full PW; emit the headline before/after artefact via `compare_pathome_versions.py`. |
| **Compute** | 1× A100-80GB, 8 CPUs, 64 GB RAM; one vLLM instance reused across all four evaluations |
| **Walltime** | 8 h budget; typical ~6 h |
| **Inputs** | both OBSERVE checkpoints, `configs/plantvillage_full_eval.yaml`, `configs/plantwild_full_eval.yaml`, traces from Phase 2 |
| **Outputs** | `results/pathome_compare/{seed,enhanced}/{pv,pw}/pathome_eval.json`, `results/pathome_compare/comparison.{json,md,tex}` |

```bash
sbatch scripts/submit_pathome_phase5_eval.sh
```

The `comparison.tex` file emits LaTeX macros (`\PathomeDeltaTthreeF`, `\PathomeDeltaTthreeECE`, `\PathomeDeltaPathLen`, …) which the paper picks up via `\input{auto_pathome_metrics}` near the headline before/after table.

---

## Configuration

Single source of truth: `configs/bugwood_pathome.yaml`. Most-tweaked knobs:

```yaml
data:
  csv_path: "BugWood_Diseases_usable.csv"
  per_class: 10              # max images per (crop, disease)
  trace_split: 7             # first N → trace seeds; remainder → references
  min_per_class: 10          # drop classes below this row count

routing:
  orchestrator: "autogen_swarm"  # or "hf_direct" for single-GPU fallback
  Tmax: 15                       # max path length per trace
  runs_per_image: 30             # stochastic runs per Bugwood seed image

model:
  backbone: "Qwen/Qwen2.5-VL-7B-Instruct"
  temperature: 0.9
  vllm_base_url: "http://localhost:8000/v1"

observe:
  backbone: "Qwen/Qwen2.5-VL-7B-Instruct"
  oc_threshold: 0.55             # paper §7.2 overconfidence cutoff
  decision_transformer:
    epochs: 50
    patience: 5
  grpo:
    epochs: 10
    rollouts_per_instance: 8
    beta_kl: 0.04
```

The two eval configs (`plantvillage_full_eval.yaml`, `plantwild_full_eval.yaml`) override `data.*` and `output.results_dir` only.

---

## Troubleshooting

### Phase 0 errors

| Symptom | Fix |
|---|---|
| `claude CLI not on PATH` | `curl -fsSL https://claude.ai/install.sh \| bash` then `claude auth login` |
| `ANTHROPIC_API_KEY not set` | `echo "ANTHROPIC_API_KEY=sk-ant-..." > .env` at repo root |
| `claude -p timed out` | A specific source page is slow. Re-run; that source is now cached. |
| `failed.jsonl` lists profiles | Re-run with `--retry-failed` once you've fixed the underlying issue (rate limit, quota, etc.) |

### Phase 1 errors

| Symptom | Fix |
|---|---|
| Bugwood download fails for some images | Ignored — Phase 1 emits records with `image=None` and a `<imgid>.failed` sidecar in `.bugwood_cache/`. Phase 2 will skip those traces. |

### Phase 2 / 4 / 5 — vLLM fails to boot

`logs/vllm-<JOB>.log` has the stderr. To force the HF-direct fallback (slower but memory-safe):
```bash
PLANTSWARM_MODE=hf_direct sbatch scripts/submit_pathome_phase2_traces.sh
```

### CUDA OOM mid-run (HF direct only)

The HFClient is patched to release reserved-but-unallocated GPU memory after every generation. If you still see OOM:

1. Confirm the SLURM script exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (already in current scripts).
2. Drop `model.max_new_tokens` from `512` → `256` in `configs/bugwood_pathome.yaml`.
3. Drop the image cap in `utils/hf_client.py:_MAX_IMAGE_SIDE` from `1024` → `768`.
4. Last resort: switch to vLLM (paged KV cache).

### Walltime kill mid-trace-generation

Trace JSONL is appended with fsync after each trace. Already-persisted `image_id`s are skipped on resubmit; just `sbatch` again.

### Layer-3 prior is degenerate

The coarse-fallback AEZ table maps the entire US footprint into 2 zones (TMP, STM). Set `PATHOME_AEZ_SHAPEFILE` to a real FAO GAEZ shapefile to recover ~17-zone resolution. State-level priors via `state_counts` work either way.

---

## Output directory map

After a complete run:

```
PlantSwarm/
├── BugWood_Diseases_usable.csv               (Setup)
├── bugwood_classes_report.tsv                (Setup)
│
├── artifacts/                                 [gitignored except seed]
│   ├── pathome_kb/<Crop>/                    (LOCAL, Phase 0 — per-crop audit)
│   │   ├── discovery_results.json
│   │   ├── raw_extractions.json
│   │   ├── final_registry.json
│   │   ├── registry.md
│   │   └── internet.xlsx
│   ├── pathome_seed/                         (LOCAL → push via git -f)
│   │   └── symptoms_seed.json
│   ├── pathome_v1_seed/                      (NOVA, Phase 1)
│   │   ├── symptoms.json
│   │   ├── refs/
│   │   ├── version.txt
│   │   └── build_summary.json
│   └── pathome_v1_enhanced/                  (NOVA, Phase 3)
│       ├── symptoms.json
│       ├── refs/
│       └── enhancement_summary.json
│
├── results/                                   [gitignored]
│   ├── bugwood_seed/
│   │   └── traces/plantswarm_traces.jsonl    (NOVA, Phase 2)
│   └── pathome_compare/
│       ├── seed/{pv,pw}/pathome_eval.json    (NOVA, Phase 5)
│       ├── enhanced/{pv,pw}/pathome_eval.json (NOVA, Phase 5)
│       ├── comparison.json                    (NOVA, Phase 5)
│       ├── comparison.md                      ← main output
│       └── comparison.tex                     ← paper macros
│
├── observe/checkpoints/                       [gitignored]
│   ├── seed/observe_grpo_epoch_*.pt          (NOVA, Phase 4)
│   └── enhanced/observe_grpo_epoch_*.pt      (NOVA, Phase 4)
│
└── logs/pathome_*-*.{out,err}                SLURM stdout/stderr
```

---

## Sync workflow

```
   ┌──────────┐  Phase 0 push  ┌────────┐  git pull   ┌──────┐
   │  Local   │───────────────→│ GitHub │────────────→│ Nova │
   │          │                │        │             │      │
   │          │   results pull │        │ Phase 5 push│      │
   │          │←───(rsync)─────│        │←────────────│      │
   └──────────┘                └────────┘             └──────┘
```

**Phase 0 push (Local → Nova):**
```bash
# After bash scripts/run_phase0_local.sh finishes:
git add -f artifacts/pathome_seed/symptoms_seed.json
git add -f artifacts/pathome_kb/                   # optional audit trail
git commit -m "phase 0 seed"
git push origin main
```

**Results pull (Nova → Local):**
```bash
# results/ and artifacts/pathome_v1_*/ stay gitignored. Pull via rsync:
rsync -avz nova-login:/work/mech-ai-scratch/tirtho/PlantSwarm/results/ ./results/
rsync -avz nova-login:/work/mech-ai-scratch/tirtho/PlantSwarm/artifacts/pathome_v1_enhanced/ \
           ./artifacts/pathome_v1_enhanced/
# OR commit just the comparison artefacts:
ssh nova-login "cd /work/.../PlantSwarm && \
  git add -f results/pathome_compare/comparison.{json,md,tex} && \
  git commit -m 'Phase 5 results' && git push"
git pull
cat results/pathome_compare/comparison.md
```

---

## Compile the paper

```bash
cd plantswarm/latex
latexmk -pdf acl_latex.tex
```

If you've run Phase 5, `\input{auto_pathome_metrics}` near the headline table picks up the `\PathomeDelta*` macros emitted by `compare_pathome_versions.py` and the table fills in automatically.

---

## Known limitations

- **US-only data.** The Bugwood IPMNet CSV is US-only at state granularity. International deployment requires a different export with finer GPS or a separate regional KB.
- **2-zone AEZ fallback.** Coarse FAO AEZ table maps the US footprint into 2 zones; full ~17-zone resolution needs `PATHOME_AEZ_SHAPEFILE` pointing at a real GAEZ shapefile.
- **Half of classes are single-state.** ~248 of 484 admitted classes appear in only one state, contributing no spatial-variance signal; the geo prior is informative on the multi-state subset only.
- **No monthly priors.** The IPMNet CSV has no capture date; the AEZ-month grid + EPPO Pearson-r validation are dropped from the methodology (paper §6 reflects this).
- **Phase 0 cost variance.** `claude -p` seed quality varies with disease prevalence in Claude's training data; rare or recently-described diseases may return empty visual fields.
- **Phase 0 isn't on Nova.** The local→GitHub→Nova handoff means a fresh full Phase 0 commits ~few-MB seed file (and optionally ~50–100 MB of per-crop registry artefacts) to git history.

---

## Citations

```bibtex
@inproceedings{plantswarm2026,
  title     = {Train on the Wild: Geospatial Multi-Agent Routing for
               Cross-Crop Plant Disease Diagnosis from Ten Field Images},
  author    = {Anonymous},
  booktitle = {Proceedings of EMNLP 2026},
  year      = {2026}
}
```

Bugwood IPMNet images are publicly available under academic and extension-service terms — see [bugwood.org](https://www.bugwood.org/) for citation expectations on individual images.
