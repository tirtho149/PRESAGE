# Pathome smoke test (2 crops, full pipeline)

A miniature end-to-end run of the Pathome pipeline on **Tomato + Soybean only** (~25 (crop, disease) classes after threshold ≥ 15). Designed to validate every code path with as little compute as possible.

**Why smoke first.** The production pipeline costs ~$50–150 in API spend (Phase 0) plus ~36–50 hours of A100 time (Phase 2) plus ~24 hours of A100 time (Phase 4). A smoke run completes in ~60–90 minutes on a single A100 for ~$5–15 (Phase 0) and exercises every phase, so plumbing issues surface before you commit to the full spend.

---

## Directory layout

```
smoke/
├── BugWood_Diseases_smoke.csv         1,002 raw rows (Tomato + Soybean)
├── BugWood_Diseases_smoke_usable.csv  filtered + normalized (produced by Setup)
├── bugwood_pathome_smoke.yaml         training config — small budgets
├── plantvillage_smoke_eval.yaml       PV eval (200-image subset)
├── plantwild_smoke_eval.yaml          PW eval (200-image subset)
├── run_phase0_full.sh                 LOCAL — Phase 0 full coverage (~45–90 min, $5–15)
├── run_phase0_local.sh                LOCAL — Phase 0 quick mode    (~5–15 min, $1–3)
├── run_smoke.sh                       Bash chain — every phase as plain `python`
├── submit_smoke.sh                    NOVA SLURM wrapper (Phases 1–5)
└── README.md                          (this file)
```

## Outputs

```
LOCAL — Phase 0:
  smoke/.bugwood_cache/<image>.jpg                       cached Bugwood JPGs (one per crop × disease × state)
  artifacts/pathome_kb/<Crop>/discovery_results.json     candidate URLs
  artifacts/pathome_kb/<Crop>/final_registry.json        canonical + regional_observations
  artifacts/pathome_kb/<Crop>/final_registry.xlsx        decision-tree Excel view
  smoke/artifacts/pathome_seed/symptoms_seed.json        ← Phase 1 input (push via git -f)

NOVA — Phases 1–5:
  smoke/artifacts/pathome_v1_seed/                                Phase 1  PathomeDB v1_seed
  smoke/results/traces/plantswarm_traces.jsonl                    Phase 2  canonical training corpus (one trace per line)
  smoke/results/traces/per_image/<image_id>__run<NN>.json         Phase 2  one pretty-printed JSON per (image, run) — visual mirror
  smoke/artifacts/pathome_v1_enhanced/                            Phase 3  enhanced DB
  smoke/observe/checkpoints/{seed,enhanced}/                      Phase 4  LoRA + GRPO checkpoints
  smoke/results/compare/{seed,enhanced}/{pv,pw}/                  Phase 5  per-condition eval json
  smoke/results/compare/comparison.{json,md,tex}                  Phase 5  seed-vs-enhanced delta table
```

**Yes — every PlantSwarm trace from Phase 2 is appended to `smoke/results/traces/plantswarm_traces.jsonl`.** One JSON object per line: `image_id`, T1–T5 labels, the routing path, per-step backbone logits, calibrated final probabilities. Phase 3 mines that file for new symptom evidence and Phase 4 consumes it as the Decision-Transformer + GRPO training corpus. ~225 traces total at smoke budget (25 classes × 3 trace seeds × 3 runs).

Everything under `smoke/` except the seed file is gitignored — Nova regenerates it from the committed seed.

---

## Data splits (train / val / test, image-disjoint)

The pipeline has three image-level splits, and they are guaranteed non-overlapping by construction:

| Split | Source | Carved out by | Smoke size | Used in |
|---|---|---|---|---|
| **train** | Bugwood | `BugwoodLoader(split="trace")` — first `trace_split=3` image numbers per (crop, disease), sorted ascending | ~75 images | Phase 2 trace generation (3 stochastic runs per image → ~225 traces) |
| **val** | Bugwood | `BugwoodLoader(split="val")` — next `per_class - trace_split = 1` image number per class | ~25 images | held-out for in-domain Bugwood evaluation (manifest at `smoke/artifacts/pathome_v1_seed/bugwood_val_manifest.json`) |
| **test** | PlantVillage + PlantWild | `PlantDiagBenchLoader(split="test")` | 200 + 200 | Phase 5 — completely separate datasets, no Bugwood image can appear |

**Non-overlap guarantees (audited at load):**

- Train vs val: `BugwoodLoader` sorts by Image Number ascending, dedupes, then slices `[:trace_split]` vs `[trace_split:per_class]`. Image-number disjoint by construction. `scripts/build_pathome.py` runs an explicit set-intersection check at load time and aborts if any image_id lands in both.
- Train/val vs test: Bugwood, PlantVillage, and PlantWild are different datasets with disjoint image ID schemes — there is no way a Bugwood image_id (`bugwood::N`) can appear in a PV/PW eval loader.
- **Phase 4 internal 80/10/10 (`train_observe_pathome.py`):** the OBSERVE Decision-Transformer + GRPO split partitions **unique source image_ids** (stripping the `::run<NN>` suffix), then assigns all runs of each image to the same fold. Runs of the same Bugwood image never split across train/val/held inside Phase 4. (Old behaviour split per trace record, which leaked all three runs of one image across all three folds.)
- **PathomeDB Layer-5 (CLIP exemplar pool)** is intentionally seeded with `reference_records=[]` so the val split never appears as a retrieval hit during Phase 2 trace generation or Phase 5 eval.

**Back-compat:** `BugwoodLoader(split="reference")` is a silent alias for `split="val"` — old configs still load.

---

## What's downscaled vs production

| Knob | Production | Smoke |
|---|---|---|
| Crops | 197 | 2 (Tomato + Soybean) |
| Classes | 484 | ~25 (after threshold ≥ 15) |
| `per_class` / `trace_split` | 10 / 7 | 4 / 3 |
| `runs_per_image` | 30 | 3 |
| `max_new_tokens` | 512 | 256 |
| `Tmax` (max routing path) | 15 | 8 |
| Phase 0 mode | full | full *or* `--quick` (3 sources/crop) |
| Phase 4 DT epochs | 50 | 3 |
| Phase 4 GRPO epochs | 10 | 1 |
| LoRA rank | 16 | 8 |
| Phase 5 PV / PW eval images | 54,306 / 18,000 | 200 / 200 |
| `bootstrap_n` | 1,000 | 100 |

Override the threshold with `SMOKE_THRESHOLD=10` for a wider smoke (~65 classes), or `SMOKE_THRESHOLD=20` for narrower (~15 classes).

---

## Where each phase runs

```
   LOCAL machine                              GitHub               Nova compute
   ─────────────                              ──────               ────────────

   bash smoke/run_phase0_full.sh   ──push──→  git pull   ──→   sbatch smoke/submit_smoke.sh
   (~45–90 min, claude -p)                    symptoms_seed.json     Phases 1–5 (~60–90 min, A100)
```

Same split as production: Phase 0 needs the `claude` CLI's OAuth login (Nova compute can't run it), everything else runs as ordinary GPU jobs.

---

## Step 1 — LOCAL: Phase 0 (KB seed)

### 1.0 Prerequisites

```bash
claude --version          # Claude Code CLI on PATH
claude auth login         # if not already done

echo "ANTHROPIC_API_KEY=sk-ant-..." > .env   # at repo root
```

### 1.1 Pick a mode

| Mode | Script | Coverage | Runtime | Cost |
|---|---|---|---|---|
| **Full** (recommended) | `smoke/run_phase0_full.sh` | every source URL, every state, every visual field | ~45–90 min | ~$5–15 |
| Quick (dev iteration) | `smoke/run_phase0_local.sh` | 3 sources/crop, no per-state image deltas | ~5–15 min | ~$1–3 |

### 1.2 Run

```bash
# Override crops (default Soybean,Corn — for Tomato + Soybean use this):
SMOKE_CROPS="Tomato,Soybean" bash smoke/run_phase0_full.sh
```

This runs five internal sub-stages:

1. **Setup** — `scripts/filter_bugwood_csv.py` normalizes `Host Name → NormCrop`, drops blank-state rows, applies threshold ≥ 15. Writes `BugWood_Diseases_smoke_usable.csv` + a per-class report at `smoke/bugwood_classes_smoke.tsv`.

2. **State-aware image cache top-up** — `scripts/ensure_state_image_cache.py` downloads one Bugwood JPG per (crop, disease, state) into `smoke/.bugwood_cache/`. These are the photographs the regional pass examines.

3. **Cross-region SAGE pipeline** — `python -m pathome_kb`:
   - *Discovery* (claude -p WebSearch) finds all candidate extension-service URLs per disease.
   - *Extraction* (claude -p) pulls verbatim quotes + treatments per source.
   - *Reconciliation* (claude -p) merges per-source extractions into a single canonical block per disease (`summary`, `diagnostic_features`, `look_alikes`, `treatments`, `affected_parts`, `pathogen_scientific_name`, `type_of_disease`).
   - Writes `artifacts/pathome_kb/<Crop>/final_registry.json`.

4. **Per-state VLM observation (deltas only)** — claude -p with the Read tool examines each cached Bugwood image plus the canonical KB from step 3 as context. Emits **deltas only** — for each canonical field, what THIS image ADDS or CONTRADICTS. If the image confirms canonical exactly, `deltas=[]`. No parallel re-extraction of canonical fields. Deltas are embedded back into `final_registry.json` under `regional_observations[<state>]`.

5. **Adapter merge** — assembles the per-crop registries into `smoke/artifacts/pathome_seed/symptoms_seed.json` (the single file Nova consumes).

### 1.3 Visualize the KB (optional)

```bash
python scripts/registry_to_excel.py artifacts/pathome_kb/Tomato/final_registry.json
python scripts/registry_to_excel.py artifacts/pathome_kb/Soybean/final_registry.json
open artifacts/pathome_kb/{Tomato,Soybean}/final_registry.xlsx
```

Single-sheet decision-tree view: one row per disease, canonical fields on the left, all per-state deltas grouped into a single multiline cell on the right.

### 1.4 Push to GitHub

```bash
git add smoke/BugWood_Diseases_smoke.csv \
        smoke/BugWood_Diseases_smoke_usable.csv \
        smoke/bugwood_classes_smoke.tsv
git add -f smoke/artifacts/pathome_seed/symptoms_seed.json \
           artifacts/pathome_kb/Tomato/{discovery_results,final_registry}.json \
           artifacts/pathome_kb/Tomato/final_registry.xlsx \
           artifacts/pathome_kb/Soybean/{discovery_results,final_registry}.json \
           artifacts/pathome_kb/Soybean/final_registry.xlsx
git commit -m "smoke phase 0 seed"
git push origin main
```

The `run_phase0_full.sh` wrapper prints these exact commands when it finishes — copy-paste them.

---

## Step 2 — NOVA: Phases 1–5 (~60–90 min, single A100)

```bash
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/PlantSwarm
git pull origin main
sbatch smoke/submit_smoke.sh
tail -f logs/pathome_smoke-*.out
```

`submit_smoke.sh` is a single A100 job (4 h walltime budget) that internally chains the next five phases by invoking `bash smoke/run_smoke.sh` with `SMOKE_SKIP_0=1`.

**Pre-flight:** the job bails out with a clear error if `smoke/artifacts/pathome_seed/symptoms_seed.json` isn't on disk — meaning your `git push` hasn't reached the remote, or Nova hasn't `git pull`-ed yet.

### Phase 1 — Build PathomeDB v1_seed (~1–2 min, CPU)

`scripts/build_pathome.py --config smoke/bugwood_pathome_smoke.yaml` reads `symptoms_seed.json` + the filtered CSV and assembles the five PathomeDB layers (T1–T5 label space, region ontology, disease references, symptom profiles, treatment ladders).

Output: `smoke/artifacts/pathome_v1_seed/`.

### Phase 2 — PlantSwarm trace generation (~10–20 min, **GPU**)

`scripts/run_pathome_traces.py` runs Qwen2.5-VL-7B as the PlantSwarm orchestrator (default `hf_direct`; switch to vLLM with `SMOKE_ORCH=autogen_swarm`) for `per_class=4 × trace_split=3 × runs_per_image=3` per class.

**Every completed trace lands on disk in two places in real time:**

```
smoke/results/traces/plantswarm_traces.jsonl                  ← canonical (one line per trace)
smoke/results/traces/per_image/<image_id>__run<NN>.json       ← per-image mirror (pretty-printed)
```

After every trace, the writer (`scripts/run_pathome_traces.py`):

1. **Appends one JSON line** to `plantswarm_traces.jsonl` and calls `f.flush() + os.fsync()` — this is the canonical training corpus consumed by Phases 3 + 4.
2. **Writes a pretty-printed JSON file** to `traces/per_image/<safe_id>.json` via tmp + atomic `os.replace()` — one file per `(image, run)` pair, e.g. `bugwood_1568038_run00.json`. A kill mid-write leaves no partial files.

So:

- the JSONL grows line-by-line as the run progresses — `tail -f` works live during Phase 2,
- the `per_image/` directory fills up file-by-file, ~225 files at smoke budget — open any one in an editor / `jq` to inspect a single trace without grep'ing the big JSONL,
- a killed / pre-empted job is fully resumable: re-submitting the SLURM script reads the existing JSONL via `existing_trace_ids()` and skips any `<image_id>::run<N>` already on disk — only the missing traces are re-run,
- there is no buffering window where a crash would lose completed traces.

One JSON line per trace. At smoke budget that's ~225 lines total. Each line includes:

- `image_id`, `crop`, `disease`, `state`, `bugwood_meta` (GPS, AEZ, month — paper §5.4)
- T1–T5 labels (`symptom_type`, `pathogen_class`, `disease_name`, `severity_class`, `crop_species`)
- the full routing path (which expert nodes the swarm visited, in order)
- per-step backbone logits + confidence levels (`high`/`medium`/`low`)
- final calibrated probability distribution
- `entropy_field` (per-step token entropy, for the OBSERVE policy's calibration loss)

Phase 3 reads this file to mine symptom evidence; Phase 4 reads it as the training corpus for the OBSERVE Decision Transformer + GRPO loop.

**Watch it stream live (Nova):**

```bash
# in another terminal while the job runs — tail the canonical JSONL
ssh tirtho@hpc-login.iastate.edu \
  'tail -f /work/mech-ai-scratch/tirtho/PlantSwarm/smoke/results/traces/plantswarm_traces.jsonl | jq -c "{id:.trace_id, T5:.labels.T5, T3:.labels.T3, path_len:(.routing_path|length)}"'

# …or watch the per-image directory fill up
ssh tirtho@hpc-login.iastate.edu \
  'watch -n 2 "ls /work/mech-ai-scratch/tirtho/PlantSwarm/smoke/results/traces/per_image | wc -l"'

# inspect one trace
ssh tirtho@hpc-login.iastate.edu \
  'jq . /work/mech-ai-scratch/tirtho/PlantSwarm/smoke/results/traces/per_image/bugwood_1568038_run00.json | less'
```

### Phase 3 — Enhance DB from traces (~1 min, CPU)

`scripts/enhance_pathome_from_traces.py --seed-db <v1_seed> --traces <jsonl> --out <v1_enhanced>` reads the trace JSONL, mines additional symptom evidence + treatment co-occurrences, and writes an enhanced DB. Both `v1_seed` and `v1_enhanced` are kept side-by-side for the comparison in Phase 5.

Output: `smoke/artifacts/pathome_v1_enhanced/`.

### Phase 4 — Train OBSERVE × 2 (~30–40 min each = ~60–80 min, **GPU**)

`scripts/train_observe_pathome.py` runs **two** training passes — one against `v1_seed`, one against `v1_enhanced`. Each pass consists of:

1. **Decision Transformer warm-start** — 3 epochs at smoke budget (production: 50). Loads the trace JSONL as (state, action, return-to-go) tuples; predicts the next routing action conditioned on a target return.
2. **GRPO fine-tuning** — 1 epoch at smoke budget (production: 10). 4 rollouts per instance, reward = `f1 - 0.4·ECE - 0.3·BT_delta - 0.05·length - 0.2·epsilon_mismatch`.

LoRA rank 8 (production: 16). Checkpoints:

```
smoke/observe/checkpoints/seed/observe_grpo_epoch_01.pt
smoke/observe/checkpoints/enhanced/observe_grpo_epoch_01.pt
```

### Phase 5 — Eval × 4 + comparison (~5–10 min, **GPU**)

`scripts/evaluate_pathome.py` runs **four** evals — seed×PV, seed×PW, enhanced×PV, enhanced×PW — each over a 200-image subset. Per-eval output:

```
smoke/results/compare/{seed,enhanced}/{pv,pw}/pathome_eval.json
```

`scripts/compare_pathome_versions.py` joins them into a single seed-vs-enhanced delta table:

```
smoke/results/compare/comparison.json   machine-readable
smoke/results/compare/comparison.md     human-readable (the headline artefact)
smoke/results/compare/comparison.tex    LaTeX table fragment for the paper
```

### What success looks like

After a clean run, `smoke/results/compare/comparison.md` contains the seed-vs-enhanced delta table for PV + PW evals. Numbers are tiny and not statistically meaningful (n=200 each, 1 GRPO epoch) — the pipeline producing the table end-to-end is the point.

If `comparison.md` is non-empty plus matching `.json` + `.tex` siblings, every phase wired correctly. From there, `scripts/submit_pathome_all.sh` runs the same code at full scale.

---

## Local-only debug (no Nova, no GPU)

If you don't have a GPU and just want to validate KB plumbing:

```bash
bash smoke/run_smoke.sh
# Setup → Phase 0 (if claude is auth'd) → Phase 1 → Phase 3.
# Phases 2/4/5 auto-skip on CPU-only machines with a clear [skip] message.
# ~10–15 min total.
```

Useful for:
- Confirming the SAGE port works against the Anthropic API
- Smoke-testing changes to `pathome/symptoms.py`, `data/bugwood_loader.py`, the build/enhance scripts
- Running on CI

---

## Skip / resume / orchestrator knobs

```bash
# Skip individual phases by ID:
SMOKE_SKIP_0=1   bash smoke/run_smoke.sh    # skip Phase 0 (seed already on disk)
SMOKE_SKIP_2=1   bash smoke/run_smoke.sh    # skip trace generation
SMOKE_SKIP_4=1   bash smoke/run_smoke.sh    # skip OBSERVE training
SMOKE_SKIP_5=1   bash smoke/run_smoke.sh    # skip eval+compare

# Restart from a specific phase:
SMOKE_FROM=4     bash smoke/run_smoke.sh    # restart at training (assumes traces exist)

# Switch trace orchestrator:
SMOKE_ORCH=autogen_swarm bash smoke/run_smoke.sh   # use vLLM (default: hf_direct)

# Adjust the Setup-stage class count threshold:
SMOKE_THRESHOLD=10 bash smoke/run_smoke.sh   # wider smoke (~65 classes)
SMOKE_THRESHOLD=20 bash smoke/run_smoke.sh   # narrower (~15 classes)

# Phase 0 knobs (run_phase0_full.sh):
FULL_QUICK=1        bash smoke/run_phase0_full.sh   # cap sources/states (~15–25 min)
FULL_KEEP_CACHE=1   bash smoke/run_phase0_full.sh   # reuse cached final_registry.json
FULL_SKIP_SETUP=1   bash smoke/run_phase0_full.sh   # CSV already filtered
FULL_SKIP_CACHE=1   bash smoke/run_phase0_full.sh   # image cache already topped up
FULL_SKIP_KB=1      bash smoke/run_phase0_full.sh   # skip the python -m pathome_kb call
```

---

## Auth requirements (LOCAL only)

Phase 0 of the smoke (the SAGE-ported `pathome_kb` pipeline) needs:

1. The `claude` CLI on PATH and authenticated (`claude auth login`).
2. `ANTHROPIC_API_KEY` in env or `.env` at repo root (used by the Anthropic SDK in extraction + reconciliation).

If either is missing, `smoke/run_phase0_full.sh` errors out with the install commands. The Nova-side `submit_smoke.sh` does **not** check for the CLI — it relies on the seed file already being on disk from a `git pull`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `smoke/artifacts/pathome_seed/symptoms_seed.json not found` (Nova) | You forgot to `git push` the seed locally, or Nova hasn't `git pull`-ed. Run `git status` on Nova to confirm the file is present. |
| `claude CLI not on PATH` (local) | `curl -fsSL https://claude.ai/install.sh \| bash` then `claude auth login` |
| `ANTHROPIC_API_KEY not set` (local) | `echo "ANTHROPIC_API_KEY=sk-ant-..." > .env` at repo root |
| Phase 2 hangs or OOMs on Nova | Try `SMOKE_ORCH=hf_direct` to bypass vLLM. The HFClient is patched against the cross-image OOM. |
| Phase 4 fails to load checkpoint | The smoke is configured for 1 GRPO epoch; checkpoint name is `observe_grpo_epoch_01.pt`. If you bumped epochs, adjust `SEED_CKPT` / `ENH_CKPT` in `run_smoke.sh`. |
| `comparison.md` only has the trace columns, no eval rows | Phase 5 was skipped (no GPU? no checkpoints?). Check the Phase 4 output and `logs/pathome_smoke-*.out`. |
| Phase 2 finishes but `plantswarm_traces.jsonl` is empty | Likely an orchestrator crash — check the `[Phase 2]` lines in `logs/pathome_smoke-*.out` for vLLM / Qwen load errors. Re-run with `SMOKE_ORCH=hf_direct`. |
