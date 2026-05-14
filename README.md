# PlantSwarm + PathomeDB — KB build + KB-grounded CLIP fine-tune

Four-step pipeline. Each step runs in a fixed place (LOCAL or NOVA) and
hands off to the next step via `git push` / `git pull`:

```
                 ┌──────────────── STEP 1 ─ LOCAL  ─────────────────┐
                 │ scripts/sh_01_phase0_local.sh                    │
                 │   Claude Phase 0 canonical KB build              │
                 │   (NON-visual: pathogen, type, parts, treatments)│
                 │   → git push                                     │
                 └──────────────────┬───────────────────────────────┘
                                    │
                 ┌──────────────── STEP 2 ─ NOVA  ──────────────────┐
                 │ scripts/sh_02_swarm_nova.sh                      │
                 │   git pull                                       │
                 │   24-agent 2-round Qwen2.5-VL real swarm         │
                 │   (visual symptoms ONLY; verifier OFF here)      │
                 │   → git push  (deltas tagged "unverified")       │
                 └──────────────────┬───────────────────────────────┘
                                    │
                 ┌──────────────── STEP 3 ─ LOCAL  ─────────────────┐
                 │ scripts/sh_03_validate_local.sh                  │
                 │   git pull                                       │
                 │   Claude + WebSearch verifier over each delta    │
                 │   (extension / APS / CABI / peer-reviewed)       │
                 │   → git push  (deltas tagged verified /          │
                 │                provisional / contradictory etc.) │
                 └──────────────────┬───────────────────────────────┘
                                    │
                 ┌──────────────── STEP 4 ─ LOCAL ──────────────────┐
                 │ scripts/sh_04_tabpfn_local.sh                    │
                 │   git pull verified KB                           │
                 │   build captions (per strategy)                  │
                 │   FROZEN encoder forward (BioCLIP / CLIP /       │
                 │     SigLIP) over Bugwood + PV + PD + PW          │
                 │   feature vec = [image_emb | caption_emb |       │
                 │                  crop_onehot]                    │
                 │   TabPFN classifier over 11-variant feature      │
                 │     ablation matrix (zero trained params on      │
                 │     visual side; TabPFN is a meta-learned        │
                 │     tabular foundation model)                    │
                 │   eval on PV + PD + PW                           │
                 │   aggregate paper-style tables                   │
                 │   → git push results                             │
                 └──────────────────────────────────────────────────┘
```

The split is deliberate. **Nova has the GPU** but no `claude` CLI;
**LOCAL has Claude** but no A100. Each step runs on the host that has
the right tool.

Two command sets are documented below. They differ only by which crops
are processed:

| Set | Crops | Wall-clock | API spend | Use case |
|---|---|---|---|---|
| **A. 2-crop (smoke)** | Soybean + Tomato | ~4-8 h end-to-end | ~$5-15 | first-time run, validates the pipeline, fits in a day |
| **B. all-crop (production)** | All 484 (crop, disease) pairs in `BugWood_Diseases_usable.csv` | ~4-7 days end-to-end | ~$80-300 | the real paper run |

---

## Set A — 2-crop smoke (Soybean + Tomato)

Start here. End-to-end in under a day; ~$5-15 in Claude API spend.

```bash
# ============================================================
# STEP 1 — LOCAL (Phase 0 canonical KB via Claude)
# ============================================================
cd ~/Desktop/PlantSwarm
CROPS=smoke bash scripts/sh_01_phase0_local.sh
# ≈ 30-45 min. Writes artifacts/pathome_kb/{Soybean,Tomato}/final_registry.json
# then commits + pushes to origin/main.

# ============================================================
# STEP 2 — NOVA (24-agent 2-round Qwen swarm; verifier OFF)
# ============================================================
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/PlantSwarm
CROPS=smoke bash scripts/sh_02_swarm_nova.sh
# ≈ 3-6 h. sbatch one Phase 0R job (vLLM + 24-agent 2-round swarm),
# blocks until done, then pushes the unverified-deltas KB back to GitHub.
# Tip: set PATHOME_TRACE_DIR=artifacts/swarm_smoke to capture per-pass
# JSONL traces (round1_outputs, round2_outputs, cross_refs).

# ============================================================
# STEP 3 — LOCAL (Claude+WebSearch validation)
# ============================================================
# (back on your laptop)
cd ~/Desktop/PlantSwarm
git pull origin main
CROPS=smoke bash scripts/sh_03_validate_local.sh
# ≈ 30-60 min on smoke. Drives pathome_kb.verifier.verify_candidates
# tuple-by-tuple, fills in verification_status + web_support per delta,
# pushes verified KB back to GitHub.

# ============================================================
# STEP 4 — LOCAL (frozen encoder + TabPFN classifier)
# ============================================================
# (back on your laptop / any small-GPU host)
cd ~/Desktop/PlantSwarm
git pull origin main
CROPS=smoke bash scripts/sh_04_tabpfn_local.sh
# ≈ 30-60 min (frozen encoder forward on Tomato images + PV/PD/PW
# + TabPFN inference over the 11-variant feature ablation matrix).
# No CLIP training; TabPFN is meta-learned. Runs on a small GPU
# for the encoder forward + CPU for TabPFN. Pushes paper-style
# tables to GitHub.
```

Final outputs after Set A:

```
artifacts/pathome_kb/Soybean/final_registry.json    canonical + verified deltas
artifacts/pathome_kb/Tomato/final_registry.json     canonical + verified deltas
data/bugwood_features/<encoder>_<strategy>.npz     frozen-encoder Bugwood features
data/eval_features/<encoder>_<strategy>_<set>.npz  frozen-encoder PV/PD/PW features
results/pathomeood_eval/<variant>/{plantvillage,plantdoc,plantwild}.json   TabPFN results
results/tables/{table_01,...,figure_03}.md          paper-style markdown
results/pathomeood_report.md                        master report
```

---

## Set B — all-crop production (484 classes)

The real run. ~4-7 days end-to-end; ~$80-300 in Claude API spend.
Recommended only after Set A has succeeded end-to-end.

```bash
# ============================================================
# STEP 1 — LOCAL (Phase 0 canonical KB for ALL 197 crops)
# ============================================================
cd ~/Desktop/PlantSwarm
CROPS=all bash scripts/sh_01_phase0_local.sh
# ≈ 16-24 h. ~$60-180 in Anthropic API spend. Writes
# artifacts/pathome_kb/<Crop>/final_registry.json for every crop in
# BugWood_Diseases_usable.csv (197 of them).

# ============================================================
# STEP 2 — NOVA (24-agent swarm over ~2,000-3,000 image tuples)
# ============================================================
ssh tirtho@hpc-login.iastate.edu
cd /work/mech-ai-scratch/tirtho/PlantSwarm
CROPS=all VLLM_N_RUNS=10 VLLM_AGREEMENT_MIN=3 \
  bash scripts/sh_02_swarm_nova.sh
# ≈ 24-48 h. ~2,000-3,000 (crop, disease, state) tuples × 25 (or 49 in
# 2-round mode) vLLM calls each. Set VLLM_SWARM_ROUNDS=1 to fall back
# to single-round mode if you want ~half the wall-clock.

# ============================================================
# STEP 3 — LOCAL (Claude+WebSearch validation over every unverified delta)
# ============================================================
cd ~/Desktop/PlantSwarm
git pull origin main
CROPS=all bash scripts/sh_03_validate_local.sh
# ≈ 1-3 days. ~$20-100 in Claude spend. Use MAX_TUPLES=N to cap if
# you want to bound spend (the leftover deltas stay tagged
# "unverified" and PathomeOOD will still use them via the fallback
# caption path).

# ============================================================
# STEP 4 — LOCAL (frozen encoder + TabPFN on full Bugwood)
# ============================================================
cd ~/Desktop/PlantSwarm
git pull origin main
CROPS=all bash scripts/sh_04_tabpfn_local.sh
# ≈ 2-4 h on a single small GPU. Encoder forward for ~12K Bugwood
# images + PV/PD/PW, for each of 3 encoders × 7 caption strategies,
# then TabPFN inference for all 11 variants on CPU. TabPFN scales
# O(N²) in train rows; we cap at 10K via stratified subsample.
```

Final outputs after Set B:

```
artifacts/pathome_kb/*/final_registry.json          197 crop registries
data/bugwood_features/*.npz                        frozen-encoder features
data/eval_features/*.npz                           frozen-encoder eval features
results/pathomeood_eval/<variant>/*.json           TabPFN results (11 variants × 3 sets + 4 baselines)
results/pathomeood_report.md                        paper-style master report
```

---

## What each step does, in one sentence

| # | Where | Script | What |
|---|---|---|---|
| 1 | LOCAL | `sh_01_phase0_local.sh` | Claude builds the canonical (text-grounded, NON-visual) KB per crop |
| 2 | NOVA | `sh_02_swarm_nova.sh` | 24-agent 2-round Qwen2.5-VL real swarm extracts image-grounded visual deltas (verifier OFF) |
| 3 | LOCAL | `sh_03_validate_local.sh` | Claude+WebSearch verifies each delta against extension / APS / CABI |
| 4 | LOCAL | `sh_04_tabpfn_local.sh` | Frozen encoder (BioCLIP / CLIP / SigLIP) emits image_emb + KB-caption_emb + crop one-hot; TabPFN classifies; eval on PV / PD / PW |

---

## Skip-knobs (re-run only some steps)

Each shell script reads env vars for partial runs:

```bash
# Step 1
PATHOME_USABLE_CSV=other.csv     # override input CSV
PATHOME_SKIP_PUSH=1              # commit but don't push

# Step 2
PATHOME_TRACE_DIR=traces/        # capture per-pass JSONL traces
VLLM_N_RUNS=5                    # cheaper smoke (default 10)
VLLM_SWARM_ROUNDS=1              # disable round 2 (cheaper, less stigmergy)
VLLM_AGREEMENT_MIN=2             # K-of-N floor (default 3)

# Step 3
MAX_TUPLES=50                    # cap on (crop, disease, state) tuples
DRY_RUN=1                        # print plan without calling Claude

# Step 4 (TabPFN path; default)
PATHOME_SKIP_CAPTIONS=1          # captions already built
PATHOME_SKIP_FEATURES=1          # encoder forwards already cached
PATHOME_SKIP_TABPFN=1            # TabPFN matrix already run (re-aggregate only)
PATHOME_SKIP_AGG=1               # skip aggregation
ENCODERS="bioclip,clip_vitb16"   # which encoders to extract features for
STRATEGIES="canonical_deltas_3"  # which caption strategies to extract
```

---

## Architecture overview

This section explains *what* is being built. For *how to run it*, use
the command sets above.

### Phase 0 — Canonical KB (Claude, LOCAL)

For each (crop, disease) pair in `BugWood_Diseases_usable.csv`:

1. **Discovery** — Claude searches extension / APS / CABI / peer-
   reviewed sources for the most authoritative descriptions.
2. **Extraction** — Claude pulls verbatim quotes for each canonical
   field (`pathogen_scientific_name`, `type_of_disease`,
   `affected_parts`, `visual_symptoms.summary`,
   `visual_symptoms.diagnostic_features`,
   `visual_symptoms.look_alikes`, `treatments`).
3. **Reconciliation** — `claude -p` (headless CLI, JSON-schema mode)
   merges the per-source extractions into one canonical record with
   URL + verbatim quote per field. No Anthropic API key path —
   everything runs on the user's Claude Code subscription.

Output: `artifacts/pathome_kb/<Crop>/final_registry.json` with the
top-level `diseases[]` array. `regional_observations` is empty at this
stage; Phase 0R fills it in.

### Phase 0R — 24-agent 2-round real swarm (Qwen2.5-VL, NOVA)

**The "real swarm" part.** Naive parallel-ensemble setups have
specialists run in isolation and a consolidator collects outputs. This
is a real swarm because it has **stigmergy** (a shared blackboard) and
**cross-talk** (specialists react to each other's findings):

```
Round 1 — independent observation
  └─ 24 specialists run in parallel on (image, canonical KB, existing KB)
  └─ each asks ONE laser-focused visual question
  └─ no peer visibility yet

Blackboard built from all round-1 outputs (dict[AGENT_NAME → output])

Round 2 — stigmergy refinement
  └─ same 24 specialists run AGAIN in parallel
  └─ each now sees the FULL blackboard rendered in its prompt
  └─ may emit cross_refs against peers:
       SUPPORT   — raises peer's effective confidence
       CHALLENGE — consolidator must adjudicate
       WITHDRAW  — self-cancel a round-1 delta

VisualDiagnosisAgent (consolidator)
  └─ sees BOTH rounds + cross-ref digest grouped by action
  └─ walks 5-step CoT (decision-graph from DR.Arti.docx):
       (1) triage which organs are visible
       (2) decisive forks
       (3) adjudicate cross_refs
       (4) dedup
       (5) emit final deltas + CoT trace
```

The 24 specialists are decomposed into 7 organ families:

| Family | Count | Specialists |
|---|---|---|
| LEAF | 8 | LeafLesionShape, LeafLesionColor, LeafLesionTexture, LeafChlorosis, LeafNecrosis, LeafCurl, LeafVeinPattern, LeafGeometry |
| STEM | 4 | StemLesion, **StemPith** (decisive SDS/BSR fork), StemSurface, StemDiscoloration |
| BELOW-GROUND | 2 | **Root** (cysts → SCN; blue masses → SDS), CrownCollar |
| REPRODUCTIVE | 2 | Flower, Fruit |
| PATHOGEN SIGNS | 1 | Sporulation (mycelium / spores / ooze) |
| WHOLE-PLANT PATTERNS | 3 | Wilting, **Defoliation** (bare-petiole SDS fork), SpatialPattern |
| DIAGNOSTIC CROSS-CUTTERS | 4 | ConcentricPattern, **ColorPalette** (color encoder), **LookAlikeCoT** (decision-graph), SeverityVisual |

Per-pass cost: 24 specialists × 2 rounds + 1 consolidator = **49 vLLM
calls**. N=10 stochastic passes per (crop, disease, state) tuple.

The swarm focuses **exclusively on visual symptoms**. Pathogen, type,
affected parts, treatments — those are all handled by Claude in Phase 0
and never re-emitted by the swarm.

### Phase 0R verification — Claude+WebSearch (LOCAL, step 3)

Nova writes deltas with `verification_status="unverified"`. Step 3
walks every unverified delta, sends it (with context) to
`pathome_kb.verifier.verify_candidates` which calls `claude -p` with
WebSearch. Each delta gets:

| `verification_status` | Meaning | Goes to KB? |
|---|---|---|
| `verified` | direct hit on multiple authoritative sources | ✓ |
| `weakly_supported` | one source agrees | ✓ |
| `provisional` | no direct support but biologically plausible | ✓ |
| `novel_plausible` | new observation, plausible mechanism | ✓ (flagged) |
| `contradictory` | sources contradict the claim | ✗ (dropped) |
| `duplicate_existing` | matches an existing delta | merged (support++) |

### Phase PathomeOOD — frozen encoder + TabPFN classifier (LOCAL, step 4)

For the small-data regime (~10–12K Bugwood images), the current step 4
**replaces full CLIP fine-tuning with a frozen-encoder + tabular
foundation classifier setup**. Zero trained parameters on the visual
side; the classifier is meta-learned [TabPFN](https://arxiv.org/abs/2207.01848).

**Feature vector per image**:
```
x = [ image_emb          (frozen visual encoder)        ≈ 512–1024 dim
    | caption_emb        (frozen text encoder on KB
                          -derived caption for this row) ≈ 512      dim
    | crop_onehot        (Bugwood crop vocabulary)       ≈ 50–200   dim ]
```
PCA-reduced to ~256 + crop one-hot before TabPFN, keeping it well
under the TabPFNv2 feature-count limit.

**11-variant feature ablation matrix** (`scripts/tabpfn_eval.py::VARIANTS`):
T01–T07 vary the caption strategy (label_only → canonical_deltas_7);
T08–T09 swap the encoder (BioCLIP → CLIP-openai → SigLIP); T10–T11
restrict the train set to KB-covered / non-covered classes. **Zero
training per variant** — TabPFN does in-context learning over the
support set in one forward pass.

Plus 4 off-shelf zero-shot baselines (CLIP / SigLIP / BioCLIP /
BioCLIP-2) computed by straight cosine-sim against class-name
templates.

Eval: top-1 / top-5 on PlantVillage, PlantDoc, PlantWild — same three
test sets as the original BioCAP-style table reproduction. Outputs
written to `results/pathomeood_eval/<variant>/{plantvillage,plantdoc,plantwild}.json`
and aggregated into `results/pathomeood_report.md` (same paper-style
table set as before).

**Legacy fine-tuning path** (`scripts/sh_04_finetune_nova.sh`,
`scripts/train_pathomeood.py`, `scripts/submit_pathomeood_*.sh`,
`train_and_eval/` subtree) is still in the repo but no longer on the
critical path. The TabPFN path is the default for small data.

Master report: `results/pathomeood_report.md`.

For the full architectural deep-dive see [PIPELINE.md](PIPELINE.md);
for the end-to-end animated walkthrough see [FLOW.md](FLOW.md).

---

## One-time prerequisites

### LOCAL

```bash
git clone https://github.com/tirtho149/PlantSwarm.git
cd PlantSwarm

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Claude CLI for Phase 0 + verifier
# (install from https://claude.com/code; run `claude` once interactively
# to authenticate)
#
# All Claude calls in this pipeline go through the headless `claude -p`
# CLI — there is no Anthropic API key path. Your Claude Code
# subscription is the only billing surface.
```

### NOVA (one-time GPU-host install)

```bash
ssh tirtho@hpc-login.iastate.edu
cd /work/<your-scratch>/
git clone https://github.com/tirtho149/PlantSwarm.git
cd PlantSwarm

# Standard deps:
pip install -r requirements.txt

# GPU-only deps (see requirements.txt's "GPU host only" section):
pip install vllm torch open_clip_torch webdataset huggingface_hub \
            transformers accelerate
```

---

## Repo layout

```
PlantSwarm/
├── README.md                              this file (run-it instructions)
├── PIPELINE.md                            architectural deep-dive
├── FLOW.md                                end-to-end flow + GIF
├── DR.Arti.docx                           reference doc with look-alike CoT
│                                          decision graphs (SDS↔BSR etc.)
│
├── BugWood_Diseases.csv                   raw IPMNet export
├── BugWood_Diseases_usable.csv            filtered (Setup output)
│
├── agents/                                24-specialist visual-symptom swarm
│   ├── base_agent.py                      Blackboard, CROSS_REF_ACTIONS,
│   │                                      DELTA_USER_PROMPT (R1 + R2)
│   ├── leaf_agents.py                     8 leaf specialists
│   ├── stem_agents.py                     4 stem specialists
│   ├── root_agents.py                     Root + CrownCollar
│   ├── reproductive_agents.py             Flower + Fruit
│   ├── sign_agents.py                     Sporulation (signs vs symptoms)
│   ├── pattern_agents.py                  Wilting + Defoliation + Spatial
│   ├── diagnostic_agents.py               Concentric + ColorPalette +
│   │                                      LookAlikeCoT + Severity
│   └── diagnosis_agent.py                 VisualDiagnosisAgent CoT consolidator
│
├── train_and_eval/                        (legacy) dual-projector CLIP code —
│                                          OFF the critical path; kept for
│                                          reference. The current step 4 uses
│                                          frozen encoder + TabPFN instead.
│   ├── open_clip/                         model + two visual projectors
│   ├── open_clip_train/                   torchrun entry (data + train adapted)
│   ├── evaluation/                        zero_shot_iid + retrieval + metrics
│   └── imageomics/                        naming_eval + disk + helpers
│
├── plantswarm/                            swarm orchestrator + captioner
│   ├── delta_pipeline.py                  2-round real swarm: run_for_state,
│   │                                      run_batch, _agreement_filter,
│   │                                      _merge_with_existing
│   ├── captioning.py                      build_disease_caption (7 strategies),
│   │                                      _top_regional_deltas (state-aware),
│   │                                      load_kb_profiles, caption_for_row
│   └── latex/                             paper sources
│
├── pathome_kb/                            Phase 0 + verifier
│   ├── pipeline.py                        per-crop orchestrator (CLI)
│   ├── internet_pipeline.py               Claude discovery + extraction +
│   │                                      reconciliation
│   ├── regional_observation.py            per-tuple Qwen-swarm caller
│   ├── verifier.py                        Claude web-search verifier
│   ├── symptoms_adapter.py                (legacy) merged-seed adapter
│   └── prompts/                           canonical-stage prompts
│
├── pathome/                               KB schema
│   └── symptoms.py                        SymptomLibrary, SymptomProfile,
│                                          CanonicalDisease, RegionalObservation,
│                                          RegionalDelta, Citation
│
├── utils/
│   ├── vllm_client.py                     OpenAI-compatible vLLM client
│   ├── geo.py                             state centroid + AEZ
│   └── env.py                             .env loader
│
├── data/bugwood_loader.py                 crop / disease normalization (Setup)
│
├── scripts/
│   ├── sh_01_phase0_local.sh              STEP 1 — LOCAL: Phase 0 + push
│   ├── sh_02_swarm_nova.sh                STEP 2 — NOVA: swarm + push
│   ├── sh_03_validate_local.sh            STEP 3 — LOCAL: validate + push
│   ├── sh_04_tabpfn_local.sh              STEP 4 — LOCAL: frozen encoder
│   │                                       + TabPFN classifier + push
│   ├── sh_04_finetune_nova.sh              (legacy) NOVA dual-projector
│   │                                       CLIP fine-tune; kept for reference
│   ├── validate_kb.py                     step-3 driver (Claude verifier)
│   │
│   ├── build_pathomeood_captions.py       KB → captions parquet
│   ├── build_features.py                  frozen encoder forward → image_emb
│   │                                       + caption_emb + crop one-hot npz
│   ├── tabpfn_eval.py                     TabPFN classifier over the 11-variant
│   │                                       feature ablation matrix + 4 baselines
│   ├── aggregate_pathomeood_tables.py     result JSONs → paper-style table .md
│   │                                       (works for both TabPFN + legacy paths)
│   │   ----- legacy (off critical path) ---------------------------------------
│   ├── build_pathomeood_shards.py         parquet → WebDataset shards
│   ├── pathomeood_variants.sh             T01..T11 training-matrix definition
│   ├── train_pathomeood.py                wrapper around open_clip_train.main
│   ├── submit_pathomeood_train.sh         SLURM: one variant
│   ├── submit_pathomeood_matrix.sh        SLURM: sbatch all 11 variants
│   ├── evaluate_pathomeood.py             zero-shot eval on PV/PD/PW
│   ├── evaluate_pathomeood_retrieval.py   Bugwood held-out R@k
│   ├── evaluate_pathomeood_fewshot.py     prototype-mean K-shot
│   ├── fetch_baselines.py                 cache 5 off-shelf CLIP baselines
│   ├── setup_plantdoc.py                  clone PlantDoc to data/eval/
│   │
│   ├── filter_bugwood_csv.py              raw CSV → filtered usable CSV
│   ├── ensure_state_image_cache.py        per-(crop, disease, state) image cache
│   ├── submit_pathome_setup_filter.sh     Nova SBATCH: filter CSV
│   ├── setup_image_cache.sh               LOCAL/Nova: image cache
│   ├── submit_phase0r_regional.sh         Nova SBATCH: vLLM + Phase 0R swarm
│   │
│   ├── viz_kb.sh / viz_traces.sh / viz_all.sh   KB + trace visualizations
│   ├── build_latex_pdf.sh                 paper compile helper
│   └── viz/                               Python visualizers
│
└── smoke/                                 (legacy 2-crop happy path)
```

---

## Tests

```bash
pytest tests/ -q
# 59 tests covering: agent parser, agreement filter, conservative
# merge, Blackboard + 2-round protocol, captioner (7 strategies +
# fallback + delta guard), shard packager.
```

All tests pass without GPU dependencies — Phase 0R / PathomeOOD code
uses lazy imports for torch / vLLM / open_clip.

---

## Skipping legs

If you've already done step N for the same crops, just re-run from
step N+1. Each script does `git pull --ff-only` at start, so as long
as you `git push` between hosts the next step will pick up the
correct state.

```bash
# Re-validate only (re-pulls Nova's deltas, re-runs verifier):
CROPS=smoke bash scripts/sh_03_validate_local.sh

# Re-train + eval without rebuilding shards:
ssh tirtho@hpc-login.iastate.edu
PATHOME_SKIP_CAPTIONS=1 bash scripts/sh_04_tabpfn_local.sh

# Just re-aggregate tables (eval results already on disk):
PATHOME_SKIP_CAPTIONS=1 PATHOME_SKIP_TRAIN=1 PATHOME_SKIP_EVAL=1 \
  bash scripts/sh_04_tabpfn_local.sh
```

---

## Consuming the KB downstream

PathomeOOD reads `final_registry.json` directly. Other consumers can
do the same:

```python
from plantswarm.captioning import load_kb_profiles, caption_for_row

profiles = load_kb_profiles("artifacts/pathome_kb", crop_filter=["Tomato"])
# dict[(crop, disease) -> disease_record from final_registry.json]

caption, used_kb = caption_for_row(
    crop="Tomato", disease="Early Blight", state="CA",
    profiles=profiles, strategy="canonical_deltas_3",
)
# multi-sentence text combining canonical summary, diagnostic features,
# look-alikes, and the top-3 regional deltas for the given state.
```

---

## Citation

See `CITATION.cff`.
