# Migration: PlantSwarm → Pathome (Train-on-the-Wild)

The paper at `plantswarm/latex/acl_latex.tex` (v: pathome_final) describes a
substantially different system than the current code implements. This file
tracks the gap and the order in which it is being closed.

## Paper system in one paragraph

Train PlantSwarm on **260 geo-tagged Bugwood field images** (10 per disease
class × 26 classes; 7 → trace generation × 30 runs = 5,460 traces; 3 →
PathomeDB Layer 5 references). Build **PathomeDB**, a 5-layer knowledge base
(mechanistic pathway → cross-crop manifestation → GPS-density-derived regional
epidemiology → diagnostic decision graph → geo-tagged reference library with
FAISS retrieval). Train **OBSERVE** (Qwen2.5-VL-**7B** + LoRA) in two phases —
Decision Transformer on traces, then GRPO refinement — with overconfidence
detection and an `epsilon + alpha = 1 − c` uncertainty budget regularizer.
Evaluate on the **complete** PlantVillage (54,306 images, 12 zero-shot classes)
and the **complete** PlantWild dataset.

## What's done in code today (`60daaa7` baseline)

| Component | Status |
|---|---|
| 5-agent swarm (Morphology/Symptom/Pathogen/Severity/Diagnosis) | ✓ in `agents/` |
| Confidence-gated routing with backtracking | ✓ `plantswarm/pipeline.py`, `autogen_pipeline.py` |
| TFDS PlantVillage loader | ✓ `data/tfds_plant_village.py` (used as training data, paper inverts this) |
| HF PlantWild loader | ✓ `data/plantwild_hf.py` |
| Bugwood folder copier (no GPS) | ✓ `DataLoader.py:5342 load_BugwoodMerged` (legacy, image-tree only) |
| OBSERVE model: routing + confidence + epistemic + aleatoric heads | ✓ `observe/model.py` (Qwen2.5-VL-**3B**) |
| OBSERVE multi-task trainer | ✓ `observe/trainer.py` (single-phase, no DT, no GRPO) |
| Incremental trace persistence + resume | ✓ `utils/routing_trace.py`, `scripts/run_plantswarm.py` |
| Calibration: ECE, temperature scaling, conformal | ✓ `calibration/` |

## Gap to paper, ordered by dependency

### Tier 1 — foundational (blocks everything else)  ✅ DONE (CSV-driven)
- ✅ `data/bugwood_loader.py` — CSV-driven adapter over
  `BugWood_Diseases.csv`. Yields `BugwoodRecord` with `image_b64`, `crop`,
  `disease`, `lat/lon` (state centroid), `aez`. Quality filters now state-
  centric (state present, normalised crop+disease present, optional min side).
  Reuses `BUGWOOD_EXACT_CROP_MAP` from `DataLoader.py` for crop normalisation
  and strips parenthetical scientific suffixes from `Subject Display Name`.
  Per-class cap + 7/3 trace/reference split unchanged. URL → local cache
  download with retries. k-medoids `select_diverse_subset` retained.
  *Class scope:* the paper's 26-class subset is **not** enforced — admits
  every `(crop, disease)` pair surviving normalisation (~982 classes, 8.2k
  records at `per_class=10, min_per_class=5`).
- ✅ `utils/geo.py` — added `state_to_latlon` + `US_STATE_CENTROID` table for
  the 50 states + DC + territories. EXIF GPS path retained for legacy folder-
  tree mode. AEZ lookup, climate vector, `encode_phi_geo` unchanged.

**Caveats inherited from CSV shape:**
- No per-image GPS — Layer 3 epidemiology is at state-centroid resolution,
  not the AEZ-month grid the paper §6.3 describes against ideal EXIF.
- No capture date — `month` is sentinel `0` for CSV records; Layer 3
  collapses the time axis and `phi_geo` zeros its month sin/cos channels.
  `pathome/layer3_geo.py` guard relaxed from `if not month` to
  `if month is None` so the sentinel survives ingestion.

### Tier 2 — knowledge base  ✅ DONE (symptom-centric refactor)
The original 5-layer split (mechanistic pathway / cross-crop manifestation /
regional epidemiology / decision graph / reference library) was collapsed
into two stores once it became clear the Bugwood CSV could not feed the
mechanistic / decision-graph layers:

- ✅ `pathome/symptoms.py` — `SymptomLibrary` of `SymptomProfile` per
  `(crop, disease)`. Each profile carries a `VisualSymptom` block (plant
  parts, color, shape/margin/texture, sporulation, distinctive signs,
  progression, confusion-pair set, free-form notes), per-state and per-AEZ
  observation counts (geo-aware, replaces L3), and a list of reference IDs
  pointing into the reference library. Auto-populates state/AEZ counts and
  reference IDs from `BugwoodRecord`s; visual fields can be hand-curated
  via a JSON sidecar (`pathome.symptoms_path` in the YAML). Exposes
  `geo_prior(disease, state)`, `prevalent_in_state(state, k)`,
  `reobservation_prompt(crop, disease)`.
- ✅ `pathome/layer5_references.py` — kept as-is. `ReferenceLibrary` over
  the held-out reference split (1,452 images) with CLIP embeddings + FAISS
  (or NumPy fallback) and `0.7·cos + 0.3·ClimSim` weighted retrieval.
- ✅ `pathome/database.py` — `PathomeDB` is now `symptoms` + `refs` only.
  `geo_prior(disease, lat, lon, month=None)` reverse-geocodes (lat, lon)
  to the nearest US state centroid and reads the prior straight off
  `SymptomLibrary.geo_prior`. Save/load is two files (`symptoms.json` +
  `refs/`) instead of five.
- 🗑 `pathome/layer1_pathway.py`, `layer2_manifestation.py`, `layer3_geo.py`,
  `layer4_decision_graph.py` deleted in the post-migration cleanup. Their
  semantics now live inside `SymptomProfile` (`VisualSymptom`,
  `state_counts`, `aez_counts`).

**Caveats inherited from CSV shape:**
- No per-image GPS — the geo prior is at US-state-centroid resolution.
- No capture date — month axis dropped; the `month` argument on
  `geo_prior(...)` is accepted for compat but ignored.
- VisualSymptom blocks ship empty by default. Curators populate them via
  a sidecar JSON keyed by `profile_id`.

### Tier 3 — OBSERVE refit
- **`observe/model.py`** — switch backbone to Qwen2.5-VL-**7B**; add
  overconfidence (OC) head (binary sigmoid); accept `phi_geo` and
  `Ref_{1:3}` as additional inputs alongside image + context.
- **`observe/loss.py`** — multi-task losses: routing CE, calibration MSE,
  consistency `|epsilon + alpha − (1−c)|`, belief LM, OC BCE.
- **`observe/decision_transformer.py`** — Phase A trainer: traces →
  return-conditioned sequences; predict next action conditioned on target
  return.
- **`observe/grpo.py`** — Phase B trainer: G=8 group rollouts, normalized
  advantage, clipped surrogate, KL to Phase-A reference.
- **`observe/active_learning.py`** — epsilon-driven query ranking over
  unlabeled cross-crop pool; expected ~950 labels to converge.

### Tier 4 — eval + scripts ✅ DONE
- ✅ `scripts/build_pathome.py` — Bugwood ingest → PathomeDB build
- ✅ `scripts/run_pathome_traces.py` — 30 runs/image trace generation with resume
- ✅ `scripts/train_observe_pathome.py` — Phase A then Phase B
- ✅ `scripts/evaluate_pathome.py` — held-out PV and PW eval w/ unseen-class slice
- ✅ `scripts/submit_pathome_phase{0..5}_*.sh` + `submit_pathome_all.sh`
  — six-phase Nova SLURM pipeline (seed → build → traces → enhance →
  train×2 → eval+compare), with vLLM booted in-job for the GPU phases.
- ✅ `plantswarm/observe_rollout.py` — GRPO `rollout_fn`
- ✅ `agents/base_agent.py` — `pathome_db` parameter + Layer 4 / Layer 3 helpers
- ✅ `plantswarm/{pipeline,autogen_pipeline}.py` — thread PathomeDB to agents
- ✅ `scripts/run_plantswarm.py` — auto-loads PathomeDB from `cfg.pathome.load_dir`

### Tier 5 — paper auto-sync ✅ DONE
- ✅ `scripts/sync_pathome_metrics.py` — emits `auto_pathome_metrics.tex` with
  macros (`\PathomePvECE`, `\PathomePwECE`, `\PathomePvTthreeF`,
  `\PathomePwTthreeF`, `\PathomeUnseenTthreeF`, `\PathomePvTPCP`,
  `\PathomePwTPCP`) for the hardcoded paper tables.

## What's still genuinely open

These items require either real Bugwood data or research decisions and
cannot be stubbed productively:

1. **OBSERVE.forward batching.** The current per-sample iteration in
   `decision_transformer.py` works for correctness but is slow. A batched
   forward needs the Qwen2.5-VL processor's pad_to_max_length plus careful
   image-token handling.
2. **GRPO rollout integration.** `plantswarm/observe_rollout.py` records
   per-step log-probs by re-querying OBSERVE; ideal would be the swarm
   itself emitting log-probs along the path. That requires AutoGen
   handoff hooks not yet exposed.
3. **VisualSymptom curation breadth.** Phase 0 (`scripts/seed_pathome_with_claude.py`)
   populates the visual block per (crop, disease) via Claude headless,
   but rare or ambiguous classes can come back with empty fields. Audit
   `artifacts/pathome_seed/symptoms_seed.json` and edit by hand for the
   long tail.
4. **EPPO validation.** Cross-validating `state_counts` against EPPO
   historical prevalence requires a separate EPPO ingestion task and a
   finer geo signal than US state centroids. Out of scope for now.
5. **Active learning oracle.** `observe/active_learning.py` selects
   queries; the human-in-the-loop labelling backend is intentionally
   left to deployment.

## Naming conventions in the new modules

- `BugwoodRecord` (analogous to `PlantRecord`) — `image_b64`, `crop`,
  `disease`, `lat`, `lon`, `aez`, `month`, `image_id`.
- `PathomeDB` is loaded once and passed to both `PlantSwarmPipeline` and
  `OBSERVE` at construction.
- All new modules cite the paper section in their module docstring (`§6.1`,
  `Eq. density`, `Algorithm 1`, etc.) so the paper is the source of truth.

## What this migration does NOT do

- It does not pre-populate Layer 1 / Layer 2 with every Bugwood class — the
  CSV admits ~982 `(crop, disease)` pairs whereas Layer 1 ships with 10
  pathogen genera and Layer 2 with 19 host-pathogen entries. Classes outside
  this coverage will fall back to `global_prior` at routing time.
- It does not run a working DT or GRPO training pass end-to-end. The
  trainers have full forward / backward logic but the rollout-collection
  loop in GRPO is stubbed pending integration with the agent runtime.
- It does not perform AEZ-month epidemiology — the CSV has neither per-image
  GPS nor capture dates, so Layer 3 collapses to a state-resolution
  histogram with `month=0`. The full AEZ-month grid (paper §6.3) requires
  EXIF-rich Bugwood exports that this CSV does not include.

These are flagged with `# TODO(pathome):` markers throughout the new code.
