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

### Tier 1 — foundational (blocks everything else)
- **`data/bugwood_loader.py`** — proper Bugwood adapter that yields `BugwoodRecord`
  with `image_b64`, `crop`, `disease`, `lat/lon`, `aez`, `month`. Hard quality
  filters (GPS ≤10km precision, ≥512², single-subject). k-medoids selection
  on CLIP embeddings to pick 10 per class.
- **`utils/geo.py`** — EXIF GPS extraction, FAO AEZ lookup from (lat, lon),
  climate-zone vector for retrieval similarity, month-of-year encoding.

### Tier 2 — knowledge base
- **`pathome/layer1_pathway.py`** — `MechanisticPathway` per pathogen genus
  (entry mechanism, enzymatic cascade steps, epistemic implication per step).
  Static knowledge; populate from paper §6.1 plus extension service literature.
- **`pathome/layer2_manifestation.py`** — crop-pathogen lesion-size /
  sporulation-timing / color-shift map. Static.
- **`pathome/layer3_geo.py`** — `RegionalEpidemiology`: P̂(d | r, σ) from
  Bugwood GPS density (Eq. density). Sparse-cell handling. EPPO Pearson r
  validation interface.
- **`pathome/layer4_decision_graph.py`** — NetworkX DiGraph: host → plant part
  → symptom category → progression → pathogen signs → environment.
- **`pathome/layer5_references.py`** — FAISS index over 78 held-out Bugwood
  references (CLIP 512-dim). Geo-weighted retrieval `0.7·cos + 0.3·ClimSim`.
- **`pathome/database.py`** — `PathomeDB` orchestrating all 5 layers; loads
  from a serialized `.pathome/` dir; exposes `query(image, gps, claim)`.

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
- ✅ `scripts/submit_pathome_pipeline.sh` — end-to-end SLURM (vLLM in-job)
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
3. **EPPO validation table.** `RegionalEpidemiology.validate_against_eppo`
   wants a `Dict[(disease, AEZ, month), float]` of historical prevalence;
   pulling that from the EPPO API is a separate ingestion task.
4. **Layer 1 / Layer 2 full coverage.** Now ships 10 pathogen genera +
   19 host-pathogen entries — sufficient for the paper's 26-class subset
   if your Bugwood folder uses these crops/diseases. Add new entries
   directly in `pathome/layer{1,2}_*.py` or via `MechanisticPathway.save`.
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

- It does not actually fetch Bugwood images. The user retains 260 hand-picked
  field photographs locally; the loader expects them at the existing
  `Curated_Bugwood_Dataset/Images/<Crop>/<Disease>/*.jpg` tree.
- It does not pre-populate Layer 1 / Layer 2 with all 26 disease classes.
  Each layer ships with two worked examples (Colletotrichum, Phytophthora)
  and a clear schema for adding the rest.
- It does not run a working DT or GRPO training pass end-to-end. The
  trainers have full forward / backward logic but the rollout-collection
  loop in GRPO is stubbed pending integration with the agent runtime.

These are flagged with `# TODO(pathome):` markers throughout the new code.
