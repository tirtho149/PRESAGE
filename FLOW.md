# PlantSwarm — End-to-End Flow

Submission-ready overview of the current pipeline. All flowcharts are
Mermaid (renderable on GitHub, mermaid.live, and any standard Markdown
viewer); data shapes are ASCII for stable layout.

Sections
1. [Top-level pipeline](#1-top-level-pipeline)
2. [Phase 0 — canonical KB (Claude)](#2-phase-0--canonical-kb-claude)
3. [Phase 0R — regional deltas (Qwen visual-symptom swarm)](#3-phase-0r--regional-deltas-qwen-swarm)
   - 3a. [Per-tuple flow (iterative KB loop)](#3a-per-tuple-flow-iterative-kb-loop)
   - 3b. [Inside one pass (24 visual specialists + CoT consolidator)](#3b-inside-one-pass-24-visual-specialists--cot-consolidator)
   - 3c. [Animated walkthrough](#3c-animated-walkthrough)
   - 3d. [Cross-pass K-of-N agreement filter](#3d-cross-run-k-of-n-agreement-filter)
   - 3e. [Conservative merge with existing KB](#3e-conservative-merge-with-existing-kb)
4. [Phase PathomeOOD — KB-grounded CLIP training](#4-phase-pathomeood--kb-grounded-clip-training)
5. [Data shape evolution](#5-data-shape-evolution)
6. [File map](#6-file-map)
7. [Env var reference](#7-env-var-reference)
8. [Run-report line](#8-run-report-line)

---

## 1. Top-level pipeline

LOCAL machine → GitHub → GPU host. Three terminal deliverables.

```mermaid
flowchart TD
    SETUP[Setup<br/>filter_bugwood_csv.py<br/>raw CSV to filtered CSV]
    CACHE[Image cache<br/>ensure_state_image_cache.py<br/>per crop disease state photo]
    P0[Phase 0 - canonical KB<br/>pathome_kb via Claude<br/>discovery / extraction / reconciliation<br/>everything NON-visual: pathogen + treatments + parts]
    PUSH([git push canonical artifacts])
    PULL([git pull on GPU host])
    VLLM[vLLM serves Qwen2.5-VL-7B-Instruct<br/>booted in-job]
    P0R[Phase 0R - visual-symptom swarm<br/>24 specialists in parallel per pass<br/>+ VisualDiagnosisAgent CoT consolidator]
    REG([artifacts/pathome_kb/Crop/final_registry.json<br/>canonical + regional_observations - KB deliverable])
    BUGWOOD[(Bugwood field photos<br/>filtered CSV + .bugwood_cache<br/>11,513 imgs / 484 classes)]
    CAP[build_pathomeood_captions.py<br/>KB-grounded captions per image<br/>state-aware delta selection]
    SH[build_pathomeood_shards.py<br/>WebDataset tar shards<br/>image + taxon.txt + caption.txt]
    TRAIN[Train PathomeOOD<br/>ViT-B/16 dual-projector from BioCLIP init<br/>projectors-only - 800K trainable params<br/>11-variant matrix T01..T11]
    CKPT([train_and_eval/checkpoints/Tnn/...<br/>model deliverable])
    EVAL[Evaluate PathomeOOD<br/>scripts/evaluate_pathomeood.py<br/>zero-shot + retrieval + few-shot<br/>vs 5 off-shelf CLIP baselines]
    METRICS([results/pathomeood_report.md<br/>paper-style table reproduction])

    SETUP --> CACHE --> P0
    P0 --> PUSH --> PULL --> P0R
    VLLM -.serves.-> P0R --> REG
    REG --> CAP
    BUGWOOD --> CAP --> SH --> TRAIN
    TRAIN --> CKPT --> EVAL --> METRICS

    classDef local fill:#dff,stroke:#066,stroke-width:1px
    classDef gpu fill:#fde,stroke:#a06,stroke-width:1px
    classDef student fill:#eef,stroke:#33a,stroke-width:1px
    classDef terminal fill:#efe,stroke:#060,stroke-width:2px
    class SETUP,CACHE,P0 local
    class VLLM,P0R gpu
    class BUGWOOD,CAP,SH,TRAIN,EVAL student
    class REG,CKPT,METRICS terminal
```

| Stage | Host | Compute | Walltime |
|---|---|---|---|
| Setup | LOCAL or Nova | CPU, &lt; 1 min | trivial |
| Image cache | LOCAL or Nova | network only | smoke ~2 min |
| Phase 0 (Claude — NON-visual KB only) | LOCAL only (OAuth) | CPU + Anthropic API | smoke ~30 min / prod 16-24 h |
| Phase 0R (24-agent Qwen visual swarm) | GPU host with vLLM | 1x A100-80GB | smoke ~20-40 min / prod 10-20 h |
| PathomeOOD captions + shards | GPU host (CPU work) | CPU, minutes | a few minutes per strategy |
| PathomeOOD training | GPU host with CUDA | 1x A100 | ~30-60 min per variant; ~5 GPU-h for 11-variant matrix |
| PathomeOOD eval | GPU host with CUDA | 1x A100 | ~2 GPU-h for all variants + 5 baselines |

---

## 2. Phase 0 — canonical KB (Claude)

Run via `python -m pathome_kb`. Three Claude-driven stages per crop, all
text-grounded (URL + verbatim quote per field). No images touched here.

```mermaid
flowchart LR
    CSV[(BugWood_Diseases_usable.csv)]
    D[Stage 1 Discovery<br/>internet_pipeline.py<br/>claude -p with WebSearch<br/>per-disease URL list]
    E[Stage 2 Extraction<br/>internet_pipeline.py<br/>claude -p per URL<br/>verbatim quotes plus treatments]
    R[Stage 3 Reconciliation<br/>internet_pipeline.py<br/>Anthropic SDK or claude -p<br/>per-field merge with citations]
    REG[(final_registry.json<br/>canonical only<br/>per crop)]

    CSV --> D --> E --> R --> REG

    classDef claude fill:#fef,stroke:#606,stroke-width:1px
    classDef file fill:#ffd,stroke:#660,stroke-width:1px
    class D,E,R claude
    class CSV,REG file
```

Output shape (one disease entry):

```jsonc
{
  "disease_name": "Charcoal Rot",
  "pathogen_scientific_name": {
    "value": "Macrophomina phaseolina",
    "url":   "https://extension.umn.edu/.../charcoal-rot-soybean",
    "quote": "Charcoal rot is caused by the soilborne fungus..."
  },
  "type_of_disease":  { "value": "Fungal",  "url": "...", "quote": "..." },
  "affected_parts":   { "value": ["Foliar","Stem","Root","Pod"], "url": "...", "quote": "..." },
  "visual_symptoms": {
    "summary":             { "value": "...", "url": "...", "quote": "..." },
    "diagnostic_features": { "value": "...", "url": "...", "quote": "..." },
    "look_alikes":         { "value": [], "url": "", "quote": "" }
  },
  "treatments":         { "value": [], "url": "...", "quote": "..." },
  "regional_observations": {}
}
```

---

## 3. Phase 0R — regional deltas (Qwen swarm)

Run via `python -m pathome_kb --regional-only`. The orchestrator is
`plantswarm.delta_pipeline.run_for_state`, called once per
(crop, disease, state, cached image) tuple.

### 3a. Per-tuple flow (iterative KB loop with web-grounded verifier)

```mermaid
flowchart TD
    INPUT[crop, disease, state, cached image]
    LOAD[Load existing KB<br/>existing_deltas_for_state<br/>empty on cold start]
    FLAT[flatten_canonical<br/>plain values for prompt]
    URL[load_image_data_url<br/>MIME detected from extension]
    T1[Trace 1<br/>seed=42]
    T2[Trace 2<br/>seed=142]
    TN[Trace N<br/>seed=42+N*100]
    AGR[K-of-N agreement<br/>noise filter / proposal-confidence prior<br/>NOT a truth criterion]
    VERIFY[Claude verifier + WebSearch<br/>retrieval-grounded validation<br/>per-candidate verification_status<br/>+ web_support citations]
    MERGE[Conservative merge with existing<br/>verified + provisional accepted<br/>contradictory dropped<br/>overlap bumps support + upgrades status]
    OUT[Merged record<br/>state, deltas, swarm_meta + verifier_meta]

    INPUT --> LOAD
    INPUT --> FLAT
    INPUT --> URL
    LOAD -.context.-> T1
    LOAD -.context.-> T2
    LOAD -.context.-> TN
    FLAT -.context.-> T1
    FLAT -.context.-> T2
    FLAT -.context.-> TN
    URL -.image.-> T1
    URL -.image.-> T2
    URL -.image.-> TN
    T1 --> AGR
    T2 --> AGR
    TN --> AGR
    AGR --> VERIFY
    LOAD -.context.-> VERIFY
    FLAT -.context.-> VERIFY
    LOAD --> MERGE
    VERIFY --> MERGE
    MERGE --> OUT

    classDef input fill:#ffd,stroke:#660
    classDef ctx fill:#dff,stroke:#066
    classDef trace fill:#fde,stroke:#a06
    classDef agg fill:#eef,stroke:#33a
    classDef verify fill:#fef,stroke:#606,stroke-width:2px
    classDef out fill:#efe,stroke:#060,stroke-width:2px
    class INPUT input
    class LOAD,FLAT,URL ctx
    class T1,T2,TN trace
    class AGR,MERGE agg
    class VERIFY verify
    class OUT out
```

**Epistemic note.** Multi-run agreement from a single base model is
correlated, not orthogonal evidence — K-of-N agreement filters one-off
hallucinations but does not establish truth. The verifier stage adds
external-evidence support: Claude searches extension factsheets, APS /
CABI references, and peer-reviewed sources, then judges each candidate
against retrieved evidence. The KB therefore evolves like a scientific
observation system: the Qwen swarm is a high-recall **hypothesis
generator**, Claude is a retrieval-grounded **evidence reconciler**.

After every tuple finishes, `_embed_into_registry` merges its per-state
record back into the disease's `regional_observations` dict — **states
not processed this run are preserved verbatim**.

### 3b. Inside one pass (24 visual specialists + CoT consolidator)

Each of the N stochastic passes runs **24 single-feature visual
specialists** in parallel on the same (image, canonical-KB, existing-KB)
input, then `VisualDiagnosisAgent` consolidates the union by walking
the look-alike decision-graph CoT documented in
`Look alike Diseases, weeds and Insect COT with decision graph.docx`.
The swarm focuses **exclusively on visual symptoms** — pathogen, type,
treatments and other non-visual KB are owned by Claude in Phase 0 and
never re-emitted by the swarm.

Specialists are grouped by organ family. Each owns ONE delta field
and asks ONE laser-focused question.

```mermaid
flowchart TD
    INPUT([image + canonical visual_symptoms + existing KB])

    subgraph LEAF["LEAF (8 specialists)"]
      direction LR
      L1[LeafLesionShapeAgent<br/>leaf_lesion_shape]
      L2[LeafLesionColorAgent<br/>leaf_lesion_color]
      L3[LeafLesionTextureAgent<br/>leaf_lesion_texture]
      L4[LeafChlorosisAgent<br/>leaf_chlorosis]
      L5[LeafNecrosisAgent<br/>leaf_necrosis]
      L6[LeafCurlAgent<br/>leaf_curl]
      L7[LeafVeinPatternAgent<br/>leaf_vein_pattern]
      L8[LeafGeometryAgent<br/>leaf_geometry]
    end

    subgraph STEM["STEM (4 specialists)"]
      direction LR
      S1[StemLesionAgent<br/>stem_lesion]
      S2[StemPithAgent<br/>stem_pith<br/>DECISIVE SDS-vs-BSR fork]
      S3[StemSurfaceAgent<br/>stem_surface]
      S4[StemDiscolorationAgent<br/>stem_discoloration]
    end

    subgraph BELOW["BELOW-GROUND (2)"]
      direction LR
      R1[RootAgent<br/>root_visible<br/>cysts SCN / blue masses SDS]
      R2[CrownCollarAgent<br/>crown_collar]
    end

    subgraph REPRO["REPRODUCTIVE (2)"]
      direction LR
      F1[FlowerAgent<br/>flower]
      F2[FruitAgent<br/>fruit]
    end

    subgraph SIGNS["PATHOGEN SIGNS (1)"]
      G1[SporulationAgent<br/>sporulation]
    end

    subgraph PAT["WHOLE-PLANT PATTERNS (3)"]
      direction LR
      P1[WiltingAgent<br/>wilting]
      P2[DefoliationAgent<br/>defoliation<br/>petioles-attached SDS fork]
      P3[SpatialPatternAgent<br/>spatial_pattern]
    end

    subgraph DIAG["DIAGNOSTIC CROSS-CUTTERS (4)"]
      direction LR
      D1[ConcentricPatternAgent<br/>concentric_pattern]
      D2[ColorPaletteAgent<br/>color_palette<br/>color encoder]
      D3[LookAlikeCoTAgent<br/>look_alikes_visual<br/>decision-graph CoT]
      D4[SeverityVisualAgent<br/>severity_visible]
    end

    CONSOL[VisualDiagnosisAgent<br/>CoT consolidator<br/>1 triage / 2 decisive forks /<br/>3 dedup / 4 emit + CoT trace]
    DONE([per-pass final deltas + kappa + CoT trace])

    INPUT --> LEAF
    INPUT --> STEM
    INPUT --> BELOW
    INPUT --> REPRO
    INPUT --> SIGNS
    INPUT --> PAT
    INPUT --> DIAG

    LEAF --> CONSOL
    STEM --> CONSOL
    BELOW --> CONSOL
    REPRO --> CONSOL
    SIGNS --> CONSOL
    PAT --> CONSOL
    DIAG --> CONSOL
    CONSOL --> DONE

    classDef input fill:#ffd,stroke:#660
    classDef leaf fill:#dfd,stroke:#060
    classDef stem fill:#fde,stroke:#a06
    classDef below fill:#fed,stroke:#a60
    classDef repro fill:#fef,stroke:#606
    classDef signs fill:#dff,stroke:#066
    classDef pat fill:#eef,stroke:#33a
    classDef diag fill:#fdf,stroke:#909
    classDef consol fill:#cfead0,stroke:#063,stroke-width:2px
    classDef done fill:#efe,stroke:#060,stroke-width:2px
    class INPUT input
    class L1,L2,L3,L4,L5,L6,L7,L8 leaf
    class S1,S2,S3,S4 stem
    class R1,R2 below
    class F1,F2 repro
    class G1 signs
    class P1,P2,P3 pat
    class D1,D2,D3,D4 diag
    class CONSOL consol
    class DONE done
```

Each specialist emits `{deltas, confidence (κ), reasoning}` for the
ONE field it owns. The consolidator sees all 24 outputs **rendered
grouped by organ family** (so the model sees the same anatomical
clustering humans use when diagnosing), walks a 4-step chain-of-thought
(triage → decisive forks → dedup → emit), and produces the pass's
final delta list plus a CoT trace string. Validation against external
evidence happens in the §3d2 verifier stage after K-of-N agreement.

**Per-pass LLM calls** = 24 specialists + 1 consolidator = **25 calls**.
Qwen2.5-VL-7B handles ~50–100 concurrent on one A100, so wall-clock per
pass is unchanged from the legacy 5-call layout (~30–60 s).

### 3c. Animated walkthrough

![Phase 0R visual-symptom swarm — animated walkthrough](docs/assets/swarm_flow.gif)

*The animation predates the 24-agent expansion (it shows the previous
5-agent layout for legibility), but the structural intent is unchanged:
specialists examine the photograph against canonical KB, the verifier
consults the web per delta, and the running log accumulates only
image-grounded deltas that survive both swarm agreement and web
verification. The current implementation runs the same loop with 24
specialists grouped into 7 organ families, all parallel.*

### 3d. Cross-run K-of-N agreement filter

After all N passes complete, per-pass final-delta lists are pooled,
grouped by field, and clustered greedily on `image_shows` Jaccard. Only
clusters covering at least K distinct pass-indices survive.

```
Trace 0 final_deltas    [d_00, d_01]
Trace 1 final_deltas    [d_10]
Trace 2 final_deltas    [d_20, d_21, d_22]
                ...
Trace N-1 final_deltas  [...]
                  |
                  |  group by field
                  v
       +--------------------------+
       | stem_pith:               |
       |   (0, d_00) (2, d_20)    |
       |   (5, d_50)              |
       | leaf_chlorosis:          |
       |   (0, d_01) (1, d_10)    |
       |   ...                    |
       +-------------+------------+
                     |
                     |  greedy Jaccard cluster within each field
                     v
       +-------------------------------------------+
       | stem_pith Cluster A:                      |
       |   (0, "white pith with brown vascular")   |
       |   (2, "split stem: pith stays white")     |
       |   (5, "white center, chocolate cortex")   |
       |   distinct_runs = {0, 2, 5}               |
       |   support = 3                             |   keep (>= K)
       |                                           |
       | leaf_chlorosis Cluster B:                 |
       |   (0, "scattered yellow speckling")       |
       |   distinct_runs = {0}                     |
       |   support = 1                             |   drop  (< K)
       +-------------------------------------------+
                     |
                     v
       candidates (K-of-N survivors), each tagged
       with __support__ and __cluster_size__
```

### 3d2. Web-grounded verifier (Claude headless + WebSearch)

After the K-of-N agreement filter produces candidate observations, the
pipeline calls `pathome_kb.verifier.verify_candidates`. Claude receives
the full candidate batch plus canonical KB plus existing regional KB,
runs WebSearch queries against extension / APS / CABI / peer-reviewed
sources, and assigns each candidate a verification status:

| Status | Meaning | Goes into KB? |
|---|---|---|
| verified | strong external support; ≥1 high-quality citation | yes |
| weakly_supported | partial or indirect support | yes |
| provisional | no evidence but plausible, not contradicted | yes (with status flag) |
| novel_plausible | no evidence but coherent with canonical | yes (with status flag) |
| contradictory | external evidence contradicts | dropped (audit trail kept) |
| duplicate_existing | restates an already-stored regional delta | dropped; existing's support bumped |

Each accepted delta carries a `web_support` list of (url, quote)
citations and a one-sentence `reasoning` string. The verifier is opt-out
via `PATHOME_USE_VERIFIER=0`; the offline fallback marks every candidate
as `verification_status="unverified"` and lets the pipeline keep running.

### 3e. Conservative merge with existing KB

Candidates from agreement are merged into the **existing** regional
deltas for this state. Existing is never wiped.

```
existing  = [E0 (field=L, support=5),
             E1 (field=S, support=3)]
candidates = [C0 (field=L, image_shows close to E0: Jaccard >= tau),
              C1 (field=P, image_shows, no existing in field P),
              C2 (field=S, image_shows, contradicts E1: Jaccard < tau)]
                |
                |  for each candidate C:
                |    if exists E with same field AND Jaccard >= tau:
                |        E.support += C.support
                |        drop C
                |    else:
                |        append C (support default 1)
                v
merged = [E0 (support = 5 + C0.support = 8),
          E1 (support = 3),
          C1 (support = 1),
          C2 (support = 1)]

counts = {n_existing: 2, n_new_candidates: 3,
          n_added: 2, n_overlaps_bumped: 1}
```

Properties:
- **Idempotent on shape**: re-running with the same candidates against
  the same existing list adds no entries; only bumps support.
- **Existing always preserved**: prior Phase 0R deltas are never
  overwritten.
- **Contradictions kept**: low-Jaccard same-field deltas are added as
  separate entries; downstream consumers see all observations.

---

## 4. Phase PathomeOOD — KB-grounded CLIP training

PathomeOOD is a two-projector CLIP (BioCAP-inspired architecture) warm-
started from BioCLIP and trained on Bugwood with KB-grounded captions
synthesised from PathomeDB. Evaluated zero-shot on PlantVillage,
PlantDoc, and PlantWild — three heavy cross-domain distribution shifts
(field photos → lab cutouts → in-the-wild → mixed field).

```mermaid
flowchart TD
    REG[(artifacts/pathome_kb/Crop/final_registry.json<br/>canonical + per-state deltas)]
    CAP[build_pathomeood_captions.py<br/>KB-rich for 25 covered classes<br/>fallback template for 459 others<br/>state-aware delta selection]
    PARQ[(data/bugwood_captions/all_strategy.parquet<br/>11,513 rows / 484 classes)]
    SH[build_pathomeood_shards.py<br/>tar shard packager]
    SHARDS[(data/wds_shards/all_strategy/<br/>train/val/holdout tar shards<br/>image + taxon.txt + caption.txt)]
    BIO[BioCLIP HF init<br/>TreeOfLife-10M pretrained<br/>no caption supervision yet]
    MODEL[PathomeOOD model<br/>ViT-B/16 with two visual projectors<br/>proj_tax + proj_caption<br/>~800K trainable params]
    TRAIN[open_clip_train.main<br/>lock-image lock-text<br/>InfoNCE per text type<br/>11-variant matrix T01..T11]
    CKPT[(train_and_eval/checkpoints/Tnn/...)]
    PV[(PlantVillage<br/>~38 classes)]
    PD[(PlantDoc<br/>~27 classes)]
    PW[(PlantWild<br/>~17 classes)]
    HO[(Bugwood holdout<br/>retrieval bench)]
    EVAL[evaluate_pathomeood.py<br/>+ evaluate_pathomeood_retrieval.py<br/>+ evaluate_pathomeood_fewshot.py]
    BASE[5 off-shelf baselines<br/>CLIP / SigLIP / FG-CLIP /<br/>BioTrove / BioCLIP / BioCLIP-2]
    AGG[aggregate_pathomeood_tables.py<br/>walks all eval JSONs<br/>11 paper-style table markdowns]
    RPT[(results/pathomeood_report.md<br/>master report)]

    REG --> CAP --> PARQ --> SH --> SHARDS
    BIO --> MODEL
    SHARDS --> TRAIN
    MODEL --> TRAIN --> CKPT
    CKPT --> EVAL
    BASE --> EVAL
    PV --> EVAL
    PD --> EVAL
    PW --> EVAL
    HO --> EVAL
    EVAL --> AGG --> RPT

    classDef kb fill:#ffe,stroke:#660
    classDef stage fill:#eef,stroke:#33a
    classDef model fill:#efd,stroke:#060
    classDef src fill:#fde,stroke:#a06
    classDef deliv fill:#efe,stroke:#060,stroke-width:2px
    class REG,PARQ,SHARDS kb
    class CAP,SH,TRAIN,EVAL,AGG stage
    class BIO,MODEL model
    class PV,PD,PW,HO,BASE src
    class CKPT,RPT deliv
```

The 11-variant training matrix (`scripts/pathomeood_variants.sh`) covers
every reproducible BioCAP-paper-style ablation on Bugwood:

| ID | Caption strategy | Projector | Epochs | Subset |
|---|---|---|---|---|
| T01 | label_only | dual | 50 | all |
| T02 | summary_only | dual | 50 | all |
| T03 | canonical_full | dual | 50 | all |
| **T04** | **canonical_deltas_3 (MAIN)** | **dual** | **50** | **all** |
| T05 | canonical_deltas_1 | dual | 50 | all |
| T06 | canonical_deltas_5 | dual | 50 | all |
| T07 | canonical_deltas_7 | dual | 50 | all |
| T08 | canonical_deltas_3 | single | 50 | all |
| T09 | canonical_deltas_3 | dual | 100 | all |
| T10 | canonical_deltas_3 | dual | 50 | covered |
| T11 | canonical_deltas_3 | dual | 50 | non_covered |

What the trainer optimises per minibatch:

```
img_tax, img_cap   = visual_encoder(images, dual_projector=True)   # frozen + two heads
text_emb           = text_encoder(taxon_or_caption_text)            # shared, frozen
if text_type == 'taxon':
    logits = logit_scale     * img_tax @ text_emb.T
else:  # caption
    logits = logit_scale_cap * img_cap @ text_emb.T
loss   = info_nce(logits, identity_target)
```

Why this design at the ~11K-image Bugwood scale:

- BioCLIP already supplies bio-vocab features from TreeOfLife-10M
  pretraining. Projectors-only training learns the routing into label-
  side vs caption-side embedding spaces without disturbing the
  pretrained encoder weights — appropriate when you have ~24 imgs/class.
- KB-grounded captions are STATE-AWARE: per-image caption includes the
  top-K regional deltas for the image's state, so two images of the
  same disease taken in different states get different captions.
- Eval covers PV (lab), PD (mixed), PW (in-wild) so distribution-
  invariance is a real test, not a same-distribution score.

---

## 5. Data shape evolution

What lives where, and what gets preserved between layers.

```
                  artifacts/pathome_kb/<Crop>/final_registry.json
                  +-------------------------------------------+
   Phase 0    >   | {                                         |
                  |   "crop": "Soybean",                      |
                  |   "diseases": [{                          |
                  |     "disease_name": "Charcoal Rot",       |
                  |     "pathogen_scientific_name": {...},    |
                  |     "visual_symptoms": {...},             |
                  |     "treatments": {...},                  |
   Phase 0R   >   |     "regional_observations": {            |
                  |       "Alabama": {                        |
                  |         "state": "Alabama",               |
                  |         "image_ids": [...],               |
                  |         "deltas": [                       |
                  |           { field, canonical_says,        |
                  |             image_shows, image_quote,     |
                  |             image_id,                     |
                  |             __support__,                  |
                  |             __cluster_size__ }, ...       |
                  |         ],                                |
                  |         "__swarm_meta__": {...}           |
                  |       }, ...                              |
                  |     }                                     |
                  |   }, ...]                                 |
                  | }                                         |
                  +-------------------------------------------+
                                  |
                                  v   plantswarm/captioning.py::load_kb_profiles
                                  |   (reads per-crop final_registry.json directly)
                                  v
                  scripts/build_pathomeood_captions.py
                  +-------------------------------------------+
                  | per-image rows:                           |
                  |   image_id, image_path, crop, disease,    |
                  |   state, taxon_text, caption_text,        |
                  |   used_kb (KB-rich vs fallback template), |
                  |   split (train / val / holdout)           |
                  +-------------------------------------------+
                                  |
                                  v   scripts/build_pathomeood_shards.py
                                  v
                  data/wds_shards/<crop>_<strategy>/
                  +-------------------------------------------+
                  | {train,val,holdout}/shard-{NNNNNN}.tar    |
                  |   <key>.jpg                               |
                  |   <key>.taxon.txt                         |
                  |   <key>.caption.txt                       |
                  +-------------------------------------------+
                                  |
                                  v   open_clip_train.main (training loop)
                                  v
                              consumers
```

Legacy adapter: `pathome_kb/symptoms_adapter.py` can still produce
`artifacts/pathome_seed/symptoms_seed.json` (merged seed) for downstream
consumers that want it — but the PathomeOOD pipeline reads each
`final_registry.json` directly via `load_kb_profiles`, so the seed
file is no longer on the critical path.

When `PATHOME_TRACE_DIR` is set, Phase 0R also writes per-pass trace
records (`phase0r_traces.jsonl`) for diagnostics and the
`viz_traces.sh` aggregator. These are not consumed by PathomeOOD.

---

## 6. File map

```
PlantSwarm/
|-- README.md                              narrative + commands
|-- FLOW.md                                this file
|
|-- BugWood_Diseases.csv                   raw IPMNet export
|-- BugWood_Diseases_usable.csv            filtered (Setup output)
|
|-- configs/bugwood_pathome.yaml           swarm + model knobs
|
|-- pathome_kb/                            Phase 0 + Phase 0R orchestration
|   |-- pipeline.py                        per-crop orchestrator (CLI)
|   |-- internet_pipeline.py               Claude discovery + extraction + reconciliation
|   |-- regional_observation.py            per-tuple Qwen-swarm caller
|   |-- verifier.py                        Claude web-search verifier (Phase 0R)
|   |-- symptoms_adapter.py                registry to SymptomProfile JSON
|   |-- prompts/                           canonical-stage prompts
|   `-- shared.py / utils.py / config.py
|
|-- plantswarm/                            Qwen swarm
|   |-- delta_pipeline.py                  run_for_state, run_batch,
|   |                                       algorithm1_handoff,
|   |                                       _merge_with_existing,
|   |                                       _agreement_filter,
|   |                                       existing_deltas_for_state,
|   |                                       _TraceWriter (PATHOME_TRACE_DIR)
|   `-- latex/                             EMNLP 2026 paper sources
|
|-- train_and_eval/                        Phase PathomeOOD training + eval
|   |-- open_clip/                         BioCAP fork of open_clip with TWO
|   |                                       visual projectors (proj + caption_proj);
|   |                                       VisualTransformer.lock keeps both
|   |                                       projectors trainable under --lock-image
|   |-- open_clip_train/                   main.py / train.py / data.py — torchrun
|   |                                       entry, adapted to 2-field shards
|   |                                       (taxon.txt + caption.txt)
|   |-- evaluation/                        zero_shot_iid.py + retrieval_openclip.py
|   |                                       + metrics, params, utils, data
|   `-- imageomics/                        naming_eval + disk + helpers
|                                           (minimum imageomics subset our eval needs)
|
|-- agents/                                24 visual-symptom specialists +
|   |                                       VisualDiagnosisAgent CoT consolidator
|   |-- base_agent.py                      DELTA_USER_PROMPT (with CoT scaffold),
|   |                                       ALLOWED_DELTA_FIELDS (25 visual fields),
|   |                                       parse_agent_output, AgentDeltaOutput,
|   |                                       BaseAgent (FOCUS_QUESTION field)
|   |-- leaf_agents.py                     8 leaf specialists (lesion shape/color/
|   |                                       texture, chlorosis, necrosis, curl,
|   |                                       vein pattern, geometry)
|   |-- stem_agents.py                     4 stem specialists (lesion, pith,
|   |                                       surface, discoloration)
|   |-- root_agents.py                     RootAgent + CrownCollarAgent
|   |-- reproductive_agents.py             FlowerAgent + FruitAgent
|   |-- sign_agents.py                     SporulationAgent (pathogen signs)
|   |-- pattern_agents.py                  Wilting + Defoliation + SpatialPattern
|   |-- diagnostic_agents.py               ConcentricPattern + ColorPalette (color
|   |                                       encoder) + LookAlikeCoT (decision-graph)
|   |                                       + SeverityVisual
|   `-- diagnosis_agent.py                 VisualDiagnosisAgent — CoT consolidator
|                                           (class name DiagnosisAgent kept for
|                                            back-compat; AGENT_NAME is
|                                            "VisualDiagnosisAgent")
|
|-- plantswarm/
|   |-- delta_pipeline.py                  run_for_state, run_batch,
|   |                                       _agreement_filter, _merge_with_existing,
|   |                                       existing_deltas_for_state,
|   |                                       SPECIALIST_CLASSES = SPECIALIST_AGENTS (24)
|   |-- captioning.py                      build_disease_caption (7 strategies),
|   |                                       _top_regional_deltas (state-aware),
|   |                                       load_kb_profiles, caption_for_row
|   |                                       (returns (caption, used_kb))
|   `-- latex/                             paper sources
|
|-- pathome/                               schema for the KB
|   `-- symptoms.py                        SymptomLibrary, SymptomProfile,
|                                           CanonicalDisease, RegionalObservation,
|                                           RegionalDelta, Citation
|
|-- utils/
|   |-- vllm_client.py                     OpenAI-compatible vLLM client
|   `-- geo.py                             state centroid + AEZ (Setup)
|
|-- data/bugwood_loader.py                 _clean_disease + _map_crop (Setup)
|
|-- scripts/
|   |-- filter_bugwood_csv.py              Setup CLI
|   |-- ensure_state_image_cache.py        image cache CLI
|   |-- registry_to_excel.py               final_registry.json to xlsx
|   |
|   |--- PathomeOOD pipeline --------------
|   |-- build_pathomeood_captions.py       KB -> per-image (taxon, caption) parquet
|   |-- build_pathomeood_shards.py         parquet -> WebDataset tar shards
|   |-- pathomeood_variants.sh             T01..T11 variant matrix (the canonical
|   |                                       single source of truth)
|   |-- train_pathomeood.py                Python wrapper around open_clip_train.main
|   |-- submit_pathomeood_train.sh         SLURM: one variant
|   |-- submit_pathomeood_matrix.sh        SLURM: sbatch all 11 variants
|   |-- evaluate_pathomeood.py             zero-shot classification on PV/PD/PW
|   |-- evaluate_pathomeood_retrieval.py   Bugwood held-out R@k
|   |-- evaluate_pathomeood_fewshot.py     prototype-mean K-shot
|   |-- setup_plantdoc.py                  clone PlantDoc to data/eval/PlantDoc/
|   |-- fetch_baselines.py                 cache 5 off-shelf CLIP-style baselines
|   |                                       (CLIP / SigLIP / FG-CLIP / BioTrove /
|   |                                        BioCLIP / BioCLIP-2; biocap excluded)
|   |-- aggregate_pathomeood_tables.py     walk results/ JSONs -> 11 paper-table .md
|   |
|   |--- per-phase shells -----------------
|   |-- submit_pathome_setup_filter.sh     Nova: Setup
|   |-- setup_image_cache.sh               LOCAL/Nova: image cache
|   |-- run_phase0_local.sh                LOCAL: Phase 0 canonical (Claude)
|   |-- submit_phase0r_regional.sh         Nova: Phase 0R (24-agent visual swarm)
|   |
|   |--- viz shells -----------------------
|   |-- viz_kb.sh                          KB stats PNGs + tex
|   |-- viz_traces.sh                      Phase 0R trace PNGs + tex
|   |-- viz_all.sh                         run every viz in sequence
|   |-- build_latex_pdf.sh                 compile the paper
|   |
|   |--- umbrellas ------------------------
|   |-- e2e_local.sh                       LOCAL leg: setup + cache + P0 + push
|   |-- e2e_nova.sh                        Nova leg: pull + P0R + PathomeOOD
|   |                                       captions/shards/train/eval + push
|   |-- e2e_visualize.sh                   LOCAL post: pull + viz + aggregate + paper
|   |-- e2e_full.sh                        the umbrella that drives all three
|   |
|   `-- viz/                               Python visualizers
|       |-- kb_stats.py                    canonical+regional summary
|       |-- trace_stats.py                 Phase 0R trace stats
|       `-- _common.py                     shared output / matplotlib helpers
|
`-- smoke/                                 2-crop happy path
    |-- run_phase0_full.sh                 LOCAL P0 + tunneled P0R
    |-- run_phase0_local.sh                LOCAL canonical-only P0
    |-- bugwood_pathome_smoke.yaml         smaller N + Tmax
    `-- README.md
```

---

## 7. Env var reference

| Env var | Default | Controls |
|---|---|---|
| VLLM_BASE_URL | http://localhost:8000/v1 | OpenAI-compatible vLLM endpoint |
| VLLM_MODEL | Qwen/Qwen2.5-VL-7B-Instruct | Served model id |
| VLLM_TIMEOUT | 180 | Per-HTTP-call timeout (s) |
| VLLM_TEMPERATURE | 0.8 | Per-call sampling temperature |
| VLLM_N_RUNS | 10 (smoke: 5) | Stochastic traces per tuple |
| VLLM_AGREEMENT_MIN | 3 (smoke: 2) | K-of-N agreement floor |
| VLLM_SIM_THRESHOLD | 0.4 | Jaccard threshold for clustering + merge |
| PATHOME_USE_VERIFIER | 1 | Set to 0 to skip the Claude web-search verifier and pass candidates straight to merge as `unverified` |
| PATHOME_VERIFIER_TIMEOUT | 600 | Verifier `claude -p` timeout (seconds) |
| PATHOME_VERIFIER_MAX_TURNS | 30 | Verifier max turns (for WebSearch loops) |
| PATHOME_IMAGE_CACHE_DIR | — | Prepended to default cache search path |
| PATHOME_TRACE_DIR | — | When set, Phase 0R appends per-trace records to `<dir>/phase0r_traces.jsonl` |
| PATHOME_TRACE_FILE | phase0r_traces.jsonl | Trace JSONL filename within `PATHOME_TRACE_DIR` |
| ANTHROPIC_API_KEY | — (optional) | Speeds up Phase 0 reconciliation; falls back to `claude -p` |
| PATHOME_ONLY_CROPS | — | Comma-separated crop allowlist |
| PATHOME_USABLE_CSV | BugWood_Diseases_usable.csv | Filtered CSV path |
| PATHOME_SEED_FILE | artifacts/pathome_seed/symptoms_seed.json | Output seed JSON path |
| PATHOME_SEED_QUICK | 0 | Cap states per disease for fast iteration |
| CROP | Tomato | Crop tag for PathomeOOD captions + shards (use `all` for full Bugwood) |
| PV_ROOT, PW_ROOT, PLANTDOC_ROOT | data/eval/{PlantVillage,PlantWild,PlantDoc/test} | Eval-set roots for `evaluate_pathomeood*.py` |
| PATHOME_SKIP_PHASE0R / _CAPTIONS / _TRAIN / _BASELINES / _EVAL / _PUSH | 0 | Skip-knobs for `scripts/e2e_nova.sh` phases |
| PATHOME_WAIT | 0 | Set to 1 to use `sbatch --wait` per training variant (sequential) |

---

## 8. Run-report line

One line per (crop, disease, state) tuple printed by `run_batch`:

```
[7/50] OK  Soybean::Charcoal Rot / Alabama  deltas=8 (N=10, K>=3, existing=4, added=2, bumped=3)
        |   |              |      |          |     |       |          |          |
        |   |              |      |          |     |       |          |          +-- overlap-bumped candidates
        |   |              |      |          |     |       |          +------------- net-new this run
        |   |              |      |          |     |       +------------------------ prior deltas loaded
        |   |              |      |          |     +-------------------------------- K = agreement floor
        |   |              |      |          +-------------------------------------- N = stochastic traces
        |   |              |      +------------------------------------------------- final merged count
        |   |              +-------------------------------------------------------- state
        |   +----------------------------------------------------------------------- crop::disease
        +--------------------------------------------------------------------------- progress
```

Reading examples:

- `existing=0, added=8` → cold start; swarm produced 8 new agreed deltas
- `existing=4, added=2, bumped=3` → iterative re-run; 4 prior preserved,
  2 net-new, 3 candidates already known (support incremented)
- `existing=4, added=0, bumped=0` → swarm produced no new info; KB
  stable for this state
