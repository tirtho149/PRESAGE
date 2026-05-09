# `pathome_kb/` — How the KB looks and how it's built

This module produces the **PathomeDB seed** that Phase 1 (`scripts/build_pathome.py`) consumes. The seed is a single JSON document with one `SymptomProfile` per `(crop, disease)` class in your filtered Bugwood CSV.

The schema deliberately **separates** what's invariant about a disease from what a specific photo of that disease in a specific state actually shows:

```
SymptomProfile
   ├── canonical          ← cross-region, sourced from extension-service URLs
   │     · summary, diagnostic_features, look_alikes
   │     · treatments              (NEW)
   │     · affected_parts, pathogen_scientific_name, type_of_disease
   │     · sources : { field → [{ value, url, quote, grounding="text" }] }
   │
   └── regional_observations[state]   ← per-state, VLM-grounded in a photo
         · image_ids                  ← Bugwood photographs from this state
         · severity                   (mild | moderate | advanced | late-season)
         · lesion_morphology          (one-sentence visual description)
         · affected_organs            (what THIS image shows)
         · spread_pattern             (lower canopy | scattered | uniform | …)
         · variations_from_canonical  ← bullets: how this image diverges
         · sources : { field → [{ value, image_id, quote, grounding="image" }] }
```

The earlier schema duplicated canonical text per state — the new split fixes that.

---

## What the KB looks like (worked example, real numbers from the Tomato + Soybean smoke)

```jsonc
{
  "profile_id": "Tomato::Early Blight",
  "crop": "Tomato",
  "disease": "Early Blight",

  "canonical": {
    "summary": "Lesions first develop on lower leaves as small, brownish-black
                spots which can expand to about 1/4-1/2 inch in diameter,
                often with concentric rings forming a target-board pattern.",
    "diagnostic_features": ["concentric rings", "yellow halo", "lower leaves first"],
    "look_alikes": ["Septoria leaf spot", "Late blight"],
    "treatments": ["crop rotation", "copper-based fungicides",
                   "remove infected plant debris", "resistant varieties"],
    "affected_parts": ["Foliar", "Stem", "Pod", "Seed"],
    "pathogen_scientific_name": "Alternaria linariae",
    "type_of_disease": "Fungal",
    "sources": {
      "summary":            [{value, url: "https://content.ces.ncsu.edu/early-blight-of-tomato",
                              quote: "verbatim from the page",
                              grounding: "text"}],
      "treatments":         [{value, url: "https://extension.umn.edu/...",
                              quote: "...rotate with non-solanaceous crops...",
                              grounding: "text"}],
      "diagnostic_features":[ ... ],
      ...
    }
  },

  "regional_observations": {
    "Alabama": {
      "image_ids": ["bugwood::1568038"],
      "severity": "late-season",
      "lesion_morphology": "Individual lesions are not resolvable at field-scale
                            distance; the canopy reads as wholesale browning and
                            shrivelling, with bare brown stems exposed.",
      "affected_organs": ["leaves", "stems"],
      "spread_pattern": "uniform across row",
      "variations_from_canonical": [
        "Image shows whole-plant defoliation rather than the small 1/4–1/2
         inch bulls-eye lesions on lower leaves the canonical describes",
        "No yellow halo or concentric-ring detail visible at this scale —
         disease is past the diagnostic-lesion stage",
        "Fruit still appears intact and ripening on the vine; canonical
         sunken calyx-end fruit lesions are not visible"
      ],
      "sources": {
        "severity":   [{value:"late-season",
                        quote:"canopy nearly defoliated...",
                        image_id:"bugwood::1568038", grounding:"image"}],
        "lesion_morphology": [{...same image, image-grounded...}],
        "spread_pattern":    [{...}],
        "variations_from_canonical": [{...}]
      }
    },

    "Connecticut": {
      "image_ids": ["bugwood::5559537"],
      "severity": "moderate",
      "lesion_morphology": "Roughly circular dark brown to black spots about
                            5–10 mm across with faint concentric rings and a
                            tan/gray center, some coalescing.",
      "affected_organs": ["leaf"],
      "spread_pattern": "scattered, lower canopy",
      "variations_from_canonical": [
        "Clear bulls-eye concentric ring pattern only weakly visible — rings
         are FAINT rather than pronounced",
        "Yellow halo around individual spots is subtle/absent on the central
         leaflet — chlorosis appears more as broad leaf yellowing",
        "Visible lesions are LARGER than canonical 1/4 inch (~6 mm) — the
         tip lesion is a coalesced patch >2 cm across"
      ],
      "sources": { ... }
    }
  }
}
```

The two regional blocks describe **different things in different photos** of the same disease — Alabama's image shows late-stage canopy collapse, Connecticut's image shows individual mid-stage lesions. Both flag concrete deltas vs the canonical text. The canonical block stays fixed across states.

---

## How it's built (5 stages)

```
                     YOU (laptop)                                              GitHub                  Nova
                     ───────────                                               ──────                  ────
BugWood_Diseases_usable.csv
 │
 ▼
┌──────────────────────────────────────────────────────────┐
│ 1. DISCOVERY            claude -p WebSearch              │
│    one search per disease (parallel)                      │
│    → discovery_results.json                              │
└──────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────┐
│ 2. EXTRACTION           fetch URL + claude -p            │
│    per-source extraction with VERBATIM quotes from page   │
│    text. Now also captures `treatments` (mgmt section).   │
│    → raw_extractions.json (per crop)                     │
└──────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────┐
│ 3. RECONCILIATION       claude -p (no API key needed)    │
│    merge per-source records into ONE canonical entry per  │
│    disease. Every field stays {value, url, quote}.        │
│    → final_registry.json (per crop)                      │
│      → ❶ canonical block of every SymptomProfile         │
└──────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────┐
│ 4a. STATE-AWARE IMAGE CACHE TOP-UP                        │
│     scripts/ensure_state_image_cache.py downloads ONE     │
│     Bugwood image per (crop, disease, state) tuple.       │
│     → smoke/.bugwood_cache/<image_number>.jpg              │
│                                                            │
│ 4b. PER-STATE VLM OBSERVATION   claude -p + Read tool      │
│     For each (crop, disease, state) with a cached image:   │
│       reads the image,                                     │
│       reads the canonical entry from step 3,               │
│       emits: severity, lesion_morphology, affected_organs,│
│               spread_pattern, variations_from_canonical    │
│     → regional_observations.json (per crop)                │
│       → ❷ regional_observations[state] of every profile   │
└──────────────────────────────────────────────────────────┘
 │
 ▼
┌──────────────────────────────────────────────────────────┐
│ 5. ADAPTER + MERGE      symptoms_adapter.merge…          │
│    layers ❶ + ❷ into one SymptomProfile per (crop, disease)│
│    → smoke/artifacts/pathome_seed/symptoms_seed.json     │
└──────────────────────────────────────────────────────────┘
 │
 ▼ git push -f
                                                          ┌─────────┐    git pull
                                                          │  GitHub │ ─────────────►  PathomeDB Phase 1
                                                          └─────────┘                 (Nova A100)
```

Stages 1–3 are the **SAGE port** (`internet_pipeline.py`). Stage 4b is the **Pathome image-grounded observation** (`regional_observation.py`). Stage 5 is the adapter (`symptoms_adapter.py`).

---

## Where each stage lives

| Stage | Code | Output | LLM input |
|---|---|---|---|
| 1 Discovery | `internet_pipeline._run_targeted_discovery` | `discovery_results.json` | (disease name) → `claude -p WebSearch` |
| 2 Extraction | `internet_pipeline._run_extraction` | `raw_extractions.json` | (URL page text) → `claude -p` (no tools) → fields with verbatim quotes + treatments |
| 3 Reconciliation | `internet_pipeline._run_reconciliation` | `final_registry.json` | (per-source records) → `claude -p` → canonical disease entry |
| 4a Image cache | `scripts/ensure_state_image_cache.py` | `smoke/.bugwood_cache/` | (no LLM) — pure URL fetch |
| 4b Regional observation | `regional_observation.run_regional_observation` | `regional_observations.json` | (image + canonical brief) → `claude -p --allowedTools Read` → severity/morphology/organs/spread/variations |
| 5 Merge | `symptoms_adapter.merge_registries_to_seed` | `symptoms_seed.json` | (no LLM) — pure assembly |

All per-crop artefacts land under `artifacts/pathome_kb/<Crop>/`. The merged seed lands at `smoke/artifacts/pathome_seed/symptoms_seed.json` (smoke) or `artifacts/pathome_seed/symptoms_seed.json` (production).

---

## What stage 4b actually puts in the LLM (the new VLM stage)

```
   ┌──────────────────────────────────────────────────────────┐
   │ Image:    /Users/.../.bugwood_cache/1568038.jpg           │
   │ Canonical: "Lesions first develop on lower leaves as       │
   │             small, brownish-black spots ~1/4-1/2 inch with │
   │             concentric rings forming a target-board…"     │
   │                                                            │
   │ claude -p --allowedTools Read --max-turns 5                 │
   │   → reads image at the given path                           │
   │   → emits JSON:                                             │
   │     { severity, severity_quote,                             │
   │       lesion_morphology, lesion_morphology_quote,           │
   │       affected_organs, affected_organs_quote,               │
   │       spread_pattern, spread_pattern_quote,                 │
   │       variations_from_canonical: ["..." , "..."] }         │
   │                                                            │
   │ Honest empty: if a field can't be determined from a single │
   │ photo (e.g. shape from a whole-canopy shot), the model     │
   │ returns "" instead of guessing.                            │
   └──────────────────────────────────────────────────────────┘
```

This is the only stage where the model is allowed to write text that isn't a verbatim quote — but every observation is anchored to the specific Bugwood image_id, and the `variations_from_canonical` bullets explicitly call out where the image disagrees with the canonical text.

---

## Aggregate (smoke; Tomato + Soybean, full coverage)

After running `bash smoke/run_phase0_full.sh`:

```
profiles total                    : 25
profiles w/ canonical summary     : 25
profiles w/ canonical treatments  :  ~22  (varies by source coverage)
profiles w/ regional observations : 25
total per-state blocks            : ~45–60
total variations bullets          : ~150–250
text-grounded citations (canonical): ~400+
image-grounded citations (regional): ~400+
```

Each (crop, disease, state) tuple where Bugwood has a cached image
gets one regional observation block; the rest of the seed is the
shared canonical text.

---

## Files on disk after a run

```
artifacts/pathome_kb/Tomato/
  ├── discovery_results.json     63 candidate URLs (claude -p WebSearch)
  ├── raw_extractions.json       per-source extraction with verbatim quotes
  ├── final_registry.json        14 canonical disease entries (with treatments)
  └── regional_observations.json 27 per-state VLM observations + variations

artifacts/pathome_kb/Soybean/
  └── (same five files)

smoke/.bugwood_cache/            ~118 JPEGs (one per (crop, disease, state) tuple)
smoke/artifacts/pathome_seed/symptoms_seed.json    final assembled KB → Phase 1 input
```

---

## CLI surface

```bash
# Single command — perfect-KB regenerate end-to-end
bash smoke/run_phase0_full.sh

# Direct pathome_kb invocation
python -m pathome_kb \
  --csv BugWood_Diseases_usable.csv \
  --out artifacts/pathome_seed/symptoms_seed.json \
  --regional                                  # turn on stage 4b

# Restrict scope while iterating
--quick                            # smaller per-stage caps
--only-crops "Tomato,Soybean"      # crop allowlist
--limit-crops 5                    # first N crops alphabetically
--resume-from extraction           # use cached upstream artefacts
--no-cache                         # re-run even if final_registry.json exists
--regional-only                    # skip stages 1-3, just rerun stage 4b on
                                   # cached final_registry.json
```

The smoke wrapper:

```bash
bash smoke/run_phase0_full.sh                      # full coverage, ~45-90 min, ~$5-15
FULL_QUICK=1 bash smoke/run_phase0_full.sh         # fast, ~15-25 min, ~$1-3
FULL_KEEP_CACHE=1 bash smoke/run_phase0_full.sh    # reuse cached registries
FULL_SKIP_KB=1 bash smoke/run_phase0_full.sh       # only cache top-up + setup
```

---

## Authentication

| What | Why | Failure mode |
|---|---|---|
| `claude` CLI on PATH | Stages 1, 2, 3, 4b all use `claude -p`; 4b uses the Read tool | `pathome_kb` exits with "claude CLI not on PATH" |
| `claude auth login` once | OAuth login for the CLI; sessions reuse the token | First `claude -p` blocks for browser sign-in |
| `ANTHROPIC_API_KEY` (optional) | Stage 3 reconciliation can use the SDK directly when present — slightly faster than subprocess. Without it, reconciliation falls back to `claude -p` automatically. | None — the fallback is transparent |

Nova compute nodes can't run `claude auth login`, which is why Phase 0 runs locally and the seed file is shuttled through git.

---

## Cost & time (approximate)

| Run mode | LLM calls | Walltime | Cost (claude -p over OAuth) |
|---|---|---|---|
| Smoke `FULL_QUICK=1` (2 crops) | ~80 | ~15–25 min | ~$1–3 |
| Smoke full (2 crops) | ~150 | ~45–90 min | ~$5–15 |
| Production full (484 classes) | ~2500 | ~16–24 h | ~$60–180 |

(Production estimates assume `MAX_PARALLEL_EXTRACTIONS=4` in `config.py` and Sonnet 4.6 default. Bigger parallelism reduces wall but not cost; OAuth quota is the practical ceiling.)

---

## What changed vs the previous schema

| Previous | Now |
|---|---|
| `SymptomProfile.visual` (cross-region text + a few enum fields) | `SymptomProfile.canonical` — full canonical disease block with `treatments` |
| `SymptomProfile.regional_visuals[state]` — duplicated canonical text per state | `SymptomProfile.regional_observations[state]` — VLM observation of THIS state's photo + `variations_from_canonical` deltas |
| Two stages (`regional_extraction.py` + `regional_image_fill.py`) | One stage (`regional_observation.py`) |
| Treatments field absent | Added to `extraction.py` and `reconciliation.py` prompts + schemas |
| Citation `grounding` field optional | Required: `"text"` (canonical) or `"image"` (regional observation) |
| Old artefacts: `regional_registries.json`, `regional_image_fills.json` | New artefact: `regional_observations.json` |
