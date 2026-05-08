"""
pathome_kb/regional_extraction.py
=================================
Per-(crop, disease, state) extraction stage. Runs after the cross-region
reconciliation has produced ``final_registry.json`` for each crop. For
every (crop, disease, state) tuple where the Bugwood CSV has at least
one image, we ask the LLM to extract symptoms specific to that region
from the cached source records — and carry the Bugwood image_id forward
on each citation so downstream consumers can show the supporting field
photograph.

We deliberately reuse the cached ``raw_extractions.json`` (per-source
extraction artefacts produced by the standard extraction stage) rather
than re-discovering and re-fetching pages. The LLM is told which state
to scope to and asked to use only sentences that mention the state by
name or describe its climate. When the source records have no
state-specific signal the LLM returns empty — which is honest.

Output: ``regional_registries.json`` per crop dir, mapping
``"<Crop>::<Disease>"`` → ``{state: registry_record}``. The adapter
turns each entry into a ``regional_visuals[state]`` block with
``image_id``-tagged citations.
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .config import MAX_PARALLEL_EXTRACTIONS
from .shared import claude_query, parse_json_result
from .utils import get_crop_dir, load_json, save_json


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

REGIONAL_EXTRACTION_PROMPT = """\
You are extracting region-specific plant disease symptoms from a corpus
of source records that have already been pulled and parsed.

Crop:    {crop}
Disease: {disease}
State:   {state}

Below are the per-source extraction records produced by an earlier pass.
Each record has a `source_url` and an `extracted_diseases` list. Most of
those diseases are not the one we care about — you must filter to records
that match `{disease}` (or a clear synonym) and then extract symptoms
specifically observed in {state}.

Source records:
{extractions}

Output a single JSON object matching the schema below. Rules:

1. Only use VERBATIM quotes from the source records' `evidence` /
   `quote` fields. Never invent text.
2. Prefer sentences that explicitly mention `{state}` or describe its
   climate (e.g. "humid subtropical Southeast", "California Central
   Valley spring"). Sentences that describe the disease in general but
   are not state-specific may be used only if no state-specific evidence
   is available; in that case set `state_specific: false`.
3. If no relevant sentences exist for {state}, return empty fields
   throughout. An empty regional record is the correct answer when the
   sources do not contain state-scoped information.
4. The `quote` MUST be a substring of the source's `evidence` field.

Schema:
{{
  "state": "{state}",
  "state_specific": true | false,
  "summary": {{ "value": "<one-sentence summary>",
                "url": "<source url>",
                "quote": "<verbatim quote>" }},
  "diagnostic_features": {{ "value": "<one-sentence diagnostic feature>",
                            "url": "<source url>",
                            "quote": "<verbatim quote>" }},
  "affected_parts": {{ "value": ["leaf", "stem", ...],
                       "url": "<source url>",
                       "quote": "<verbatim quote>" }},
  "look_alikes": {{ "value": ["disease A", "disease B"],
                    "url": "<source url>",
                    "quote": "<verbatim quote>" }}
}}
"""

REGIONAL_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "state": {"type": "string"},
        "state_specific": {"type": "boolean"},
        "summary": {
            "type": ["object", "null"],
            "properties": {
                "value": {"type": ["string", "null"]},
                "url": {"type": ["string", "null"]},
                "quote": {"type": ["string", "null"]},
            },
            "required": ["value", "url", "quote"],
        },
        "diagnostic_features": {
            "type": ["object", "null"],
            "properties": {
                "value": {"type": ["string", "null"]},
                "url": {"type": ["string", "null"]},
                "quote": {"type": ["string", "null"]},
            },
            "required": ["value", "url", "quote"],
        },
        "affected_parts": {
            "type": ["object", "null"],
            "properties": {
                "value": {"type": ["array", "null"], "items": {"type": "string"}},
                "url": {"type": ["string", "null"]},
                "quote": {"type": ["string", "null"]},
            },
            "required": ["value", "url", "quote"],
        },
        "look_alikes": {
            "type": ["object", "null"],
            "properties": {
                "value": {"type": ["array", "null"], "items": {"type": "string"}},
                "url": {"type": ["string", "null"]},
                "quote": {"type": ["string", "null"]},
            },
            "required": ["value", "url", "quote"],
        },
    },
    "required": ["state", "state_specific", "summary", "diagnostic_features",
                 "affected_parts", "look_alikes"],
}


# ---------------------------------------------------------------------------
# CSV → (crop, disease, state) → [image_id]
# ---------------------------------------------------------------------------

def build_state_image_map(
    csv_path: str | Path,
    crop_normalize_fn=None,
    disease_clean_fn=None,
) -> Dict[Tuple[str, str, str], List[str]]:
    """Group the CSV by (NormCrop, NormDisease, Location) → list of bugwood image_ids."""
    out: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            crop = (row.get("NormCrop") or "").strip()
            disease = (row.get("NormDisease") or "").strip()
            state = (row.get("Location") or "").strip()
            if not (crop and disease and state):
                continue
            image_number = (row.get("Image Number") or "").strip()
            if not image_number:
                continue
            out[(crop, disease, state)].append(f"bugwood::{image_number}")
    return dict(out)


# ---------------------------------------------------------------------------
# Per-(profile, state) extraction
# ---------------------------------------------------------------------------

def _filter_extractions_for_disease(
    extractions: dict, disease: str,
) -> List[dict]:
    """Return source records that mention the target disease.

    Loose match: any source whose extracted_diseases contains a value
    whose disease_name has a 3+-character overlap with the target
    disease (case-insensitive substring). Not perfect but good enough
    to exclude unrelated diseases on multi-disease pages.
    """
    target = disease.lower()
    out: List[dict] = []
    for src in extractions.get("extractions", []):
        kept_diseases = []
        for d in src.get("extracted_diseases", []) or []:
            name_field = d.get("disease_name") or {}
            name = (name_field.get("value") if isinstance(name_field, dict) else name_field) or ""
            name = str(name).lower()
            if not name:
                continue
            if target in name or name in target:
                kept_diseases.append(d)
        if kept_diseases:
            kept_src = dict(src)
            kept_src["extracted_diseases"] = kept_diseases
            out.append(kept_src)
    return out


def _extract_one_state(args: tuple) -> Tuple[str, str, dict]:
    """One claude -p call for one (profile_id, state). Returns (profile_id, state, record)."""
    profile_id, crop, disease, state, extractions, image_ids = args

    prompt = REGIONAL_EXTRACTION_PROMPT.format(
        crop=crop, disease=disease, state=state,
        extractions=json.dumps({"extractions": extractions}, indent=2)[:18000],
    )
    raw = claude_query(
        prompt=prompt,
        system_prompt=(
            "You are a region-aware plant pathology data extractor. "
            "Use only verbatim quotes from the provided records. Output JSON only."
        ),
        json_schema=REGIONAL_EXTRACTION_SCHEMA,
        max_turns=5,
        timeout_secs=180,
    )
    record = parse_json_result(raw, f"regional_{profile_id}_{state}")
    if not isinstance(record, dict):
        record = {}
    # Inject image_ids on every populated citation. Pick the first for the
    # field's primary image_id; downstream callers can dereference all of them.
    primary_image_id = image_ids[0] if image_ids else ""
    for field_key in ("summary", "diagnostic_features", "affected_parts", "look_alikes"):
        cit = record.get(field_key)
        if isinstance(cit, dict) and (cit.get("value") not in (None, "", [])):
            cit["image_id"] = primary_image_id
    record["__image_ids__"] = image_ids   # all of them, for reference_image_ids
    return profile_id, state, record


def run_regional_extraction(
    crop: str,
    diseases: List[str],
    state_image_map: Dict[Tuple[str, str, str], List[str]],
    quick: bool = False,
) -> Dict[str, Dict[str, dict]]:
    """For one crop, build {profile_id: {state: regional_record}}."""
    print(f"\n{'='*60}")
    print(f"REGIONAL EXTRACTION — {crop} ({len(diseases)} diseases)")
    print(f"{'='*60}")
    t0 = time.time()

    output_dir = get_crop_dir(crop)
    extractions_path = output_dir / "raw_extractions.json"
    if not extractions_path.is_file():
        print(f"  [skip] no raw_extractions.json for {crop}")
        return {}

    extractions = load_json("raw_extractions.json", output_dir=output_dir)

    # Build the work list: (crop, disease, state) tuples with images.
    todo: List[tuple] = []
    for disease in diseases:
        per_disease_records = _filter_extractions_for_disease(extractions, disease)
        if not per_disease_records:
            continue
        # Find every state for this (crop, disease)
        states = sorted({
            state for (c, d, state), imgs in state_image_map.items()
            if c == crop and d == disease and imgs
        })
        if not states:
            continue
        if quick:
            states = states[:2]   # cap states/disease in quick mode
        for state in states:
            image_ids = state_image_map.get((crop, disease, state), [])
            profile_id = f"{crop}::{disease}"
            todo.append((profile_id, crop, disease, state, per_disease_records, image_ids))

    print(f"  (profile, state) tuples to extract: {len(todo)}")
    if not todo:
        return {}

    results: Dict[str, Dict[str, dict]] = defaultdict(dict)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_EXTRACTIONS) as pool:
        futures = {pool.submit(_extract_one_state, args): args for args in todo}
        for future in concurrent.futures.as_completed(futures):
            try:
                profile_id, state, record = future.result()
                completed += 1
                summary_val = (record.get("summary") or {}).get("value") or ""
                state_specific = record.get("state_specific")
                tag = "✓" if summary_val else "·"
                print(f"  [{completed}/{len(todo)}] {tag} {profile_id} / {state}  "
                      f"state_specific={state_specific}")
                results[profile_id][state] = record
            except Exception as e:
                print(f"  ERROR: {e}")

    save_json("regional_registries.json", results, output_dir=output_dir)
    print(f"  Saved: {output_dir / 'regional_registries.json'}")
    print(f"  Done in {time.time() - t0:.0f}s")
    return results
