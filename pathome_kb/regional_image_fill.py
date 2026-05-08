"""
pathome_kb/regional_image_fill.py
=================================
Image-grounded fill-in stage for ``regional_visuals[state]``.

After the text-grounded regional pass produces sourced citations from
extension-service literature, many discrete enum-style fields stay empty
(color, shape, margin, texture, sporulation, progression) because the
prose sources don't structure information that way. This stage looks at
the actual Bugwood photograph for each (crop, disease, state) tuple and
asks Claude (vision-capable, via ``claude -p`` with the Read tool) to
fill in *only those empty fields* using a state-/crop-/disease-aware
prompt.

The text-grounded fields (notes, distinctive_signs, plant_parts, etc.)
that the SAGE pipeline already populated are LEFT UNTOUCHED — this is
strictly additive. New citations carry ``grounding="image"`` to
distinguish them from URL-cited text.

Pipeline position:
    discovery → extraction → reconciliation
        → regional_extraction (text-grounded per-state)
        → THIS STAGE (image-grounded fill of empty fields)

Output: each crop's regional_registries.json gets an extra
``image_fills[<profile_id>][<state>]`` block; the adapter merges those
into ``SymptomProfile.regional_visuals[state]``.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

from .config import MAX_PARALLEL_EXTRACTIONS
from .regional_extraction import build_state_image_map
from .shared import claude_query_with_image, parse_json_result
from .utils import OUTPUT_DIR, get_crop_dir, load_json, save_json


# ---------------------------------------------------------------------------
# Cache lookup
# ---------------------------------------------------------------------------

_CACHE_DIRS = [
    Path("smoke/.bugwood_cache"),
    Path(".bugwood_cache"),
]


def _resolve_cached_image(image_id: str) -> Path | None:
    """Return the on-disk path for a Bugwood image_id, if cached locally.

    image_id format: "bugwood::<number>" → file <number>.{jpg,jpeg,png,webp}
    """
    if not image_id.startswith("bugwood::"):
        return None
    number = image_id.split("::", 1)[1]
    for d in _CACHE_DIRS:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = d / f"{number}{ext}"
            if p.is_file() and p.stat().st_size > 0:
                return p.resolve()
    return None


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# All fields the VLM is allowed to fill. Only fields that are currently
# empty in the regional_visuals[state] block will be requested.
ENUM_FIELDS = [
    "color", "shape", "margin", "texture",
    "sporulation", "progression",
    "plant_parts", "distinctive_signs",
]

IMAGE_FILL_PROMPT = """\
You are looking at one Bugwood Network field photograph of a plant disease.

Crop:    {crop}
Disease: {disease}
State:   {state}
Image:   {image_path}

You have already been given prose-style symptom descriptions from
extension-service literature. Some discrete visual fields are still
empty. Your job is to fill in ONLY those empty fields by looking at
the image. The empty fields are:

{empty_fields}

Reasoning rules:
1. Describe what you actually see in the image. Do not invent symptoms
   that aren't visible.
2. Be specific to this photo from {state} — if the lesion appears
   smaller / larger / paler / more advanced than typical, say so.
3. If a field cannot be determined from the image (e.g. you can't
   tell sporulation from a single photo), set its value to an empty
   array / empty string.
4. The "quote" field for each populated entry is your one-sentence
   description of what the image shows for that field.

Output a single JSON object matching the schema below. Only include
keys for the empty fields requested above. Empty fields you can't
determine: include the key with an empty value array/string.

Schema (each populated field):
{{
  "<field_name>": {{
    "value": <string or array of strings>,
    "quote": "<your one-sentence visual description>"
  }}
}}
"""


def _empty_field_image_fill_schema(empty_fields: List[str]) -> dict:
    """Per-call JSON schema constrained to the requested empty fields."""
    str_field = {
        "type": ["object", "null"],
        "properties": {
            "value": {"type": ["string", "null"]},
            "quote": {"type": ["string", "null"]},
        },
        "required": ["value", "quote"],
    }
    arr_field = {
        "type": ["object", "null"],
        "properties": {
            "value": {"type": ["array", "null"], "items": {"type": "string"}},
            "quote": {"type": ["string", "null"]},
        },
        "required": ["value", "quote"],
    }
    array_keys = {"plant_parts", "color", "texture", "sporulation", "distinctive_signs"}
    return {
        "type": "object",
        "properties": {k: (arr_field if k in array_keys else str_field) for k in empty_fields},
    }


# ---------------------------------------------------------------------------
# Per-(profile, state) call
# ---------------------------------------------------------------------------

def _empty_fields_for_state(text_record: dict) -> List[str]:
    """Inspect a regional record (output of regional_extraction) and return
    which ENUM_FIELDS are NOT covered by text-grounded content."""
    out: List[str] = []
    # Mapping from regional record fields to VisualSymptom fields:
    #   summary           → notes (not in ENUM_FIELDS; handled separately)
    #   diagnostic_features → distinctive_signs
    #   affected_parts    → plant_parts
    #   look_alikes       → confusion_diseases (not VLM-fillable from one image)
    has_distinctive = bool((text_record.get("diagnostic_features") or {}).get("value"))
    has_plant_parts = bool((text_record.get("affected_parts") or {}).get("value"))
    if not has_distinctive:
        out.append("distinctive_signs")
    if not has_plant_parts:
        out.append("plant_parts")
    # The discrete enum fields are always candidates because the text-grounded
    # regional record never produces them.
    out.extend(["color", "shape", "margin", "texture", "sporulation", "progression"])
    return out


def _fill_one(args: tuple) -> Tuple[str, str, dict]:
    profile_id, crop, disease, state, image_path, empty_fields = args
    prompt = IMAGE_FILL_PROMPT.format(
        crop=crop, disease=disease, state=state,
        image_path=str(image_path),
        empty_fields="\n".join(f"  - {f}" for f in empty_fields),
    )
    schema = _empty_field_image_fill_schema(empty_fields)
    raw = claude_query_with_image(
        prompt=prompt,
        image_path=image_path,
        system_prompt=(
            "You are a plant pathology vision agent. Describe only what is "
            "visible in the provided image. Output strictly JSON matching "
            "the schema."
        ),
        json_schema=schema,
        max_turns=5,
        timeout_secs=240,
    )
    record = parse_json_result(raw, f"image_fill_{profile_id}_{state}")
    if not isinstance(record, dict):
        record = {}
    return profile_id, state, record


# ---------------------------------------------------------------------------
# Per-crop runner
# ---------------------------------------------------------------------------

def run_regional_image_fill(
    crop: str,
    state_image_map: Dict[Tuple[str, str, str], List[str]],
    quick: bool = False,
) -> Dict[str, Dict[str, dict]]:
    """For one crop, image-fill empty fields in every (disease, state) entry
    of regional_registries.json.

    Returns ``{profile_id: {state: image_fill_record}}`` and also persists
    ``regional_image_fills.json`` next to the regional registry.
    """
    print(f"\n{'='*60}")
    print(f"REGIONAL IMAGE FILL — {crop}")
    print(f"{'='*60}")
    t0 = time.time()
    output_dir = get_crop_dir(crop)
    regional_path = output_dir / "regional_registries.json"
    if not regional_path.is_file():
        print(f"  [skip] no regional_registries.json for {crop}")
        return {}
    regional = load_json("regional_registries.json", output_dir=output_dir)

    todo: List[tuple] = []
    skipped_no_image = 0
    for profile_id, by_state in regional.items():
        if not isinstance(by_state, dict):
            continue
        try:
            crop_n, disease_n = profile_id.split("::", 1)
        except ValueError:
            continue
        if crop_n != crop:
            continue
        for state, record in by_state.items():
            if not isinstance(record, dict):
                continue
            empty = _empty_fields_for_state(record)
            if not empty:
                continue
            image_ids = state_image_map.get((crop_n, disease_n, state), [])
            image_path = None
            for img_id in image_ids:
                p = _resolve_cached_image(img_id)
                if p:
                    image_path = p
                    break
            if image_path is None:
                skipped_no_image += 1
                continue
            if quick:
                empty = empty[:4]   # cap at 4 fields/state in quick mode
            todo.append((profile_id, crop_n, disease_n, state, image_path, empty))

    print(f"  (profile, state) tuples to image-fill: {len(todo)}")
    if skipped_no_image:
        print(f"  [skipped: no cached image for {skipped_no_image} tuples]")
    if not todo:
        save_json("regional_image_fills.json", {}, output_dir=output_dir)
        return {}

    fills: Dict[str, Dict[str, dict]] = {}
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_EXTRACTIONS) as pool:
        futures = {pool.submit(_fill_one, t): t for t in todo}
        for fut in concurrent.futures.as_completed(futures):
            try:
                profile_id, state, record = fut.result()
                completed += 1
                n_filled = sum(
                    1 for k, v in record.items()
                    if isinstance(v, dict) and v.get("value") not in (None, "", [])
                )
                tag = "✓" if n_filled else "·"
                print(f"  [{completed}/{len(todo)}] {tag} {profile_id} / {state}  filled={n_filled}")
                fills.setdefault(profile_id, {})[state] = record
            except Exception as e:
                print(f"  ERROR: {e}")

    save_json("regional_image_fills.json", fills, output_dir=output_dir)
    print(f"  Saved: {output_dir / 'regional_image_fills.json'}")
    print(f"  Done in {time.time() - t0:.0f}s")
    return fills


# Convenience wrapper for the orchestrator
def run_regional_image_fill_all(
    csv_path: str | Path,
    crops: List[str],
    quick: bool = False,
) -> Dict[str, Dict[str, Dict[str, dict]]]:
    state_image_map = build_state_image_map(csv_path)
    fills_by_crop: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for crop in crops:
        fills_by_crop[crop] = run_regional_image_fill(
            crop=crop, state_image_map=state_image_map, quick=quick,
        )
    return fills_by_crop
