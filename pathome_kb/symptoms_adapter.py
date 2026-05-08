"""
pathome_kb/symptoms_adapter.py
==============================
Convert SAGE-style ``final_registry.json`` records into the JSON format
``pathome.SymptomLibrary.load`` consumes.

SAGE registry shape (per disease):
    {
      "disease_name":             "Early Blight",
      "pathogen_scientific_name": {value, url, quote},
      "type_of_disease":          {value, url, quote},
      "affected_parts":           {value: [...], url, quote},
      "visual_symptoms": {
        "summary":             {value, url, quote},
        "diagnostic_features": {value, url, quote},
        "look_alikes":         {value: [...], url, quote},
      },
      "confidence": "high"|"medium"|"low",
      "num_sources": int,
      "conflicts": [...]
    }

Adapter mapping:
- ``affected_parts.value``                 → ``VisualSymptom.plant_parts``
- ``visual_symptoms.diagnostic_features``  → ``VisualSymptom.distinctive_signs``
- ``visual_symptoms.look_alikes.value``    → ``VisualSymptom.confusion_diseases``
- ``visual_symptoms.summary.value``        → ``VisualSymptom.notes``
- The structured tuples (color, shape, margin, texture, sporulation,
  progression) are left empty by this pipeline because the SAGE pipeline
  emits free-form prose; downstream Phase 2 routing reads those fields
  off the auto-built reobservation_prompt rather than from prose, so
  empty fields are valid.
- Every populated field is accompanied by a ``Citation`` in
  ``VisualSymptom.sources[<field>]``.

The adapter never invents content — if the SAGE record is empty, the
SymptomProfile is created with an empty visual block and no citations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .utils import save_json  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _val(field: Any) -> Optional[Any]:
    """Pull the ``value`` out of a SAGE cited field, or return None."""
    if not isinstance(field, dict):
        return field if field else None
    v = field.get("value")
    if v in (None, "", []):
        return None
    return v


def _citation_record(field: Any, key_for_value: str = "") -> Optional[dict]:
    """Convert a SAGE cited field into a Citation-ready dict, or None."""
    if not isinstance(field, dict):
        return None
    v = field.get("value")
    if v in (None, "", []):
        return None
    if isinstance(v, list):
        v = "; ".join(str(x) for x in v if x)
    url = (field.get("url") or "").strip()
    quote = (field.get("quote") or "").strip()
    if not (url or quote):
        # nothing to cite
        return None
    return {"value": str(v), "url": url, "quote": quote}


def _strs(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if x is not None and str(x).strip()]
    return [str(value).strip()] if str(value).strip() else []


# ---------------------------------------------------------------------------
# One disease record → one symptoms-profile-shaped dict
# ---------------------------------------------------------------------------

def disease_to_profile_dict(
    crop: str,
    disease: str,
    record: dict,
) -> dict:
    """Map a SAGE registry entry to the dict shape SymptomProfile.from_dict accepts."""
    visual_section = record.get("visual_symptoms") or {}

    plant_parts = _strs(_val(record.get("affected_parts")))
    distinctive_signs_raw = _val(visual_section.get("diagnostic_features"))
    distinctive_signs = _strs(distinctive_signs_raw)
    look_alikes = _strs(_val(visual_section.get("look_alikes")))
    summary = _val(visual_section.get("summary"))
    notes = str(summary) if summary else ""

    sources: Dict[str, List[dict]] = {}

    cit = _citation_record(record.get("affected_parts"))
    if cit and plant_parts:
        sources["plant_parts"] = [cit]

    cit = _citation_record(visual_section.get("diagnostic_features"))
    if cit and distinctive_signs:
        sources["distinctive_signs"] = [cit]

    cit = _citation_record(visual_section.get("look_alikes"))
    if cit and look_alikes:
        sources["confusion_diseases"] = [cit]

    cit = _citation_record(visual_section.get("summary"))
    if cit and notes:
        sources["notes"] = [cit]

    pathogen_cit = _citation_record(record.get("pathogen_scientific_name"))
    if pathogen_cit:
        sources.setdefault("pathogen_scientific_name", []).append(pathogen_cit)

    type_cit = _citation_record(record.get("type_of_disease"))
    if type_cit:
        sources.setdefault("type_of_disease", []).append(type_cit)

    visual = {
        "plant_parts": plant_parts,
        "color": [],
        "shape": "",
        "margin": "",
        "texture": [],
        "sporulation": [],
        "distinctive_signs": distinctive_signs,
        "progression": "",
        "confusion_diseases": look_alikes,
        "notes": notes,
        "sources": sources,
        "reference_image_ids": [],
    }

    return {
        "profile_id": f"{crop}::{disease}",
        "crop": crop,
        "disease": disease,
        "visual": visual,
        "regional_visuals": {},
        "state_counts": {},
        "aez_counts": {},
        "total_observations": 0,
        "reference_ids": [],
        "reobservation_prompt": "",
        "swarm_observations": None,
    }


# ---------------------------------------------------------------------------
# Merge per-crop registries → SymptomLibrary seed JSON
# ---------------------------------------------------------------------------

def regional_record_to_visual_dict(
    state: str,
    record: dict,
) -> dict:
    """Convert a per-state regional registry record (from
    ``regional_extraction``) into a VisualSymptom-shaped dict, with
    image_id-tagged citations and reference_image_ids populated.
    """
    summary = record.get("summary") or {}
    diagnostic = record.get("diagnostic_features") or {}
    affected = record.get("affected_parts") or {}
    look = record.get("look_alikes") or {}
    image_ids = list(record.get("__image_ids__") or [])
    primary_image = image_ids[0] if image_ids else ""

    plant_parts = _strs(_val(affected))
    distinctive_signs = _strs(_val(diagnostic))
    confusion_diseases = _strs(_val(look))
    notes = str(_val(summary) or "")

    sources: Dict[str, List[dict]] = {}

    def _cite(field_dict: dict, ground_image: bool = True) -> Optional[dict]:
        if not isinstance(field_dict, dict):
            return None
        v = field_dict.get("value")
        if v in (None, "", []):
            return None
        if isinstance(v, list):
            v = "; ".join(str(x) for x in v if x)
        url = (field_dict.get("url") or "").strip()
        quote = (field_dict.get("quote") or "").strip()
        if not (url or quote):
            return None
        out = {"value": str(v), "url": url, "quote": quote}
        if ground_image and primary_image:
            out["image_id"] = primary_image
        return out

    cit = _cite(affected)
    if cit and plant_parts:
        sources["plant_parts"] = [cit]
    cit = _cite(diagnostic)
    if cit and distinctive_signs:
        sources["distinctive_signs"] = [cit]
    cit = _cite(look)
    if cit and confusion_diseases:
        sources["confusion_diseases"] = [cit]
    cit = _cite(summary)
    if cit and notes:
        sources["notes"] = [cit]

    return {
        "plant_parts": plant_parts,
        "color": [],
        "shape": "",
        "margin": "",
        "texture": [],
        "sporulation": [],
        "distinctive_signs": distinctive_signs,
        "progression": "",
        "confusion_diseases": confusion_diseases,
        "notes": notes,
        "sources": sources,
        "reference_image_ids": image_ids,
    }


def _merge_image_fill(
    visual_dict: dict,
    image_fill: dict,
    primary_image_id: str,
) -> dict:
    """Layer image-grounded fields on top of the text-grounded visual dict.

    Only fields currently empty in ``visual_dict`` are filled. Each
    image-grounded entry adds a Citation with grounding="image".
    """
    array_keys = {"plant_parts", "color", "texture", "sporulation",
                  "distinctive_signs", "confusion_diseases"}
    for field, payload in image_fill.items():
        if not isinstance(payload, dict):
            continue
        v = payload.get("value")
        if v in (None, "", []):
            continue
        # Don't overwrite text-grounded content
        existing = visual_dict.get(field)
        if (field in array_keys and existing) or (field not in array_keys and existing):
            continue
        if isinstance(v, list):
            visual_dict[field] = [str(x) for x in v if x]
            display = "; ".join(visual_dict[field])
        else:
            visual_dict[field] = str(v)
            display = visual_dict[field]
        cit = {
            "value": display,
            "url": "",
            "quote": str(payload.get("quote", "")).strip(),
            "image_id": primary_image_id,
            "grounding": "image",
        }
        visual_dict.setdefault("sources", {}).setdefault(field, []).append(cit)
    return visual_dict


def merge_registries_to_seed(
    registries: Iterable[Tuple[str, dict]],
    expected_classes: Iterable[Tuple[str, str]],
    regional_by_crop: Optional[Dict[str, Dict[str, Dict[str, dict]]]] = None,
    image_fills_by_crop: Optional[Dict[str, Dict[str, Dict[str, dict]]]] = None,
    min_observations: int = 3,
) -> dict:
    """Merge crop→registry pairs into the seed JSON.

    For every (crop, disease) in ``expected_classes`` we emit one
    SymptomProfile. If the registry has data, we use it; otherwise we
    emit an empty profile so the build pass picks it up.

    ``regional_by_crop`` is the optional output of
    ``regional_extraction.run_regional_extraction``: a dict mapping
    ``crop -> profile_id -> state -> regional_record``. When supplied,
    each profile's ``regional_visuals[state]`` is populated alongside
    the cross-region ``visual``.

    ``image_fills_by_crop`` is the optional output of
    ``regional_image_fill.run_regional_image_fill``: same shape, but
    each record is a dict of image-grounded field fills. Fields are
    layered on top of the regional text-grounded block when empty,
    with grounding="image" citations.
    """
    by_crop_disease: Dict[Tuple[str, str], dict] = {}
    for crop, registry in registries:
        if not isinstance(registry, dict):
            continue
        for d in registry.get("diseases", []) or []:
            disease = (d.get("disease_name") or "").strip()
            if not disease:
                continue
            by_crop_disease[(crop, disease)] = d

    regional_by_crop = regional_by_crop or {}
    image_fills_by_crop = image_fills_by_crop or {}

    profiles = []
    for crop, disease in expected_classes:
        record = by_crop_disease.get((crop, disease)) or {}
        prof = disease_to_profile_dict(crop, disease, record)
        # Attach regional_visuals[state] when we have a regional record.
        crop_regional = regional_by_crop.get(crop) or {}
        per_profile_regional = crop_regional.get(prof["profile_id"]) or {}
        crop_image_fills = image_fills_by_crop.get(crop) or {}
        per_profile_fills = crop_image_fills.get(prof["profile_id"]) or {}
        if per_profile_regional or per_profile_fills:
            regional_visuals: Dict[str, dict] = {}
            states = set(per_profile_regional.keys()) | set(per_profile_fills.keys())
            for state in states:
                rec = per_profile_regional.get(state) or {}
                fill = per_profile_fills.get(state) or {}
                visual_dict = regional_record_to_visual_dict(state, rec)
                if fill:
                    image_ids = visual_dict.get("reference_image_ids") or []
                    primary_image = image_ids[0] if image_ids else ""
                    _merge_image_fill(visual_dict, fill, primary_image)
                regional_visuals[state] = visual_dict
            prof["regional_visuals"] = regional_visuals
        profiles.append(prof)

    return {
        "min_observations": min_observations,
        "profiles": profiles,
    }


def write_seed_json(payload: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return p
