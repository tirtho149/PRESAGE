"""
plantswarm/captioning.py
========================
Build per-image captions for BioCAP-on-Bugwood training.

Two text fields are emitted per image, matching BioCAP's two-projector
design:

    taxon  : short label-side string ("Tomato Early blight")
    caption: long descriptive text built from PathomeDB KB

The descriptive caption is composed from the per-crop
``artifacts/pathome_kb/<Crop>/final_registry.json`` records produced by
Phase 0 + Phase 0R. The seven caption strategies below correspond to
the ablations in the BioCAP paper:

    label_only          Table 3 row "None"
    summary_only        canonical summary only
    canonical_full      Table 3 "KB-canonical" — summary + diagnostic
                        features + look-alikes + affected parts
    canonical_deltas_1  Table 6 num_examples=1
    canonical_deltas_3  Table 3 main method + Table 6 num_examples=3
    canonical_deltas_5  Table 6 num_examples=5
    canonical_deltas_7  Table 6 num_examples=7

All "deltas_*" strategies hard-fail if regional_observations is empty
for any profile being captioned — the user must run Phase 0R first
(``scripts/submit_phase0r_regional.sh``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Strategy registry — keep sorted; tests assert this exact set.
STRATEGIES: Tuple[str, ...] = (
    "label_only",
    "summary_only",
    "canonical_full",
    "canonical_deltas_1",
    "canonical_deltas_3",
    "canonical_deltas_5",
    "canonical_deltas_7",
)

# Strategies that require at least one regional delta per profile.
DELTA_STRATEGIES: Tuple[str, ...] = tuple(s for s in STRATEGIES if "deltas_" in s)


HEALTHY_TEMPLATE = (
    "A healthy {crop} leaf with no visible disease symptoms — uniform "
    "green color, no lesions, no spots, no wilting, no chlorosis, no "
    "necrosis."
)


# ---------------------------------------------------------------------------
# Field-flattening helpers (lifted from observe/prototypes.py)
# ---------------------------------------------------------------------------

def _flatten_field(raw: Any) -> str:
    """Render a KB field value (str, list, dict-with-value) as plain text."""
    if raw is None:
        return ""
    if isinstance(raw, dict):
        raw = raw.get("value")
        if raw is None:
            return ""
    if isinstance(raw, list):
        return "; ".join(str(x) for x in raw if x is not None and str(x).strip())
    return str(raw).strip()


def _canonical_from_disease(disease_record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a final_registry.json disease record into the flat
    canonical-dict shape expected by ``build_disease_prototype``.

    The registry uses ``visual_symptoms.summary`` etc; the prototype
    builder expects flat keys at top level.
    """
    visual = disease_record.get("visual_symptoms") or {}
    return {
        "summary":                  visual.get("summary"),
        "diagnostic_features":      visual.get("diagnostic_features"),
        "look_alikes":              visual.get("look_alikes"),
        "affected_parts":           disease_record.get("affected_parts"),
        "pathogen_scientific_name": disease_record.get("pathogen_scientific_name"),
        "type_of_disease":          disease_record.get("type_of_disease"),
    }


# ---------------------------------------------------------------------------
# Regional delta ranking — used by all "deltas_*" strategies
# ---------------------------------------------------------------------------

_STATUS_RANK = {
    "verified":         5,
    "weakly_supported": 4,
    "provisional":      3,
    "novel_plausible":  2,
    "unverified":       1,
    "contradictory":    0,
}


def _top_regional_deltas(
    regional_observations: Dict[str, Any],
    top_k: int,
    state_filter: Optional[str] = None,
) -> List[str]:
    """Pick the top-K deltas across states (or restricted to one state).

    Returns short phrases like ``"in TX, lesions show concentric rings"``.
    Empty list if no usable deltas. If ``state_filter`` is given, only
    deltas from that state are considered.
    """
    candidates: List[Dict[str, Any]] = []
    for state, ro in (regional_observations or {}).items():
        if state_filter and state != state_filter:
            continue
        if not isinstance(ro, dict):
            continue
        for d in ro.get("deltas") or []:
            if not isinstance(d, dict):
                continue
            image_shows = str(d.get("image_shows") or "").strip()
            if not image_shows:
                continue
            candidates.append({
                "state":       state,
                "image_shows": image_shows,
                "field":       str(d.get("field") or "other"),
                "support":     int(d.get("swarm_support") or d.get("support") or 0),
                "status":      str(d.get("verification_status") or "unverified"),
            })

    candidates.sort(
        key=lambda d: (-_STATUS_RANK.get(d["status"], 0), -d["support"]),
    )
    out: List[str] = []
    seen = set()
    for c in candidates:
        phrase = f"in {c['state']}, {c['image_shows'][:160]}"
        key = (c["field"], phrase[:80])
        if key in seen:
            continue
        seen.add(key)
        out.append(phrase)
        if len(out) >= top_k:
            break
    return out


# ---------------------------------------------------------------------------
# Per-strategy caption builders
# ---------------------------------------------------------------------------

def _identity_clause(crop: str, disease: str, canonical: Dict[str, Any]) -> str:
    pathogen = _flatten_field(canonical.get("pathogen_scientific_name"))
    dtype    = _flatten_field(canonical.get("type_of_disease"))
    s = f"A field photograph of {crop} affected by {disease}"
    tag = []
    if pathogen: tag.append(pathogen)
    if dtype:    tag.append(f"{dtype.lower()} disease")
    if tag:
        s += " (" + ", ".join(tag) + ")"
    return s + "."


def _canonical_clauses(canonical: Dict[str, Any], include_full: bool) -> List[str]:
    """Body clauses from canonical KB text. ``include_full`` adds
    diagnostic features, look-alikes, and affected parts."""
    parts: List[str] = []
    summary = _flatten_field(canonical.get("summary"))
    if summary:
        parts.append(summary if summary.endswith(".") else summary + ".")
    if not include_full:
        return parts
    diag = _flatten_field(canonical.get("diagnostic_features"))
    if diag:
        parts.append(f"Diagnostic features: {diag}.")
    la = _flatten_field(canonical.get("look_alikes"))
    if la:
        parts.append(f"May be confused with: {la}.")
    ap = _flatten_field(canonical.get("affected_parts"))
    if ap:
        parts.append(f"Affected parts: {ap}.")
    return parts


def build_disease_caption(
    *,
    crop: str,
    disease: str,
    disease_record: Dict[str, Any],
    strategy: str,
    state: Optional[str] = None,
    max_chars: int = 1024,
) -> str:
    """Build one caption for one image under one strategy.

    ``disease_record`` is a raw entry from
    ``artifacts/pathome_kb/<Crop>/final_registry.json``. ``state`` (if
    given) restricts delta selection to that state.

    Raises ``ValueError`` for delta strategies when the disease has no
    regional_observations populated.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown caption strategy: {strategy!r} (expected one of {STRATEGIES})")

    if strategy == "label_only":
        return taxon_text(crop, disease)

    canonical = _canonical_from_disease(disease_record)
    parts: List[str] = [_identity_clause(crop, disease, canonical)]

    if strategy == "summary_only":
        parts.extend(_canonical_clauses(canonical, include_full=False))
    else:
        parts.extend(_canonical_clauses(canonical, include_full=True))

    if strategy in DELTA_STRATEGIES:
        k = int(strategy.rsplit("_", 1)[-1])  # canonical_deltas_3 -> 3
        ro = disease_record.get("regional_observations") or {}
        deltas = _top_regional_deltas(ro, top_k=k, state_filter=state)
        if not deltas:
            # Fall back to cross-state deltas if state-restricted lookup empty.
            deltas = _top_regional_deltas(ro, top_k=k, state_filter=None)
        if not deltas:
            raise ValueError(
                f"strategy={strategy!r} requires deltas, but disease "
                f"{crop}/{disease} has no regional_observations. "
                f"Run scripts/submit_phase0r_regional.sh first."
            )
        parts.append("Regional variations: " + "; ".join(deltas) + ".")

    text = " ".join(parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def build_healthy_caption(crop: str) -> str:
    return HEALTHY_TEMPLATE.format(crop=crop)


def taxon_text(crop: str, disease: str) -> str:
    """Short label-side string for the taxonomy projector.

    Format: ``"<Crop> <Disease>"`` (no punctuation, no template wrapper).
    BioCAP's open_clip_train pipeline tokenizes this directly.
    """
    return f"{crop} {disease}".strip()


# ---------------------------------------------------------------------------
# Loader — read every per-crop final_registry.json under artifacts/pathome_kb/
# ---------------------------------------------------------------------------

def load_kb_profiles(
    kb_root: str | Path = "artifacts/pathome_kb",
    crop_filter: Optional[Iterable[str]] = None,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Read every ``<crop>/final_registry.json`` and return a flat
    ``{(crop, disease): disease_record}`` mapping.

    ``crop_filter`` (iterable of crop names) restricts the load.
    """
    root = Path(kb_root)
    if not root.is_dir():
        raise FileNotFoundError(f"KB root not found: {root}")

    wanted = set(crop_filter) if crop_filter else None
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for crop_dir in sorted(root.iterdir()):
        if not crop_dir.is_dir():
            continue
        reg = crop_dir / "final_registry.json"
        if not reg.is_file():
            continue
        crop = crop_dir.name
        if wanted is not None and crop not in wanted:
            continue
        data = json.loads(reg.read_text())
        for disease_record in data.get("diseases") or []:
            disease = disease_record.get("disease_name")
            if not disease:
                continue
            out[(crop, disease)] = disease_record
    return out


def assert_deltas_populated(
    profiles: Dict[Tuple[str, str], Dict[str, Any]],
    *,
    strategies: Iterable[str],
) -> None:
    """Hard-fail if any profile is missing deltas while a delta strategy
    is requested. Per the user's directive: 'Block until Phase 0R runs.'
    """
    delta_needed = any(s in DELTA_STRATEGIES for s in strategies)
    if not delta_needed:
        return
    missing: List[str] = []
    for (crop, disease), rec in profiles.items():
        ro = rec.get("regional_observations") or {}
        has_delta = any(
            isinstance(v, dict) and (v.get("deltas") or [])
            for v in ro.values()
        )
        if not has_delta:
            missing.append(f"{crop}/{disease}")
    if missing:
        head = ", ".join(missing[:5]) + (f", ... (+{len(missing)-5} more)" if len(missing) > 5 else "")
        raise RuntimeError(
            f"Phase 0R has not populated regional_observations for "
            f"{len(missing)} profile(s): {head}. Run "
            f"scripts/submit_phase0r_regional.sh on Nova before building "
            f"caption variants that require deltas."
        )


# ---------------------------------------------------------------------------
# High-level convenience: one (crop, disease, state) → caption per strategy
# ---------------------------------------------------------------------------

FALLBACK_TEMPLATE = "A field photograph of {crop} affected by {disease}."


def build_fallback_caption(crop: str, disease: str, strategy: str) -> str:
    """Minimal caption for (crop, disease) pairs that have no KB profile.

    Bugwood spans 484 (crop, disease) pairs, but PathomeDB currently
    only has profiles for two crops (Tomato + Soybean = 25 pairs). The
    remaining 459 pairs ride on this fallback so they can still join
    training and so PV/PlantDoc/PlantWild can be evaluated on every
    matching class — not just the KB-covered ones.

    The fallback intentionally mirrors the open-vocabulary template the
    original OBSERVE eval used for unseen PV classes, so the eval-time
    text geometry matches train-time supervision for non-KB classes.
    """
    if strategy == "label_only":
        return taxon_text(crop, disease)
    return FALLBACK_TEMPLATE.format(crop=crop, disease=disease)


def caption_for_row(
    *,
    crop: str,
    disease: str,
    state: Optional[str],
    profiles: Dict[Tuple[str, str], Dict[str, Any]],
    strategy: str,
) -> Tuple[str, bool]:
    """Resolve the right disease record and emit one caption.

    Returns ``(caption_text, used_kb)``. ``used_kb`` is True when the
    (crop, disease) had a KB profile and full ``build_disease_caption``
    was used; False when the row fell back to the minimal template.

    Bypasses the KB for 'healthy' diseases (PathomeDB is disease-only).
    """
    if disease.lower() == "healthy":
        return build_healthy_caption(crop), False
    rec = profiles.get((crop, disease))
    if rec is None:
        return build_fallback_caption(crop, disease, strategy), False
    cap = build_disease_caption(
        crop=crop, disease=disease,
        disease_record=rec, strategy=strategy, state=state,
    )
    return cap, True
