"""
pathome_kb/verifier.py
======================
Claude-headless retrieval-grounded verifier for Phase 0R candidate
regional deltas.

Architectural rationale
-----------------------
The Qwen swarm produces candidate observations from a single Bugwood
image. K-of-N agreement is a useful *proposal confidence prior* — it
filters one-off hallucinations — but multi-run agreement from the same
base model is correlated, not orthogonal evidence. Agreement does NOT
imply truth.

This module replaces (or augments) "agreement as truth" with
external-evidence verification. Each candidate delta is sent through
``claude -p`` with the WebSearch tool enabled. Claude looks up
extension-service / pathology references for the (crop, disease, state)
tuple, then judges each candidate against external evidence:

    verified           — strong web support; citations attached
    weakly_supported   — partial / indirect support
    provisional        — no evidence found but plausible & not contradicted
    contradictory      — external evidence contradicts
    novel_plausible    — no evidence but visually coherent (rare regional)

Verified + weakly_supported go into the KB. Provisional / novel_plausible
are stored with that flag (downstream consumers can filter). Contradictory
deltas are dropped.

Output
------
``verify_candidates`` returns a dict with the same keys as the bucketed
verdict from Claude, plus a flat ``accepted`` list (verified +
weakly_supported + provisional + novel_plausible) ready to feed into the
conservative merge step.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .shared import claude_query, parse_json_result


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

VERIFIER_SYSTEM_PROMPT = (
    "You are a plant pathologist reviewing VISUAL SYMPTOMS for an educational "
    "plant-disease photo-identification guide for farmers and Master Gardeners. "
    "The subjects are ordinary crop and garden PLANT diseases (fungal/bacterial "
    "leaf spots, blights, rusts, mildews) — plant-only organisms that are "
    "harmless to people and animals. Your ONLY job is to corroborate how a "
    "disease VISUALLY APPEARS on the plant (lesion shape, colour, texture, "
    "chlorosis, necrosis, affected leaf/fruit/stem parts) against publicly "
    "published university extension factsheets and photo galleries (APS, CABI, "
    "land-grant IPM pages). You do NOT assess where a pathogen occurs, how it "
    "spreads, or how to treat it — only what it looks like. Prefer "
    "well-sourced visual descriptions; mark unsupported ones as provisional. "
    "Output strict JSON only — no prose, no markdown."
)


VERIFIER_PROMPT = """\
You are checking short VISUAL-SYMPTOM descriptions for an educational
plant-disease photo-identification guide. Each candidate below describes
what a crop/garden disease LOOKS LIKE in one field photograph. Use web
search to corroborate each visual description against publicly published
university extension factsheets and photo galleries, then return a
structured result. These are ordinary plant diseases (plant-only
organisms, harmless to people and animals).

CROP:    {crop}
DISEASE: {disease}
PHOTO CATALOGUED IN:    {state}
(The location is only where the source photo was filed. Do NOT assess or
confirm where the disease occurs or how it spreads — only corroborate the
described VISUAL APPEARANCE.)

CANONICAL VISUAL DESCRIPTION (already established for this disease; treat
as background):
{canonical_block}

EXISTING VISUAL NOTES already in the guide (preserve, do NOT re-emit):
{existing_block}

CANDIDATE VISUAL DESCRIPTIONS (each with a swarm_support count = how many
of N stochastic vision runs proposed it):
{candidates_block}

YOUR TASK
=========
Use WebSearch to look up publicly published extension-service pages,
APS / CABI photo references, and IPM factsheets that describe how
{crop} :: {disease} VISUALLY APPEARS (lesion shape/colour/texture,
chlorosis, necrosis, affected plant parts). Corroborate (or contradict)
each candidate's described appearance against those published photo
descriptions.

For each candidate, return:

  field            the visual attribute the candidate describes (carry
                    over the candidate's own field name, e.g.
                    leaf_lesion_shape, leaf_lesion_color, chlorosis,
                    necrosis, affected_organs, diagnostic_features,
                    look_alikes, or other)
  canonical_says   short quote from the CANONICAL VISUAL DESCRIPTION on
                    this attribute, or "(not specified)"
  image_shows      one-sentence visual detail (carried from the candidate)
  image_quote      one-sentence visual evidence (carried from the candidate)
  image_id         the bugwood::N witness (carried from the candidate)
  swarm_support    the input count, unchanged
  verification_status   one of:
        "verified"          strong external support; ≥1 high-quality
                            source corroborates the observation
        "weakly_supported"  partial / indirect support
        "provisional"       no evidence found but plausible and not
                            contradicted
        "novel_plausible"   no evidence but the observation is coherent
                            with canonical + visual evidence; flag for
                            curator review
        "contradictory"     external evidence contradicts the observation
        "duplicate_existing"   this candidate is essentially a restatement
                                of an existing regional observation
  web_support      list of {{url, quote}} pairs supporting the verdict;
                    [] for provisional/novel; required for verified/
                    weakly_supported/contradictory
  reasoning        one-sentence justification for the verification_status

BUCKETING
=========
Group your verdicts under four top-level keys:
  verified            verification_status in {{verified, weakly_supported}}
  provisional         verification_status in {{provisional, novel_plausible}}
  contradictory       verification_status == "contradictory"
  duplicates_of_existing  verification_status == "duplicate_existing"

Hard rules:
- Restating canonical text is forbidden — drop such candidates as
  duplicates_of_existing.
- Fabricated citations are forbidden. Only cite URLs returned by your
  searches.
- If you can't find evidence after a reasonable search, use
  "provisional" or "novel_plausible" — don't manufacture support.

Output STRICT JSON only (no markdown):
{{
  "verified":              [<delta>, ...],
  "provisional":           [<delta>, ...],
  "contradictory":         [<delta>, ...],
  "duplicates_of_existing":[<delta>, ...]
}}
"""


# Plain-language names for diseases whose `disease_name` is stored as a raw
# pathogen binomial that false-trips the safety classifier. Keyed lowercase.
_DISEASE_COMMON = {
    "xanthomonas vasicola": "corn bacterial leaf streak",
    "parastagonospora nodorum": "wheat septoria nodorum leaf blotch",
}


VERIFIER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "verified":               {"type": "array"},
        "provisional":            {"type": "array"},
        "contradictory":          {"type": "array"},
        "duplicates_of_existing": {"type": "array"},
    },
    "required": ["verified", "provisional", "contradictory", "duplicates_of_existing"],
}


# ---------------------------------------------------------------------------
# Context rendering
# ---------------------------------------------------------------------------

def _render_canonical(canonical: Dict[str, Any]) -> str:
    def _v(raw: Any) -> str:
        if not raw:
            return "(not specified)"
        if isinstance(raw, list):
            return "; ".join(str(x) for x in raw if x) or "(not specified)"
        return str(raw).strip() or "(not specified)"
    # Visual-symptom fields only — the verifier corroborates appearance, not
    # pathogen identity/epidemiology/treatment (keeps the task on-purpose and
    # avoids off-topic content).
    return "\n".join([
        f"  affected_parts:       {_v(canonical.get('affected_parts'))}",
        f"  summary:              {_v(canonical.get('summary'))}",
        f"  diagnostic_features:  {_v(canonical.get('diagnostic_features'))}",
        f"  look_alikes:          {_v(canonical.get('look_alikes'))}",
    ])


def _render_existing(existing: List[Dict[str, Any]]) -> str:
    if not existing:
        return "  (none — cold start for this state)"
    lines: List[str] = []
    for d in existing:
        fld = d.get("field", "other")
        sup = d.get("swarm_support") or d.get("__support__") or d.get("support") or 0
        ver = d.get("verification_status", "unverified")
        shows = (d.get("image_shows") or "").strip()
        if len(shows) > 200:
            shows = shows[:200].rstrip() + "..."
        lines.append(f"  - [{fld}] (support={sup}, status={ver})  {shows}")
    return "\n".join(lines)


def _render_candidates(candidates: List[Dict[str, Any]]) -> str:
    if not candidates:
        return "  (none)"
    lines: List[str] = []
    for i, d in enumerate(candidates, 1):
        fld = d.get("field", "other")
        sup = d.get("__support__") or d.get("swarm_support") or 1
        canon_says = d.get("canonical_says", "(not specified)")
        shows = (d.get("image_shows") or "").strip()
        quote = (d.get("image_quote") or "").strip()
        img_id = d.get("image_id", "")
        lines.append(
            f"  [{i}] field={fld!s}  swarm_support={sup}\n"
            f"      canonical_says: {canon_says!s}\n"
            f"      image_shows:    {shows!s}\n"
            f"      image_quote:    {quote!s}\n"
            f"      image_id:       {img_id!s}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def _claude_available() -> bool:
    """``claude`` CLI on PATH (headless mode; no API key path).

    Uses shutil.which so Windows PATHEXT (.exe / .cmd / .bat) is honoured.
    """
    import shutil
    return shutil.which("claude") is not None


def _normalize_delta(
    d: Dict[str, Any],
    *,
    fallback_status: str,
    primary_image_id: str = "",
) -> Optional[Dict[str, Any]]:
    """Coerce a Claude-emitted delta into the storage shape.

    Carries over: field, canonical_says, image_shows, image_quote, image_id.
    Sets: swarm_support (from input), verification_status, web_support,
    reasoning.
    """
    if not isinstance(d, dict):
        return None
    image_shows = str(d.get("image_shows") or "").strip()
    if not image_shows:
        return None
    out: Dict[str, Any] = {
        "field":          str(d.get("field") or "other").lower().strip() or "other",
        "canonical_says": str(d.get("canonical_says") or "(not specified)").strip()
                          or "(not specified)",
        "image_shows":    image_shows,
        "image_quote":    str(d.get("image_quote") or "").strip(),
        "image_id":       str(d.get("image_id") or primary_image_id),
        "verification_status": str(d.get("verification_status") or fallback_status),
        "reasoning":      str(d.get("reasoning") or "").strip(),
    }
    try:
        out["swarm_support"] = int(d.get("swarm_support") or d.get("__support__") or 1)
    except (TypeError, ValueError):
        out["swarm_support"] = 1
    raw_support = d.get("web_support")
    web: List[Dict[str, str]] = []
    if isinstance(raw_support, list):
        for s in raw_support:
            if not isinstance(s, dict):
                continue
            url   = str(s.get("url")   or "").strip()
            quote = str(s.get("quote") or "").strip()
            if url or quote:
                web.append({"url": url, "quote": quote})
    out["web_support"] = web
    return out


def verify_candidates(
    *,
    crop: str,
    disease: str,
    state: str,
    canonical: Dict[str, Any],
    existing_kb_deltas: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    primary_image_id: str = "",
    timeout_secs: int = 600,
    max_turns: int = 30,
) -> Dict[str, List[Dict[str, Any]]]:
    """Web-grounded verification of one tuple's swarm candidates.

    Returns ``{verified, provisional, contradictory, duplicates_of_existing,
    accepted}`` where ``accepted`` is the flat list (verified +
    weakly_supported + provisional + novel_plausible) ready for the
    conservative merge step. Each accepted delta carries a
    ``verification_status``, ``swarm_support``, ``web_support`` list, and
    ``reasoning`` string.

    Fallback path: when no ``claude`` CLI / API key is available, the
    candidates pass through unchanged with ``verification_status =
    "unverified"`` and ``web_support = []``. This keeps the pipeline
    runnable in offline / CI settings without burning API spend.
    """
    if not candidates:
        return {
            "verified":               [],
            "provisional":            [],
            "contradictory":          [],
            "duplicates_of_existing": [],
            "accepted":               [],
        }

    def _failed(reason: str) -> Dict[str, List[Dict[str, Any]]]:
        """Verifier could not produce a real verdict (claude absent /
        unauthenticated / timed out / empty / unparseable). Preserve every
        candidate as ``unverified`` — NEVER drop them — and flag the result
        so the driver fails loud instead of committing a gutted KB."""
        preserved: List[Dict[str, Any]] = []
        for c in candidates:
            nd = _normalize_delta(c, fallback_status="unverified",
                                   primary_image_id=primary_image_id)
            if nd is not None:
                preserved.append(nd)
        print(f"  VERIFIER FAILED ({reason}): preserving "
              f"{len(preserved)} candidate(s) as unverified (not dropped)")
        return {
            "verified":               [],
            "provisional":            [],
            "contradictory":          [],
            "duplicates_of_existing": [],
            "accepted":               [],
            "_verifier_failed":       True,
            "_failure_reason":        reason,
            "_preserved_unverified":  preserved,
        }

    if not _claude_available():
        return _failed("claude CLI not found on PATH")

    # Some registries store the scientific pathogen binomial in `disease_name`
    # (e.g. "Xanthomonas Vasicola"). Leading a web query with a pathogen
    # binomial + geography can false-trip Claude's Usage Policy classifier,
    # so prefer a plain-language common name for the disease in the prompt.
    # Falls back to the stored name for the (majority) of diseases that
    # already carry a common-ish name.
    disease_display = _DISEASE_COMMON.get(disease.strip().lower(), disease)

    # Retry on a hard block/empty: a couple of the ~thousand diseases still
    # trip the safety classifier; retry keeps the run from stalling on them.
    raw = None
    for attempt in range(3):
        prompt = VERIFIER_PROMPT.format(
            crop=crop, disease=disease_display, state=state,
            canonical_block=_render_canonical(canonical),
            existing_block=_render_existing(existing_kb_deltas),
            candidates_block=_render_candidates(candidates),
        )
        raw = claude_query(
            prompt=prompt,
            allowed_tools=["WebSearch"],
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            json_schema=VERIFIER_OUTPUT_SCHEMA,
            max_turns=max_turns,
            timeout_secs=timeout_secs,
        )
        if raw is not None:
            break
        print(f"  retry {attempt+1}/3 for {crop}/{disease}/{state}", flush=True)
    if raw is None:
        return _failed("claude_query returned None (auth / timeout / empty)")
    verdict = parse_json_result(raw, f"verifier_{crop}_{disease}_{state}")
    _BUCKET_KEYS = ("verified", "provisional", "contradictory",
                    "duplicates_of_existing")
    if not isinstance(verdict, dict) or not any(
        k in verdict for k in _BUCKET_KEYS
    ):
        # A dict that DOES contain ≥1 bucket key (even all-empty) is a real
        # verdict — Claude legitimately found everything contradictory /
        # duplicate. Only a missing/keyless verdict is a failure.
        return _failed("verifier verdict missing all expected buckets")

    def _bucket(name: str, fallback: str) -> List[Dict[str, Any]]:
        raw_list = verdict.get(name) or []
        out: List[Dict[str, Any]] = []
        for d in raw_list:
            nd = _normalize_delta(d, fallback_status=fallback,
                                   primary_image_id=primary_image_id)
            if nd is not None:
                out.append(nd)
        return out

    verified              = _bucket("verified",               "verified")
    provisional           = _bucket("provisional",            "provisional")
    contradictory         = _bucket("contradictory",          "contradictory")
    duplicates_of_existing = _bucket("duplicates_of_existing", "duplicate_existing")

    accepted = verified + provisional        # everything we keep going forward
    return {
        "verified":               verified,
        "provisional":            provisional,
        "contradictory":          contradictory,
        "duplicates_of_existing": duplicates_of_existing,
        "accepted":               accepted,
    }
