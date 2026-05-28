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
    "You are a plant pathology evidence reviewer. You receive candidate "
    "regional observations from a vision swarm and validate them against "
    "the web (extension factsheets, APS / CABI references, peer-reviewed "
    "literature). You are conservative: you reward strong external evidence "
    "and penalise fabrication. Output strict JSON only — no prose, no "
    "markdown."
)


VERIFIER_PROMPT = """\
You are validating regional plant disease observations against agronomic
literature. The vision swarm has emitted candidate observations from a
single Bugwood field photograph. Your job is to use web search to
verify each candidate against external evidence, then return a
structured verdict.

CROP:    {crop}
DISEASE: {disease}
STATE:   {state}

CANONICAL KB (already established for this disease; treat as background):
{canonical_block}

EXISTING REGIONAL OBSERVATIONS for {state} (already in the KB; preserve,
do NOT re-emit):
{existing_block}

CANDIDATE OBSERVATIONS from the Qwen swarm (each with a swarm_support
count = how many of N stochastic runs proposed it):
{candidates_block}

YOUR TASK
=========
Use WebSearch to look up extension-service pages, APS / CABI
references, peer-reviewed literature, and pathology resources for
{crop} :: {disease} (and where relevant, the specific state context).
Search for evidence supporting (or contradicting) each candidate
observation.

For each candidate, return:

  field            one of: lesion_morphology, severity, affected_organs,
                            spread_pattern, diagnostic_features,
                            look_alikes, treatments, type_of_disease, other
  canonical_says   short quote from CANONICAL KB on this field, or
                    "(not specified)"
  image_shows      one-sentence state-specific addition or contradiction
                    (carried from the candidate)
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
    return "\n".join([
        f"  pathogen:             {_v(canonical.get('pathogen_scientific_name'))}",
        f"  type_of_disease:      {_v(canonical.get('type_of_disease'))}",
        f"  affected_parts:       {_v(canonical.get('affected_parts'))}",
        f"  summary:              {_v(canonical.get('summary'))}",
        f"  diagnostic_features:  {_v(canonical.get('diagnostic_features'))}",
        f"  look_alikes:          {_v(canonical.get('look_alikes'))}",
        f"  treatments:           {_v(canonical.get('treatments'))}",
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

    prompt = VERIFIER_PROMPT.format(
        crop=crop, disease=disease, state=state,
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
