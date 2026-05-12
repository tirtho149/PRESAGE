"""
agents/base_agent.py
====================
Base class for the Qwen delta-extraction swarm — paper-faithful routing
edition (PlantSwarm §4 / Algorithm 1, adapted for deltas).

Each call this agent makes returns four things:

    - deltas          : list of {field, canonical_says, image_shows, image_quote}
                        for the canonical fields this agent owns
    - confidence (κ)  : "high" | "medium" | "low"
    - handoff_target  : the agent that should run next, or None to terminate
    - reasoning       : one-line free-text justification (kept in the
                        context buffer so the next agent can see it)

The orchestrator (``plantswarm.delta_pipeline``) overrides the model's
chosen handoff when paper Algorithm 1 dictates a different one:

    κ=low  AND backtrack_count == 0          → MorphologyAgent (regrounding)
    κ=low  AND backtrack_count >= 1          → default forward (loop guard)
    κ=high AND all_specialists_contributed   → DiagnosisAgent (early terminate)
    otherwise                                → model's chosen handoff
"""

from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from utils.vllm_client import VLLMClient


# ---------------------------------------------------------------------------
# Delta field vocabulary
# ---------------------------------------------------------------------------

ALLOWED_DELTA_FIELDS = (
    "lesion_morphology",
    "severity",
    "affected_organs",
    "spread_pattern",
    "diagnostic_features",
    "look_alikes",
    "treatments",
    "type_of_disease",
    "other",
)

CONFIDENCE_LEVELS = ("high", "medium", "low")

# Map each owned delta field → canonical KB key(s) used to render the slice.
_DELTA_FIELD_TO_CANONICAL = {
    "lesion_morphology":   ("summary",),
    "affected_organs":     ("affected_parts",),
    "diagnostic_features": ("diagnostic_features",),
    "spread_pattern":      ("notes",),
    "look_alikes":         ("look_alikes",),
    "type_of_disease":     ("type_of_disease",),
    "severity":            (),
    "treatments":          ("treatments",),
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentDeltaOutput:
    agent_name: str
    deltas: List[Dict[str, str]] = field(default_factory=list)
    confidence: str = "medium"
    handoff_target: Optional[str] = None
    reasoning: str = ""
    raw_text: str = ""           # for debug / trace logging


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _clean(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, list):
        return "; ".join(str(x) for x in s if x is not None and str(x).strip())
    return str(s).strip()


def _validate_delta(d: Any, allowed_fields: set) -> Optional[Dict[str, str]]:
    """Normalize one model-emitted delta into the schema, or None to drop."""
    if not isinstance(d, dict):
        return None
    image_shows = _clean(d.get("image_shows"))
    if not image_shows:
        return None
    fld = _clean(d.get("field")).lower() or "other"
    if fld not in allowed_fields:
        fld = "other"
    return {
        "field":          fld,
        "canonical_says": _clean(d.get("canonical_says")) or "(not specified)",
        "image_shows":    image_shows,
        "image_quote":    _clean(d.get("image_quote")),
    }


def _coerce_confidence(c: Any) -> str:
    s = _clean(c).lower()
    if s in CONFIDENCE_LEVELS:
        return s
    # Word-boundary match so "highly uncertain" doesn't coerce to "high".
    # Only match when the level appears as a whole word.
    tokens = re.findall(r"[a-z]+", s)
    for level in CONFIDENCE_LEVELS:
        if level in tokens:
            return level
    return "medium"


def _coerce_handoff(t: Any, menu: List[str]) -> Optional[str]:
    s = _clean(t)
    if not s or s.lower() in ("none", "null", "terminate"):
        return None
    for name in menu:
        if name.lower() == s.lower():
            return name
    # Substring fallback ("SymptomAgent" inside "next: SymptomAgent")
    for name in menu:
        if name.lower() in s.lower():
            return name
    return None


# ---------------------------------------------------------------------------
# Output JSON parser
# ---------------------------------------------------------------------------

def parse_agent_output(
    text: str,
    owned_fields: List[str],
    handoff_menu: List[str],
) -> Tuple[List[Dict[str, str]], str, Optional[str], str]:
    """Return (deltas, confidence, handoff_target, reasoning).

    Recovers from markdown fences and from stray prose around the JSON.
    Coerces unknown handoff strings to menu members; rejects empty
    image_shows; coerces off-domain field names to 'other'.
    """
    if not text:
        return [], "medium", None, ""

    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)

    obj: Any = None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group())
            except json.JSONDecodeError:
                obj = None

    if not isinstance(obj, dict):
        return [], "medium", None, ""

    allowed_fields = set(owned_fields) | {"other"}
    deltas: List[Dict[str, str]] = []
    for d in obj.get("deltas") or []:
        v = _validate_delta(d, allowed_fields)
        if v is not None:
            deltas.append(v)

    confidence = _coerce_confidence(obj.get("confidence"))
    handoff    = _coerce_handoff(obj.get("handoff_target"), handoff_menu)
    reasoning  = _clean(obj.get("reasoning"))
    return deltas, confidence, handoff, reasoning


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

DELTA_USER_PROMPT = """\
You are looking at one Bugwood Network field photograph of a plant
disease. The canonical knowledge base for this disease is given below.

Crop:    {crop}
Disease: {disease}
State:   {state}

CANONICAL KB (slice for {agent_name} — do NOT re-describe these contents):
{canonical_slice}

You own these delta fields: {owned_fields}

{prior_context}\
Your job has two parts:

(1) DELTAS — Look at the IMAGE and compare it to the canonical KB for
    YOUR owned fields. For each owned field, decide:
      · Does the image show something canonical does NOT capture?
      · Does the image CONTRADICT canonical?
    If yes, emit a delta. If the image confirms canonical exactly for
    that field, do not emit a delta for it. Each delta MUST be supported
    by something visible in this photo. Restating canonical text is
    forbidden.

(2) ROUTING — Report your confidence and pick the next agent:
      · confidence: "high" if your deltas are well-grounded in clear
        visual evidence; "medium" if some are uncertain; "low" if the
        image is ambiguous or you couldn't ground anything.
      · handoff_target: pick ONE of {handoff_menu_str}. Pick
        MorphologyAgent if you want regrounding on low confidence.
        Pick DiagnosisAgent if all specialists have contributed and
        you're confident the picture is complete. Otherwise pick the
        specialist that would add the most.

Output STRICT JSON, no markdown fences, no preamble:
{{
  "deltas": [
    {{
      "field":          "<one of: {owned_fields}>",
      "canonical_says": "<short quote from canonical above on this field, or '(not specified)'>",
      "image_shows":    "<state-specific addition or contradiction — one sentence>",
      "image_quote":    "<one-sentence visual evidence — what you literally see>"
    }}
  ],
  "confidence":     "high" | "medium" | "low",
  "handoff_target": "<one of {handoff_menu_str}>",
  "reasoning":      "<one-line justification for confidence + handoff>"
}}

If the image confirms canonical exactly for every owned field, return
{{"deltas": [], "confidence": "high", "handoff_target": "DiagnosisAgent", "reasoning": "..."}}.
"""


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent(abc.ABC):
    """Subclasses must set:
        - AGENT_NAME       str
        - OWNED_FIELDS     List[str], subset of ALLOWED_DELTA_FIELDS
        - HANDOFF_MENU     List[str], valid next-agent names
        - DEFAULT_FORWARD  str, the natural next agent (used when routing
                           logic doesn't pick something else)
        - SYSTEM_PROMPT    str
    """

    AGENT_NAME: str = "BaseAgent"
    OWNED_FIELDS: List[str] = []
    HANDOFF_MENU: List[str] = []
    DEFAULT_FORWARD: str = "DiagnosisAgent"
    SYSTEM_PROMPT: str = (
        "You are a plant pathology vision agent. Output strict JSON only — "
        "no prose, no markdown, no preamble."
    )

    def __init__(self, client: VLLMClient):
        self.client = client

    # ------------------------------------------------------------------
    # Public call (one routed step)
    # ------------------------------------------------------------------

    def extract_with_routing(
        self,
        *,
        crop: str,
        disease: str,
        state: str,
        canonical: Dict[str, Any],
        image_b64: str,
        prior_context: List["AgentDeltaOutput"],
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AgentDeltaOutput:
        """One stochastic step of the routed swarm.

        ``prior_context`` is the run's context buffer — the deltas /
        confidence / reasoning from every agent that has already run in
        this trace. We render it inline so the model can refine or
        contradict earlier agents' findings.
        """
        canonical_slice = self._format_canonical_slice(canonical)
        prior_block = self._format_prior_context(prior_context)
        owned_list = ", ".join(self.OWNED_FIELDS) or "other"
        menu_str = ", ".join(self.HANDOFF_MENU) or "DiagnosisAgent"

        user_prompt = DELTA_USER_PROMPT.format(
            crop=crop,
            disease=disease,
            state=state,
            agent_name=self.AGENT_NAME,
            canonical_slice=canonical_slice,
            owned_fields=owned_list,
            handoff_menu_str=menu_str,
            prior_context=prior_block,
        )

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": user_prompt},
            ],
        }]

        text, _tokens = self.client.chat(
            messages=messages,
            system_prompt=self.SYSTEM_PROMPT,
            seed=seed,
            temperature=temperature,
        )

        deltas, confidence, handoff, reasoning = parse_agent_output(
            text=text,
            owned_fields=self.OWNED_FIELDS,
            handoff_menu=self.HANDOFF_MENU,
        )

        return AgentDeltaOutput(
            agent_name=self.AGENT_NAME,
            deltas=deltas,
            confidence=confidence,
            handoff_target=handoff,
            reasoning=reasoning,
            raw_text=text,
        )

    # ------------------------------------------------------------------
    # Canonical slice (owned-field view)
    # ------------------------------------------------------------------

    def _format_canonical_slice(self, canonical: Dict[str, Any]) -> str:
        lines: List[str] = []
        if canonical.get("pathogen_scientific_name"):
            lines.append(f"  pathogen: {_clean(canonical['pathogen_scientific_name'])}")
        if (
            canonical.get("type_of_disease")
            and "type_of_disease" not in self.OWNED_FIELDS
        ):
            lines.append(f"  type: {_clean(canonical['type_of_disease'])}")
        for owned in self.OWNED_FIELDS:
            canon_keys = _DELTA_FIELD_TO_CANONICAL.get(owned, ())
            value = ""
            for key in canon_keys:
                raw = canonical.get(key)
                if raw:
                    value = _clean(raw)
                    if value:
                        break
            lines.append(f"  {owned}: {value or '(not specified)'}")
        return "\n".join(lines) if lines else "  (canonical not available)"

    # ------------------------------------------------------------------
    # Prior-agent context (in-trace memory)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_prior_context(prior: List["AgentDeltaOutput"]) -> str:
        """Render prior agents' output so this agent can refine/contradict.

        Empty string when no prior agents have run yet — keeps the prompt
        clean for the entry agent.
        """
        if not prior:
            return ""
        lines: List[str] = ["", "PRIOR AGENT CONTEXT (most recent last):"]
        for step, out in enumerate(prior, 1):
            lines.append(f"  [{step}] {out.agent_name} (confidence={out.confidence})")
            if out.reasoning:
                lines.append(f"      reasoning: {out.reasoning}")
            if out.deltas:
                for d in out.deltas:
                    img = d.get("image_shows", "")
                    if len(img) > 200:
                        img = img[:200].rstrip() + "…"
                    lines.append(f"      delta[{d.get('field','')}]: {img}")
            else:
                lines.append(f"      (no deltas emitted)")
        lines.append("")
        return "\n".join(lines) + "\n"
