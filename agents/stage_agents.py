"""
agents/stage_agents.py
======================
DR.Arti.docx — five-stage look-alike decision-graph, as a CoT chain.

DR.Arti.docx is not a feature-extraction spec. It is a set of pairwise
look-alike discrimination chains-of-thought (SDS vs BSR, IDC vs SCN,
Palmer vs waterhemp, corn rootworm vs cucumber beetle). Every one of
them walks the SAME five ordered stages:

  1. Context / timing / site-history priors        -> "Lean X"
  2. Gross / foliar symptoms                        -> continue if ambiguous
  3. THE decisive fork (split the stem, dig roots,
     check leg color, petiole-vs-blade length)      -> diagnostic
  4. Supporting / corroborating evidence            -> adjust confidence
  5. State final reasoning  OR  explicit uncertainty
     + recommend a follow-up test

This module implements that chain for the disease pipeline. The five
stage agents run **sequentially** (not as a parallel ensemble): each
stage sees the canonical KB, the field photograph, and every prior
stage's structured output. The candidates being discriminated are the
canonical disease vs the canonical ``look_alikes`` list from
``final_registry.json`` (Phase 0 KB). When ``look_alikes`` is empty the
chain degrades gracefully to "image vs canonical" discrepancy capture —
it still emits deltas.

Output contract (unchanged downstream): only :class:`VerdictStage`
emits the schema-compliant ``deltas`` (field / canonical_says /
image_shows / image_quote) consumed by the K-of-N agreement filter,
the verifier, and the conservative merge. Stages 1-4 carry their
findings forward in ``reasoning`` + ``raw_text`` and emit no deltas.
``VerdictStage`` additionally records a ``look_alike_verdict`` block
(canonical | look_alike:<name> | ambiguous + recommended follow-up)
that the pipeline stashes into ``__swarm_meta__`` for traceability;
unknown keys are ignored by the existing delta plumbing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agents.base_agent import (
    ALLOWED_DELTA_FIELDS,
    AgentDeltaOutput,
    BaseAgent,
    _clean,
    parse_agent_output,
)

if TYPE_CHECKING:
    from utils.vllm_client import VLLMClient


# Delta-bearing fields VerdictStage may emit (everything except the
# "other" catch-all, which parse_agent_output re-adds anyway).
_VERDICT_FIELDS: List[str] = [f for f in ALLOWED_DELTA_FIELDS if f != "other"]


# ---------------------------------------------------------------------------
# Shared rendering helpers
# ---------------------------------------------------------------------------

def _format_candidates(canonical: Dict[str, Any], disease: str) -> str:
    """The discrimination set: canonical disease vs its look-alikes."""
    looks = canonical.get("look_alikes")
    if isinstance(looks, str):
        looks = [looks]
    looks = [str(x).strip() for x in (looks or []) if str(x).strip()]
    lines = [f"  C0 (canonical): {disease}"]
    if looks:
        for i, la in enumerate(looks, 1):
            lines.append(f"  L{i} (look-alike): {la}")
    else:
        lines.append("  (no canonical look_alikes listed — discriminate "
                      "image vs canonical only)")
    return "\n".join(lines)


def _format_canonical(canonical: Dict[str, Any]) -> str:
    lines: List[str] = []
    for k in ("pathogen_scientific_name", "type_of_disease", "summary",
              "diagnostic_features", "look_alikes", "affected_parts"):
        v = canonical.get(k)
        if v:
            lines.append(f"  {k}: {_clean(v)}")
    return "\n".join(lines) if lines else "  (canonical KB unavailable)"


def _format_existing(existing_kb_deltas: List[Dict[str, Any]], state: str) -> str:
    if not existing_kb_deltas:
        return ""
    lines = [f"\nPRIOR image-grounded observations for {state} "
             f"(do NOT restate — only add or contradict):"]
    for d in existing_kb_deltas[:40]:
        lines.append(f"  [{d.get('field','')}] {d.get('image_shows','')}")
    return "\n".join(lines) + "\n"


def _format_prior_stages(prior: List[AgentDeltaOutput]) -> str:
    if not prior:
        return "  (this is the first stage — no prior stage output)"
    lines: List[str] = []
    for o in prior:
        lines.append(f"  >> {o.agent_name} (confidence={o.confidence})")
        if o.reasoning:
            lines.append(f"     finding: {o.reasoning}")
        # raw_text carries the stage's full structured JSON; include it
        # verbatim so the next stage can reason over the detail, not
        # just the one-line summary.
        if o.raw_text:
            snippet = o.raw_text.strip().replace("\n", " ")
            lines.append(f"     detail: {snippet[:900]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage base
# ---------------------------------------------------------------------------

class StageAgent(BaseAgent):
    """One stage of the DR.Arti decision-graph chain.

    Subclasses set ``AGENT_NAME``, ``STAGE_INDEX`` (1-5), ``SYSTEM_PROMPT``
    and a ``STAGE_TASK`` block (the stage-specific CoT instructions).
    The chain runner calls :meth:`run_stage` in order, threading each
    stage's :class:`AgentDeltaOutput` forward via ``prior_stages``.
    """

    STAGE_INDEX: int = 0
    STAGE_TASK: str = ""
    # Stages 1-4 own no delta fields; VerdictStage overrides.
    OWNED_FIELDS: List[str] = []

    _PROMPT = """\
You are walking a structured plant-disease look-alike decision graph
(the DR.Arti reference). You are at STAGE {stage_index}/5: {agent_name}.

ONE Bugwood field photograph is attached.

Crop:    {crop}
Disease: {disease}
State:   {state}

CANONICAL KB (Phase 0; text-grounded — do NOT merely restate):
{canonical_block}

DISCRIMINATION SET (decide among these — the canonical disease vs its
documented look-alikes):
{candidates_block}
{existing_block}
PRIOR STAGE OUTPUTS (earlier steps of THIS decision graph — build on
them, do not contradict without explicit reason):
{prior_block}

STAGE {stage_index} TASK — {agent_name}:
{stage_task}

Output STRICT JSON, no markdown fences, no preamble, matching the
schema in the STAGE TASK above.
"""

    def run_stage(
        self,
        *,
        crop: str,
        disease: str,
        state: str,
        canonical: Dict[str, Any],
        image_data_url: str,
        existing_kb_deltas: Optional[List[Dict[str, Any]]] = None,
        prior_stages: Optional[List[AgentDeltaOutput]] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AgentDeltaOutput:
        prompt = self._PROMPT.format(
            stage_index=self.STAGE_INDEX,
            agent_name=self.AGENT_NAME,
            crop=crop, disease=disease, state=state,
            canonical_block=_format_canonical(canonical),
            candidates_block=_format_candidates(canonical, disease),
            existing_block=_format_existing(existing_kb_deltas or [], state),
            prior_block=_format_prior_stages(prior_stages or []),
            stage_task=self.STAGE_TASK,
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text",      "text":      prompt},
            ],
        }]
        try:
            text, _tokens = self.client.chat(
                messages=messages, system_prompt=self.SYSTEM_PROMPT,
                seed=seed, temperature=temperature,
            )
        except Exception as e:  # noqa: BLE001 — stage failure must not kill the pass
            return AgentDeltaOutput(
                agent_name=self.AGENT_NAME,
                confidence="low",
                reasoning=f"stage error: {type(e).__name__}: {e}",
                round_idx=self.STAGE_INDEX,
            )
        return self._parse(text)

    # Stages 1-4 parse a free-form structured block; VerdictStage
    # overrides to also extract schema deltas + the verdict record.
    def _parse(self, text: str) -> AgentDeltaOutput:
        obj = _loads(text)
        reasoning = _clean(obj.get("reasoning")) if obj else ""
        confidence = _clean(obj.get("confidence")).lower() if obj else "medium"
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"
        return AgentDeltaOutput(
            agent_name=self.AGENT_NAME,
            deltas=[],
            confidence=confidence,
            reasoning=reasoning or (text.strip()[:300] if text else ""),
            raw_text=text or "",
            round_idx=self.STAGE_INDEX,
        )

    # Back-compat: the legacy specialist path calls extract_deltas. A
    # stage agent should never be invoked that way, but fail safe.
    def extract_deltas(self, *args: Any, **kwargs: Any) -> AgentDeltaOutput:  # type: ignore[override]
        return AgentDeltaOutput(
            agent_name=self.AGENT_NAME,
            reasoning="StageAgent must be run via run_stage(), not "
                      "extract_deltas(); returning empty.",
            round_idx=self.STAGE_INDEX,
        )


def _loads(text: str) -> Dict[str, Any]:
    """Tolerant JSON object pull (fenced / embedded / strict)."""
    if not text:
        return {}
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return {}
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            return {}
    return obj if isinstance(obj, dict) else {}


# ---------------------------------------------------------------------------
# Stage 1 — Context / timing / site-history priors
# ---------------------------------------------------------------------------

class ContextStage(StageAgent):
    AGENT_NAME = "ContextStage"
    STAGE_INDEX = 1
    SYSTEM_PROMPT = (
        "You are the CONTEXT stage of a plant-disease look-alike "
        "decision graph. You reason about timing, growth stage, region, "
        "and site/field-history priors — NOT fine lesion morphology yet. "
        "Output strict JSON only."
    )
    STAGE_TASK = """\
Mirror DR.Arti Stage 1 ("Check growth stage and field history → Lean X").
From the canonical KB, the U.S. state, and any phenology/scene context
visible in the photo (crop stage, season cues, planting pattern,
soil/site hints), assign a PRIOR LEAN over the discrimination set —
which candidates the context favors and which it argues against. Do
NOT decide yet; this only sets priors.

JSON schema:
{
  "stage": "context",
  "leans": [
    {"candidate": "C0|L1|L2|...", "name": "<disease name>",
     "lean": "favored|neutral|against",
     "why": "<one sentence grounded in KB/state/visible context>"}
  ],
  "candidates_remaining": ["C0", "L1", ...],
  "confidence": "high|medium|low",
  "reasoning": "<one-line: context favors X over Y because Z>"
}"""


# ---------------------------------------------------------------------------
# Stage 2 — Gross / foliar symptoms (continue if ambiguous)
# ---------------------------------------------------------------------------

class GrossSymptomStage(StageAgent):
    AGENT_NAME = "GrossSymptomStage"
    STAGE_INDEX = 2
    SYSTEM_PROMPT = (
        "You are the GROSS-SYMPTOM stage of a look-alike decision graph. "
        "You describe the dominant whole-plant / foliar symptom visible "
        "in the photo and narrow the candidate set, explicitly allowing "
        "'still ambiguous, continue'. Output strict JSON only."
    )
    STAGE_TASK = """\
Mirror DR.Arti Stage 2 ("Evaluate foliar symptoms … both possible,
continue"). Describe the DOMINANT gross/foliar symptom actually visible
(distribution, chlorosis/necrosis pattern, wilting topology, canopy
layer). For EACH candidate still in contention from Stage 1, state
whether this gross symptom is consistent, inconsistent, or
non-discriminating. It is correct and expected to conclude "still
ambiguous — continue to the decisive fork".

JSON schema:
{
  "stage": "gross_symptoms",
  "dominant_symptom": "<one sentence, image-grounded>",
  "per_candidate": [
    {"candidate": "C0|L1|...", "name": "<name>",
     "consistency": "consistent|inconsistent|non_discriminating",
     "why": "<one sentence>"}
  ],
  "candidates_remaining": ["C0", "L1", ...],
  "still_ambiguous": true,
  "confidence": "high|medium|low",
  "reasoning": "<one-line gross-symptom read>"
}"""


# ---------------------------------------------------------------------------
# Stage 3 — THE decisive fork
# ---------------------------------------------------------------------------

class DecisiveForkStage(StageAgent):
    AGENT_NAME = "DecisiveForkStage"
    STAGE_INDEX = 3
    SYSTEM_PROMPT = (
        "You are the DECISIVE-FORK stage — the highest-weight step. You "
        "identify the single most diagnostic visual fork the photo can "
        "actually answer (e.g. split-stem pith color, petiole-vs-blade "
        "length, root cysts / blue masses, sporulation type, concentric "
        "rings) and apply it. If the decisive structure is NOT visible "
        "in this photo you must say so explicitly. Output strict JSON only."
    )
    STAGE_TASK = """\
Mirror DR.Arti Stage 3 ("Force the decisive fork"): the doc's examples
are "split the lower stem: white pith → SDS vs brown cardboard pith →
BSR", "dig roots: lemon-shaped cysts → SCN", "leg color → rootworm vs
cucumber beetle", "petiole ≥ blade length → Palmer vs waterhemp".

Choose the ONE decisive fork that (a) best separates the remaining
candidates and (b) is answerable from THIS photograph. State the fork
as a question, answer it from the pixels, and name which candidate the
answer points to. If the decisive structure is not visible (e.g. stem
not split, roots not exposed), set "fork_visible": false and explain —
do NOT guess.

JSON schema:
{
  "stage": "decisive_fork",
  "fork_question": "<the decisive question>",
  "fork_visible": true,
  "observed": "<what the pixels actually show, or 'structure not visible'>",
  "points_to": "C0|L1|...|none",
  "points_to_name": "<disease name or ''>",
  "candidates_remaining": ["C0", "L1", ...],
  "confidence": "high|medium|low",
  "reasoning": "<one-line: fork F observed as O, therefore lean toward N>"
}"""


# ---------------------------------------------------------------------------
# Stage 4 — Supporting / corroborating evidence
# ---------------------------------------------------------------------------

class SupportingEvidenceStage(StageAgent):
    AGENT_NAME = "SupportingEvidenceStage"
    STAGE_INDEX = 4
    SYSTEM_PROMPT = (
        "You are the SUPPORTING-EVIDENCE stage. You list corroborating, "
        "NON-decisive features that strengthen or weaken the Stage-3 "
        "lean. You never overturn the decisive fork — you only adjust "
        "confidence. Output strict JSON only."
    )
    STAGE_TASK = """\
Mirror DR.Arti Stage 4 ("Check roots and crown as SUPPORTING evidence").
Given the Stage-3 lean, list secondary visual features in the photo
that corroborate or weaken it (color palette, secondary lesions,
pathogen signs, severity, spatial pattern). These adjust confidence
only — if they appear to contradict the decisive fork, flag the
conflict for the verdict stage rather than overturning the fork here.

JSON schema:
{
  "stage": "supporting_evidence",
  "supports": [
    {"feature": "<short>", "for_candidate": "C0|L1|...",
     "strength": "strong|moderate|weak",
     "image_quote": "<one-sentence pixel evidence>"}
  ],
  "conflicts_with_fork": "<describe any contradiction, or ''>",
  "confidence": "high|medium|low",
  "reasoning": "<one-line: supporting evidence net-strengthens/weakens lean>"
}"""


# ---------------------------------------------------------------------------
# Stage 5 — Verdict (or explicit uncertainty + follow-up) + deltas
# ---------------------------------------------------------------------------

class VerdictStage(StageAgent):
    AGENT_NAME = "VerdictStage"
    STAGE_INDEX = 5
    OWNED_FIELDS = _VERDICT_FIELDS
    SYSTEM_PROMPT = (
        "You are the VERDICT stage. You state the final reasoning of the "
        "decision graph: either the photo supports the canonical "
        "diagnosis, or it better matches a named look-alike, OR it "
        "cannot be distinguished from the photo and a specific "
        "follow-up test is required. You ALSO emit image-grounded KB "
        "deltas. Output strict JSON only."
    )
    STAGE_TASK = """\
Mirror DR.Arti Stage 5 ("State final reasoning" / "Explicit uncertainty
and lab follow-up"). Decide ONE verdict:
  - "canonical"  : the photo supports the canonical disease
  - "look_alike" : the photo better matches a listed look-alike
                   (give its name)
  - "ambiguous"  : the photo cannot decide it — give a CONCRETE
                   recommended follow-up (e.g. "split lower stem and
                   inspect pith", "dig and wash roots for cysts",
                   "lab culture / PCR")

Then emit DELTAS: every image-grounded fact that ADDS to or CONTRADICTS
the canonical KB visual_symptoms for THIS state. A delta is NOT a
restatement of canonical — it is new state-specific visual information
the decision graph surfaced. Confirming canonical exactly => no delta
for that field. Pick "field" from the allowed list.

Allowed delta fields:
{allowed}

JSON schema:
{{
  "stage": "verdict",
  "verdict": "canonical|look_alike|ambiguous",
  "matched_look_alike": "<name if verdict=look_alike, else ''>",
  "recommended_followup": "<concrete test if verdict=ambiguous, else ''>",
  "deltas": [
    {{"field": "<one allowed field>",
      "canonical_says": "<short canonical quote or '(not specified)'>",
      "image_shows": "<state-specific addition/contradiction, one sentence>",
      "image_quote": "<one-sentence pixel evidence>"}}
  ],
  "confidence": "high|medium|low",
  "reasoning": "<final decision-graph sentence, doc-style: 'Because <fork> plus <support> match <X>, I classify this as <X>, not <Y>.'>"
}}"""

    # STAGE_TASK has literal JSON braces -> format the allowed-field
    # list at construction so the chain's .format() call stays simple.
    def __init__(self, client: "VLLMClient"):
        super().__init__(client)
        self.STAGE_TASK = VerdictStage.STAGE_TASK.replace(
            "{allowed}", ", ".join(_VERDICT_FIELDS)
        )

    def _parse(self, text: str) -> AgentDeltaOutput:
        deltas, confidence, reasoning, _xrefs = parse_agent_output(
            text=text, owned_fields=self.OWNED_FIELDS,
        )
        obj = _loads(text)
        verdict = {
            "verdict": _clean(obj.get("verdict")).lower() or "ambiguous",
            "matched_look_alike": _clean(obj.get("matched_look_alike")),
            "recommended_followup": _clean(obj.get("recommended_followup")),
        }
        out = AgentDeltaOutput(
            agent_name=self.AGENT_NAME,
            deltas=deltas,
            confidence=confidence,
            reasoning=reasoning or _clean(obj.get("reasoning")),
            raw_text=text or "",
            round_idx=self.STAGE_INDEX,
        )
        # Attach the look-alike verdict for swarm_meta/trace. Unknown
        # attribute is harmless to the existing delta plumbing.
        out.look_alike_verdict = verdict  # type: ignore[attr-defined]
        return out


# Ordered DR.Arti decision-graph chain. Stage 5 (VerdictStage) replaces
# the legacy separate DiagnosisAgent consolidator.
STAGE_CHAIN = (
    ContextStage,
    GrossSymptomStage,
    DecisiveForkStage,
    SupportingEvidenceStage,
    VerdictStage,
)
