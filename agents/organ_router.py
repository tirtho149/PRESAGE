"""
agents/organ_router.py
======================
DR.Arti-style decision tree: detect the organ, then activate ONLY the
deep specialists for that organ.

Each Bugwood photograph shows essentially ONE structure — a leaf shot
is just a leaf, a stem shot is just a stem. Running every agent on
every image is wasteful and vague (the fruit agent on a leaf photo
just says "no fruit visible"). Instead:

    OrganDetectionAgent  (1 visual call: which organ is this?)
        |
        v
    route_for_organ(organ)  ->  the DEEP single-feature specialists
                                 for that organ + always-on
                                 cross-cutters (color / severity /
                                 look-alike / sporulation)
        |
        v
    only that branch fans out (parallel + blackboard round-2) ->
    VisualDiagnosisAgent consolidator -> K-of-N -> verifier -> merge

This is the decision-tree shape DR.Arti.docx describes: a routing root
node, then a deep dive down exactly one branch. The "deep" agents are
the original 24 single-feature specialists (leaf lesion shape, leaf
color, stem pith, ...) — they are reused here, just GATED so only the
relevant ones are ever active. ColorPaletteAgent (the dedicated colour
agent) and the look-alike / severity / sporulation cross-cutters are
in every route because colour, severity, visual confusables, and
pathogen signs can appear on any organ.

Visual-symptoms only — every routed agent is a BaseAgent specialist
that compares the photo to the canonical visual_symptoms KB and emits
nothing about pathogen / type / treatment.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# Import the deep specialists straight from their modules (not via
# agents/__init__) — organ_router is imported BY agents/__init__, so
# going through the package would be circular.
from agents.leaf_agents import (
    LeafLesionShapeAgent, LeafLesionColorAgent, LeafLesionTextureAgent,
    LeafChlorosisAgent, LeafNecrosisAgent, LeafCurlAgent,
    LeafVeinPatternAgent, LeafGeometryAgent,
)
from agents.stem_agents import (
    StemLesionAgent, StemPithAgent, StemSurfaceAgent, StemDiscolorationAgent,
)
from agents.root_agents import RootAgent, CrownCollarAgent
from agents.reproductive_agents import FlowerAgent, FruitAgent
from agents.sign_agents import SporulationAgent
from agents.pattern_agents import (
    WiltingAgent, DefoliationAgent, SpatialPatternAgent,
)
from agents.diagnostic_agents import (
    ConcentricPatternAgent, ColorPaletteAgent,
    LookAlikeCoTAgent, SeverityVisualAgent,
)

if TYPE_CHECKING:
    from utils.vllm_client import VLLMClient


# Cross-cutters that can fire on ANY organ: the dedicated colour
# encoder, visible severity, the look-alike decision-graph, and
# pathogen signs (sporulation/ooze/pustules show up on leaf, stem,
# fruit, root alike).
_ALWAYS_ON: Tuple[type, ...] = (
    ColorPaletteAgent,
    SeverityVisualAgent,
    LookAlikeCoTAgent,
    SporulationAgent,
)

_LEAF: Tuple[type, ...] = (
    LeafLesionShapeAgent, LeafLesionColorAgent, LeafLesionTextureAgent,
    LeafChlorosisAgent, LeafNecrosisAgent, LeafCurlAgent,
    LeafVeinPatternAgent, LeafGeometryAgent,
)
_STEM: Tuple[type, ...] = (
    StemLesionAgent, StemPithAgent, StemSurfaceAgent, StemDiscolorationAgent,
)
_ALL_24: Tuple[type, ...] = (
    _LEAF + _STEM + (RootAgent, CrownCollarAgent, FlowerAgent, FruitAgent,
                     SporulationAgent, WiltingAgent, DefoliationAgent,
                     SpatialPatternAgent, ConcentricPatternAgent,
                     ColorPaletteAgent, LookAlikeCoTAgent, SeverityVisualAgent)
)


def _route(*groups: Tuple[type, ...]) -> Tuple[type, ...]:
    """Concatenate groups + the always-on cross-cutters, de-duped,
    order-stable."""
    seen: set = set()
    out: List[type] = []
    for g in (*groups, _ALWAYS_ON):
        for cls in g:
            if cls not in seen:
                seen.add(cls)
                out.append(cls)
    return tuple(out)


# organ label -> the deep specialists that activate for it.
ORGAN_ROUTES: Dict[str, Tuple[type, ...]] = {
    "leaf":        _route(_LEAF, (ConcentricPatternAgent,)),
    "stem":        _route(_STEM),
    "root":        _route((RootAgent,)),
    "crown":       _route((CrownCollarAgent,)),
    "flower":      _route((FlowerAgent,)),
    "fruit":       _route((FruitAgent, ConcentricPatternAgent)),
    "whole_plant": _route((WiltingAgent, DefoliationAgent, SpatialPatternAgent)),
    # Unsure / multi-organ / scene we can't classify: degrade to full
    # 24-specialist coverage so nothing is missed.
    "other":       _ALL_24,
}

# Synonyms the detector might emit, mapped to a canonical organ key.
_ORGAN_ALIASES: Dict[str, str] = {
    "leaf": "leaf", "leaves": "leaf", "foliage": "leaf", "leaflet": "leaf",
    "stem": "stem", "stalk": "stem", "petiole": "stem", "shoot": "stem",
    "branch": "stem", "twig": "stem", "cane": "stem", "vine": "stem",
    "root": "root", "roots": "root", "taproot": "root",
    "crown": "crown", "collar": "crown", "crown_collar": "crown",
    "soil_line": "crown", "base": "crown",
    "flower": "flower", "blossom": "flower", "bloom": "flower",
    "inflorescence": "flower",
    "fruit": "fruit", "pod": "fruit", "berry": "fruit", "grain": "fruit",
    "kernel": "fruit", "seed": "fruit", "boll": "fruit", "tuber": "fruit",
    "whole_plant": "whole_plant", "whole plant": "whole_plant",
    "plant": "whole_plant", "canopy": "whole_plant", "field": "whole_plant",
    "scene": "whole_plant",
}

VALID_ORGANS: Tuple[str, ...] = (
    "leaf", "stem", "root", "crown", "flower", "fruit", "whole_plant",
)


def normalize_organ(raw: Any) -> str:
    """Map a free-text organ guess to a canonical route key.
    Anything unrecognized -> 'other' (full-coverage fallback)."""
    s = str(raw or "").strip().lower()
    if s in ORGAN_ROUTES:
        return s
    if s in _ORGAN_ALIASES:
        return _ORGAN_ALIASES[s]
    for key in VALID_ORGANS:
        if key in s:
            return key
    for alias, key in _ORGAN_ALIASES.items():
        if alias in s:
            return key
    return "other"


# Routed roster is UNCAPPED by default: after the (fixed)
# organ-detection call, EVERY deep specialist for the detected organ
# fans out — combined with the 2-round blackboard this is the real
# swarm (more agents interacting = richer stigmergy). An optional
# soft cap is available via PATHOME_MAX_ROUTED_AGENTS (0 / unset =
# no cap); if set > 0 the first N classes in route order are kept
# (organ-specific deep specialists first).
import os as _os

MAX_ROUTED_AGENTS: int = max(0, int(_os.environ.get(
    "PATHOME_MAX_ROUTED_AGENTS", "0")))


def route_for_organ(organ: str) -> Tuple[type, ...]:
    """The deep specialist classes to activate for a detected organ.
    Full organ route by default; truncated to the first
    ``MAX_ROUTED_AGENTS`` only if that env cap is set > 0."""
    full = ORGAN_ROUTES.get(normalize_organ(organ), ORGAN_ROUTES["other"])
    return full[:MAX_ROUTED_AGENTS] if MAX_ROUTED_AGENTS > 0 else full


class OrganDetectionAgent:
    """Root of the decision tree. One visual call: which single organ
    dominates this photograph? Pure visual triage — no canonical KB,
    no delta emission, no disease/pathogen claims.
    """

    AGENT_NAME = "OrganDetectionAgent"
    SYSTEM_PROMPT = (
        "You are OrganDetectionAgent. Look at ONE plant photograph and "
        "decide which SINGLE plant structure dominates the frame. You "
        "do not diagnose anything, you do not name diseases — you only "
        "route. Output strict JSON only."
    )

    _PROMPT = """\
Look at this plant photograph. Which SINGLE structure dominates the
frame? Pick exactly one:

  leaf        — leaves / foliage / a leaflet is the main subject
  stem        — stem / stalk / petiole / branch / cane / vine
  root        — exposed roots / taproot
  crown       — crown / collar / the stem-meets-soil zone
  flower      — flower / blossom / inflorescence
  fruit       — fruit / pod / berry / grain / tuber
  whole_plant — a whole plant or a field/canopy scene (no single
                organ dominates)
  other       — cannot tell, or genuinely mixed with no dominant organ

Choose by what is IN FOCUS and fills most of the frame. A leaf with a
blurred stem behind it is "leaf". A whole canopy shot is "whole_plant".

Output STRICT JSON, no markdown, no preamble:
{"organ": "<one of the 8 labels>",
 "confidence": "high|medium|low",
 "why": "<one short clause: what fills the frame>"}
"""

    def __init__(self, client: "VLLMClient"):
        self.client = client

    def detect(
        self,
        image_data_url: str,
        *,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, str]:
        """Return {'organ','confidence','why'}. On any failure ->
        organ='other' so the pipeline degrades to full coverage."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text",      "text":      self._PROMPT},
            ],
        }]
        try:
            text, _ = self.client.chat(
                messages=messages, system_prompt=self.SYSTEM_PROMPT,
                seed=seed, temperature=temperature,
            )
        except Exception as e:  # noqa: BLE001
            return {"organ": "other", "confidence": "low",
                    "why": f"detector error: {type(e).__name__}: {e}"}

        obj = _loads(text)
        organ = normalize_organ(obj.get("organ"))
        conf = str(obj.get("confidence") or "").strip().lower()
        if conf not in ("high", "medium", "low"):
            conf = "medium"
        return {
            "organ": organ,
            "confidence": conf,
            "why": str(obj.get("why") or "")[:200],
        }


def _loads(text: str) -> Dict[str, Any]:
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
