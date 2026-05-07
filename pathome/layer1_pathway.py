"""
pathome/layer1_pathway.py
=========================
Layer 1: Mechanistic infection pathway per pathogen genus (crop-agnostic).
Paper §6.1.

For each pathogen the layer encodes the causal cascade from entry to symptom.
This is the source of cross-crop transferability — enzymatic mechanisms are
crop-agnostic, so an agent that understands Step 4 can diagnose any host.

Each ``PathwayStep`` records:
  - ``stage``: human-readable cascade stage (e.g. "spore germination")
  - ``trigger_conditions``: env/host conditions required for the step
  - ``visual_features_increase``: features that get more prominent at this step
  - ``visual_features_decrease``: features that fade
  - ``causal_explanation``: WHY a feature appears (e.g. "radial pectinase
    diffusion produces circular lesion")
  - ``epistemic_implication``: what an OBSERVE agent should do at this step
    (e.g. "Step 2 features → high eps_t, more evidence helps";
    "Step 4 orange sporulation → immediate early termination")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PathwayStep:
    stage: str
    trigger_conditions: List[str] = field(default_factory=list)
    visual_features_increase: List[str] = field(default_factory=list)
    visual_features_decrease: List[str] = field(default_factory=list)
    causal_explanation: str = ""
    epistemic_implication: str = ""


@dataclass
class MechanisticPathway:
    pathogen_genus: str             # e.g. "Colletotrichum", "Phytophthora"
    pathogen_class: str             # T2 label: "Fungal" | "Bacterial" | ...
    steps: List[PathwayStep] = field(default_factory=list)
    notes: str = ""

    def step_at(self, idx: int) -> Optional[PathwayStep]:
        return self.steps[idx] if 0 <= idx < len(self.steps) else None


# ---------------------------------------------------------------------------
# Worked examples — full coverage is deferred to the migrated paper data
# ---------------------------------------------------------------------------

_BUILTIN: Dict[str, MechanisticPathway] = {}


def _register(pw: MechanisticPathway) -> None:
    _BUILTIN[pw.pathogen_genus.lower()] = pw


_register(MechanisticPathway(
    pathogen_genus="Colletotrichum",
    pathogen_class="Fungal",
    steps=[
        PathwayStep(
            stage="appressorium formation",
            trigger_conditions=["leaf wetness >6h", "T 22-28C"],
            visual_features_increase=["small dark spots"],
            causal_explanation="Melanised appressorium punctures cuticle; visible as pinhead spot.",
            epistemic_implication="Step 1 features alone → high eps_t; more evidence helps.",
        ),
        PathwayStep(
            stage="cuticle penetration + biotrophic phase",
            trigger_conditions=["host susceptibility"],
            visual_features_increase=["water-soaked margins"],
            causal_explanation="Biotrophic invasion of epidermis; chlorosis around penetration site.",
            epistemic_implication="Confusable with bacterial early lesions → backtrack to MorphologyAgent.",
        ),
        PathwayStep(
            stage="necrotrophic switch + radial pectinase diffusion",
            visual_features_increase=["circular sunken lesion"],
            visual_features_decrease=["leaf gloss"],
            causal_explanation="Pectinase degrades middle lamella radially → circular sunken lesion.",
            epistemic_implication="Circular shape + sunken centre → diagnostic, low eps_t.",
        ),
        PathwayStep(
            stage="sporulation (acervuli)",
            visual_features_increase=["orange/salmon spore masses"],
            causal_explanation="Carotenoid conidia in acervuli → orange dots on lesion centre.",
            epistemic_implication="Definitive: short-circuit to DiagnosisAgent (early termination).",
        ),
    ],
    notes="Paper §6.1 — same enzymatic cascade on mango, soybean, strawberry.",
))

_register(MechanisticPathway(
    pathogen_genus="Phytophthora",
    pathogen_class="Oomycete",
    steps=[
        PathwayStep(
            stage="zoospore release",
            trigger_conditions=["free water on leaf", "T 12-22C"],
            visual_features_increase=["water-soaked patches"],
            causal_explanation="Motile zoospores swim to stomata in surface water.",
            epistemic_implication="Wet morphology pattern → SeverityAgent for stage assessment.",
        ),
        PathwayStep(
            stage="lesion expansion",
            visual_features_increase=["irregular dark blotches", "purple-brown margins"],
            causal_explanation="Coenocytic mycelium invades intercellularly; rapid spread.",
            epistemic_implication="Cool humid conditions amplify Layer-3 prior weight.",
        ),
        PathwayStep(
            stage="sporangium production",
            visual_features_increase=["white downy growth on underside"],
            causal_explanation="Sporangiophores emerge through stomata, lemon-shaped sporangia.",
            epistemic_implication="Underside white fluff at lesion margin → diagnostic.",
        ),
    ],
    notes="P. infestans on tomato/potato; P. capsici on cucurbits.",
))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def get_builtin(genus: str) -> Optional[MechanisticPathway]:
    return _BUILTIN.get(genus.lower())


def all_builtin() -> List[MechanisticPathway]:
    return list(_BUILTIN.values())


def save(pathways: List[MechanisticPathway], path: str) -> None:
    """Serialise to JSON for human editing."""
    payload = []
    for pw in pathways:
        payload.append({
            "pathogen_genus": pw.pathogen_genus,
            "pathogen_class": pw.pathogen_class,
            "notes": pw.notes,
            "steps": [step.__dict__ for step in pw.steps],
        })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load(path: str) -> List[MechanisticPathway]:
    with open(path) as f:
        payload = json.load(f)
    out: List[MechanisticPathway] = []
    for entry in payload:
        steps = [PathwayStep(**s) for s in entry.get("steps", [])]
        out.append(MechanisticPathway(
            pathogen_genus=entry["pathogen_genus"],
            pathogen_class=entry.get("pathogen_class", ""),
            notes=entry.get("notes", ""),
            steps=steps,
        ))
    return out
