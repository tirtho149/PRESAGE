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


_register(MechanisticPathway(
    pathogen_genus="Xanthomonas",
    pathogen_class="Bacterial",
    steps=[
        PathwayStep(
            stage="hydathode / wound entry",
            trigger_conditions=["leaf wetness", "wounding", "T 25-30C"],
            visual_features_increase=["water-soaked angular spots"],
            causal_explanation="Bacteria enter through hydathodes, multiply in mesophyll.",
            epistemic_implication="Angular margin (vein-bounded) → bacterial; high eps_t until margin confirmed.",
        ),
        PathwayStep(
            stage="vascular invasion (if present)",
            visual_features_increase=["yellow halo", "vein chlorosis"],
            causal_explanation="Type III secretion suppresses host immunity; vascular spread.",
            epistemic_implication="Halo around angular spot → diagnostic for many xanthomonads.",
        ),
        PathwayStep(
            stage="bacterial ooze",
            visual_features_increase=["sticky exudate on humid days"],
            causal_explanation="Bacterial exudate at lesion in high humidity.",
            epistemic_implication="Ooze test (squeeze + water) → SymptomAgent confirms bacterial.",
        ),
    ],
    notes="Bacterial leaf spot/blight family. Angular margin is the single most informative feature.",
))


_register(MechanisticPathway(
    pathogen_genus="Botrytis",
    pathogen_class="Fungal",
    steps=[
        PathwayStep(
            stage="conidial germination",
            trigger_conditions=["high humidity", "senescing tissue", "T 18-23C"],
            visual_features_increase=["pale necrotic spots on petals/fruit"],
            causal_explanation="Conidia germinate on weakened tissue; require free moisture.",
            epistemic_implication="Senescing-tissue affinity → SeverityAgent for stage cue.",
        ),
        PathwayStep(
            stage="grey mould production",
            visual_features_increase=["grey fuzzy sporulation"],
            causal_explanation="Massed conidiophores produce diagnostic grey mould.",
            epistemic_implication="Grey velvet on lesion → terminate routing immediately.",
        ),
        PathwayStep(
            stage="sclerotium formation (late)",
            visual_features_increase=["small black sclerotia"],
            causal_explanation="Sclerotia are survival structures; appear late.",
            epistemic_implication="Sclerotia confirm Botrytis even when grey mould absent.",
        ),
    ],
    notes="B. cinerea grey mould — broad host range, temperate humid.",
))


_register(MechanisticPathway(
    pathogen_genus="Puccinia",
    pathogen_class="Fungal",
    steps=[
        PathwayStep(
            stage="urediniospore landing",
            trigger_conditions=["6-10h leaf wetness", "T 10-20C"],
            visual_features_increase=["yellow flecks"],
            causal_explanation="Obligate biotroph; needs wet leaf + cool conditions.",
            epistemic_implication="Yellow flecks alone → high eps_t; confirm with pustule emergence.",
        ),
        PathwayStep(
            stage="pustule eruption",
            visual_features_increase=["orange/yellow/brown rust pustules"],
            causal_explanation="Uredinia rupture epidermis releasing urediniospores.",
            epistemic_implication="Powdery pustule colour → diagnostic for rust species.",
        ),
        PathwayStep(
            stage="telium formation (late season)",
            visual_features_increase=["black teliospores"],
            causal_explanation="Overwintering teliospores form on senescing tissue.",
            epistemic_implication="Black telia confirm rust even when uredinia have abscised.",
        ),
    ],
    notes="P. striiformis on wheat; P. graminis cereal stem rust.",
))


_register(MechanisticPathway(
    pathogen_genus="Alternaria",
    pathogen_class="Fungal",
    steps=[
        PathwayStep(
            stage="conidial germination + cuticle penetration",
            trigger_conditions=["alternating wet-dry", "T 20-30C"],
            visual_features_increase=["small dark spots"],
            causal_explanation="Direct cuticle penetration via appressoria.",
            epistemic_implication="Early indistinguishable from Anthracnose — backtrack required.",
        ),
        PathwayStep(
            stage="lesion expansion with concentric rings",
            visual_features_increase=["dark concentric rings (target spot)"],
            causal_explanation="Cyclical sporulation produces target-pattern rings.",
            epistemic_implication="Concentric rings → diagnostic for Alternaria.",
        ),
        PathwayStep(
            stage="sporulation",
            visual_features_increase=["dark velvety sporulation"],
            causal_explanation="Beak-shaped conidia produced abundantly on lesion.",
            epistemic_implication="Dark velvet plus rings → terminate routing.",
        ),
    ],
    notes="A. solani early blight; A. brassicicola crucifer leaf spot.",
))


_register(MechanisticPathway(
    pathogen_genus="Fusarium",
    pathogen_class="Fungal",
    steps=[
        PathwayStep(
            stage="root / crown infection",
            trigger_conditions=["soil-borne inoculum", "host stress"],
            visual_features_increase=["lower-leaf yellowing"],
            causal_explanation="Vascular wilt — colonisation of xylem from root entry.",
            epistemic_implication="Wilt without leaf-spot pattern → SeverityAgent + SymptomAgent.",
        ),
        PathwayStep(
            stage="vascular discolouration",
            visual_features_increase=["brown vascular streaks in stem cross-section"],
            causal_explanation="Tylose formation + fungal blockage → brown streaks.",
            epistemic_implication="Stem cross-section browning → diagnostic.",
        ),
        PathwayStep(
            stage="aerial sporulation (late)",
            visual_features_increase=["pink/orange sporulation on stem"],
            causal_explanation="Macroconidia + sporodochia at advanced stage.",
            epistemic_implication="Pink sporulation on dead tissue confirms Fusarium.",
        ),
    ],
    notes="F. oxysporum vascular wilt across many hosts.",
))


_register(MechanisticPathway(
    pathogen_genus="Cercospora",
    pathogen_class="Fungal",
    steps=[
        PathwayStep(
            stage="conidial deposition",
            trigger_conditions=["leaf wetness", "T 25-30C"],
            visual_features_increase=["small chlorotic spots"],
            causal_explanation="Conidia germinate on leaf surface with free moisture.",
            epistemic_implication="Generic chlorotic spotting → backtrack needed.",
        ),
        PathwayStep(
            stage="leaf spot with grey center / dark border",
            visual_features_increase=["frogeye lesion (grey center, dark margin)"],
            causal_explanation="Cercosporin photo-toxin produces grey necrotic centre.",
            epistemic_implication="Frogeye pattern → diagnostic for Cercospora.",
        ),
    ],
    notes="C. zeae-maydis on corn; C. arachidicola on peanut.",
))


_register(MechanisticPathway(
    pathogen_genus="Erysiphe",
    pathogen_class="Fungal",
    steps=[
        PathwayStep(
            stage="conidial germination on leaf surface",
            trigger_conditions=["high humidity, no free water", "T 20-27C"],
            visual_features_increase=["white powdery spots on leaf surface"],
            causal_explanation="Obligate biotroph; superficial mycelium with haustoria.",
            epistemic_implication="White powder on upper leaf → diagnostic for powdery mildew.",
        ),
        PathwayStep(
            stage="cleistothecia formation (late)",
            visual_features_increase=["small black fruiting bodies"],
            causal_explanation="Sexual structures appear at season end.",
            epistemic_implication="Cleistothecia confirm species when conidia abscised.",
        ),
    ],
    notes="Powdery mildew family; many crops.",
))


_register(MechanisticPathway(
    pathogen_genus="TomatoMosaic",
    pathogen_class="Viral",
    steps=[
        PathwayStep(
            stage="virion entry via wounding",
            trigger_conditions=["mechanical contact", "vector pressure"],
            visual_features_increase=["mosaic mottle on young leaves"],
            causal_explanation="Tobamovirus particles enter through micro-wounds.",
            epistemic_implication="Mosaic pattern, NO sporulation → viral; eps_t low.",
        ),
        PathwayStep(
            stage="systemic spread",
            visual_features_increase=["leaf curling, stunting", "vein clearing"],
            causal_explanation="Phloem-borne systemic infection.",
            epistemic_implication="Stunting + mosaic + no fungal/bacterial signs → terminate.",
        ),
    ],
    notes="ToMV / TMV on Solanaceae. Mosaic complex includes many viruses.",
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
