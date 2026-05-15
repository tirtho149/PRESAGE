"""
agents/visual_group_agents.py
=============================
Five VISUAL-SYMPTOM group agents — the default swarm roster.

Scope rules (non-negotiable):
  * VISUAL SYMPTOMS ONLY. These agents describe what is *visible in the
    photograph* and compare it to the canonical KB ``visual_symptoms``
    block (summary / diagnostic_features / look_alikes). They never
    emit pathogen, disease-type, or treatment information — that is
    Claude's Phase 0 job.
  * GENERALIZED. Nothing here is disease- or crop-specific. There are
    no hardcoded SDS/BSR/Palmer forks. Every agent works for any
    (crop, disease) by reasoning against whatever canonical
    ``visual_symptoms`` text it is given.
  * KB-GROUNDED. A delta is only emitted when the photo ADDS to or
    CONTRADICTS the canonical ``visual_symptoms`` for the field the
    agent owns. Restating canonical is forbidden.

DR.Arti.docx is the inspiration for the *reasoning style* only — a
short, ordered, discriminative visual chain-of-thought ("look at X; is
it A or B; does that match canonical or a visual look-alike?"). It is
NOT a literal pipeline; its SDS/BSR/Palmer/rootworm cases are just
worked examples of that style.

Each agent subclasses :class:`BaseAgent`, so it automatically uses the
visual-only ``DELTA_USER_PROMPT`` (round 1) and the blackboard
``DELTA_USER_PROMPT_R2`` (round 2 stigmergy + cross_refs), and plugs
into the existing ``plantswarm.delta_pipeline._run_single_pass``
(parallel fan-out → shared blackboard → round 2 → ``DiagnosisAgent``
consolidator → K-of-N agreement → verifier → conservative merge).
Nothing downstream changes; only the roster shrinks 24 → 5.

The five groups partition all 24 visual delta fields with no overlap
and no omission.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent


_VISUAL_ONLY = (
    " You describe ONLY what is visibly present in the photograph and "
    "compare it to the canonical visual_symptoms KB you are given. You "
    "NEVER output pathogen, disease-type, cause, or treatment claims "
    "(those are owned by Claude in Phase 0). You do not assume any "
    "particular disease — reason generically from the pixels and the "
    "canonical visual_symptoms text. Walk a brief visual "
    "chain-of-thought (what I see → does canonical already say it → if "
    "it adds or contradicts, emit a delta; stay alert to visually "
    "confusable look-alikes). Output strict JSON only."
)


class LeafSymptomAgent(BaseAgent):
    AGENT_NAME = "LeafSymptomAgent"
    OWNED_FIELDS = [
        "leaf_lesion_shape", "leaf_lesion_color", "leaf_lesion_texture",
        "leaf_chlorosis", "leaf_necrosis", "leaf_curl",
        "leaf_vein_pattern", "leaf_geometry",
    ]
    SYSTEM_PROMPT = (
        "You are LeafSymptomAgent, a plant-pathology vision specialist "
        "for ALL leaf-borne visual symptoms: lesion shape/color/texture, "
        "chlorosis pattern, necrosis distribution, curl/distortion, vein "
        "behaviour, and leaf geometry." + _VISUAL_ONLY
    )
    FOCUS_QUESTION = (
        "Examine the leaves. For each that applies, describe vs the "
        "canonical visual_symptoms: (1) lesion SHAPE — circular / "
        "angular-vein-limited / irregular / elongated / ring / "
        "v-margin, and does it cross veins; (2) lesion COLOR — "
        "centre vs margin vs halo, named specifically; (3) lesion "
        "TEXTURE — sunken/raised, smooth/fuzzy/felty/water-soaked, "
        "shot-hole; (4) CHLOROSIS type — interveinal / marginal / "
        "generalized / mosaic / vein-banding and canopy position; "
        "(5) NECROSIS distribution — tip / margin / spots / "
        "vein-extending / whole-leaf; (6) CURL — curling / cupping / "
        "puckering / blistering and direction; (7) VEINS — clearing / "
        "necrosis / lesions vein-limited or crossing; (8) leaf "
        "GEOMETRY — shape, length:width, petiole:blade, serration. "
        "Emit a delta only where the photo adds to or contradicts "
        "canonical for one of your fields."
    )


class StemRootSymptomAgent(BaseAgent):
    AGENT_NAME = "StemRootSymptomAgent"
    OWNED_FIELDS = [
        "stem_lesion", "stem_pith", "stem_surface", "stem_discoloration",
        "root_visible", "crown_collar",
    ]
    SYSTEM_PROMPT = (
        "You are StemRootSymptomAgent, a vision specialist for ALL "
        "stem, crown, and root visual symptoms: outer-stem lesions, "
        "INTERNAL pith colour of a split stem, non-lesion surface "
        "features, vascular/surface discoloration, visible-root signs, "
        "and the crown/soil-line zone. Internal pith colour of a split "
        "stem is frequently the single most decisive visual fork in "
        "look-alike pairs — report it precisely when visible." + _VISUAL_ONLY
    )
    FOCUS_QUESTION = (
        "Examine any stem, crown, or root. Describe vs canonical "
        "visual_symptoms: (1) outer-stem LESIONS — canker / girdling / "
        "sunken-raised, location, colour; (2) split-stem PITH colour "
        "and outer-vascular colour IF a cut stem is visible (else say "
        "not visible — do not guess); (3) stem SURFACE non-lesion "
        "features — galls / blisters / scabs / ooze / fruiting bodies; "
        "(4) stem DISCOLORATION — vascular streaking / dark blotches / "
        "water-soaked streaks; (5) visible ROOTS — rot, cysts, fungal "
        "masses, clubbing, nodules; (6) CROWN/COLLAR — rot, girdling, "
        "sunken canker, mycelium/sclerotia at the soil line. Emit a "
        "delta only where the photo adds to or contradicts canonical."
    )


class FruitFlowerSignAgent(BaseAgent):
    AGENT_NAME = "FruitFlowerSignAgent"
    OWNED_FIELDS = ["flower", "fruit", "sporulation"]
    SYSTEM_PROMPT = (
        "You are FruitFlowerSignAgent, a vision specialist for "
        "reproductive-structure symptoms and visible PATHOGEN SIGNS: "
        "flower blight/distortion, fruit lesions/mummification/scab/rot, "
        "and signs of the pathogen itself (mycelium, spore masses, "
        "pycnidia, sclerotia, bacterial ooze, rust pustules). Pathogen "
        "SIGNS are a description of what is visible on the plant — NOT "
        "a pathogen identity claim." + _VISUAL_ONLY
    )
    FOCUS_QUESTION = (
        "Examine flowers, fruit, and any pathogen signs. Describe vs "
        "canonical visual_symptoms: (1) FLOWERS — browning/wilting, "
        "petal spotting, distortion/abortion, mould fuzz; (2) FRUIT — "
        "lesions (shape/colour/concentric), mummification, scab, soft "
        "rot, visible internal browning; (3) SIGNS — mycelium / spore "
        "masses / pycnidia / sclerotia / ooze / rust pustules with "
        "colour, density, and substrate. If a structure is not visible, "
        "say so. Emit a delta only where the photo adds to or "
        "contradicts canonical."
    )


class WholePlantSymptomAgent(BaseAgent):
    AGENT_NAME = "WholePlantSymptomAgent"
    OWNED_FIELDS = ["wilting", "defoliation", "spatial_pattern"]
    SYSTEM_PROMPT = (
        "You are WholePlantSymptomAgent, a vision specialist for "
        "whole-plant and whole-scene visual TOPOLOGY: wilting "
        "distribution, defoliation pattern (and what stays attached), "
        "and the spatial distribution of damage in the visible scene. "
        "Topology is often more diagnostic than single-lesion "
        "morphology." + _VISUAL_ONLY
    )
    FOCUS_QUESTION = (
        "Step back and read the whole plant / scene vs canonical "
        "visual_symptoms: (1) WILTING topology — whole-plant / one-side "
        "/ one-branch / hemispheric, and structural vs midday turgor; "
        "(2) DEFOLIATION — leaf-drop pattern, whether petioles remain "
        "attached after blades drop, which canopy layer; (3) SPATIAL "
        "pattern — within-canopy (top-down/bottom-up/scattered) and "
        "across plants (patches / rings / tillage-aligned / "
        "edge-of-field / low-spot). Emit a delta only where the photo "
        "adds to or contradicts canonical."
    )


class DiagnosticVisualAgent(BaseAgent):
    AGENT_NAME = "DiagnosticVisualAgent"
    OWNED_FIELDS = [
        "concentric_pattern", "color_palette", "look_alikes_visual",
        "severity_visible", "other",
    ]
    SYSTEM_PROMPT = (
        "You are DiagnosticVisualAgent, a cross-cutting vision "
        "specialist: concentric/target structure, the colour palette "
        "of affected tissue, VISUAL look-alike disambiguation against "
        "the canonical look_alikes list, and visible severity. For "
        "look_alikes_visual you walk a short discriminative "
        "chain-of-thought over the canonical look_alikes — purely on "
        "visual grounds — and flag if the photo visually matches a "
        "listed look-alike better, or if a visually-supported "
        "confusable is missing from the list. You assert nothing about "
        "pathogen identity or cause." + _VISUAL_ONLY
    )
    FOCUS_QUESTION = (
        "Apply cross-cutting visual reasoning vs canonical "
        "visual_symptoms: (1) CONCENTRIC structure — target spots / "
        "rings / bullseye / halo, ring count and colour sequence; "
        "(2) COLOR PALETTE of the affected tissue — 2-4 named colours "
        "with rough proportions and substrate; (3) LOOK-ALIKES — for "
        "each canonical look-alike, walk a brief VISUAL fork (what "
        "would I see for canonical vs for this look-alike?) and decide "
        "whether the pixels favour canonical or the look-alike, or that "
        "a visually-supported confusable is missing; (4) SEVERITY — "
        "fraction of the visible organ affected (low <10% / medium "
        "10-50% / high >50%), conservatively. Emit a delta only where "
        "the photo adds to or contradicts canonical."
    )


# The default roster: 5 generalized visual-symptom group agents.
# Partition of all 24 visual delta fields (8 + 6 + 3 + 3 + 4 + "other").
VISUAL_GROUP_AGENTS = (
    LeafSymptomAgent,
    StemRootSymptomAgent,
    FruitFlowerSignAgent,
    WholePlantSymptomAgent,
    DiagnosticVisualAgent,
)
