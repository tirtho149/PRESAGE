"""
agents/morphology_agent.py
==========================
MorphologyAgent — emits deltas for lesion morphology, affected plant
organs, and diagnostic features visible in the photograph.

Also the **regrounding target**: when any agent has low confidence and
hasn't backtracked yet, the orchestrator routes back here so the next
step starts from a visual re-observation.
"""

from __future__ import annotations

from agents.base_agent import BaseAgent


class MorphologyAgent(BaseAgent):
    AGENT_NAME = "MorphologyAgent"
    OWNED_FIELDS = ["lesion_morphology", "affected_organs", "diagnostic_features"]
    HANDOFF_MENU = ["SymptomAgent", "SeverityAgent", "DiagnosisAgent"]
    DEFAULT_FORWARD = "SymptomAgent"

    SYSTEM_PROMPT = (
        "You are MorphologyAgent. Inspect the photograph for lesion shape, "
        "size, margin, colour, surface texture; which plant organs are "
        "affected; and any diagnostic features visible in this specific "
        "image. Output strict JSON only — no prose, no markdown."
    )
