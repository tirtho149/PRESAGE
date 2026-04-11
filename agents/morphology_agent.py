"""
agents/morphology_agent.py
===========================
MorphologyAgent — visual grounding only (PlantSwarm §4, Table: agents).
Entry point and backtrack target for low-confidence downstream agents.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent, ContextEntry
from utils.vllm_client import VLLMClient


class MorphologyAgent(BaseAgent):
    """Agent 1 of 5. Describes lesion morphology; no task predictions."""

    AGENT_NAME = "MorphologyAgent"
    TASK_IDS: List[str] = []
    HANDOFF_MENU = ["SymptomAgent", "SeverityAgent"]

    SYSTEM_PROMPT = (
        "You are MorphologyAgent. Describe lesion shape, color, distribution, affected tissue, "
        "and surface texture in detail. Do NOT name a disease, pathogen, or crop species. "
        "After your description, choose ONE next specialist: SymptomAgent (for symptom-pattern "
        "classification) or SeverityAgent (if severity or extent is the most salient cue). "
        "Report confidence as exactly one of: high / medium / low. "
        "State your chosen handoff target explicitly in text (SymptomAgent or SeverityAgent)."
    )

    def __init__(self, client: VLLMClient, label_space: Dict[str, List[str]], **kwargs: Any):
        super().__init__(client, label_space, **kwargs)

    def _parse_response(
        self,
        text: str,
        context: List[ContextEntry],
        backtrack_count: int,
    ) -> Tuple[Dict[str, str], str, Optional[str]]:
        confidence = self._extract_confidence(text)
        text_lower = text.lower()
        if "severityagent" in text_lower or "severity agent" in text_lower:
            handoff = "SeverityAgent"
        else:
            handoff = "SymptomAgent"
        return {}, confidence, handoff

    def _score_all_tasks(self, context_text: str, image_b64: str) -> Dict[str, Dict[str, float]]:
        return {}
