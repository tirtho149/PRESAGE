"""
agents/severity_agent.py
========================
SeverityAgent — T4 severity and T5 crop species (PlantSwarm Table: tasks).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent, ContextEntry
from utils.vllm_client import VLLMClient


class SeverityAgent(BaseAgent):
    """Agent 4 of 5. Scope: T4, T5."""

    AGENT_NAME = "SeverityAgent"
    TASK_IDS = ["T4", "T5"]
    HANDOFF_MENU = ["MorphologyAgent", "DiagnosisAgent"]

    SYSTEM_PROMPT = (
        "You are SeverityAgent. Task T4: severity class from the fixed label set. "
        "Task T5: crop species from the fixed label set. "
        "Read all prior messages. Report confidence (high / medium / low). "
        "If confidence=low AND no prior backtrack: handoff to MorphologyAgent. "
        "Otherwise handoff to DiagnosisAgent."
    )

    def __init__(self, client: VLLMClient, label_space: Dict[str, List[str]], **kwargs: Any):
        super().__init__(client, label_space, **kwargs)
        self._t4 = label_space.get("T4", ["Moderate"])
        self._t5 = label_space.get("T5", ["Other"])

    def _parse_response(
        self,
        text: str,
        context: List[ContextEntry],
        backtrack_count: int,
    ) -> Tuple[Dict[str, str], str, Optional[str]]:
        confidence = self._extract_confidence(text)
        severity = self._extract_label(text, self._t4) or self._t4[0]
        crop = self._extract_label(text, self._t5) or self._t5[-1]
        all_tasks_covered = True
        handoff = self._routing_decision(
            confidence=confidence,
            backtrack_count=backtrack_count,
            all_tasks_covered=all_tasks_covered,
            handoff_menu=self.HANDOFF_MENU,
            default_forward="DiagnosisAgent",
        )
        return {"T4": severity, "T5": crop}, confidence, handoff
