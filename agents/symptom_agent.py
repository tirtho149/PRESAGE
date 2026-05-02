"""
agents/symptom_agent.py
=======================
SymptomAgent — T1 symptom complex (PlantSwarm Table: tasks).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent, ContextEntry
from utils.vllm_client import VLLMClient


class SymptomAgent(BaseAgent):
    """Agent 2 of 5. Scope: T1 (symptom_type)."""

    AGENT_NAME = "SymptomAgent"
    TASK_IDS = ["T1"]
    HANDOFF_MENU = ["MorphologyAgent", "PathogenAgent", "SeverityAgent", "DiagnosisAgent"]

    SYSTEM_PROMPT = (
        "You are SymptomAgent. Task T1: classify the symptom complex using exactly one label "
        "from your task vocabulary (see user context). Read all prior messages. "
        "Give one-line reasoning and confidence (high / medium / low). "
        "If confidence=low AND no prior backtrack: handoff to MorphologyAgent. "
        "If confidence=low AND already backtracked: handoff forward. "
        "If confidence=high AND your scope tasks are fully resolved: handoff to DiagnosisAgent. "
        "Otherwise handoff to PathogenAgent or SeverityAgent as most relevant. "
        "Name the next agent explicitly in your reply."
    )

    def __init__(self, client: VLLMClient, label_space: Dict[str, List[str]], **kwargs: Any):
        super().__init__(client, label_space, **kwargs)
        self._t1_labels = label_space.get("T1", ["Other"])

    def _parse_response(
        self,
        text: str,
        context: List[ContextEntry],
        backtrack_count: int,
    ) -> Tuple[Dict[str, str], str, Optional[str]]:
        confidence = self._extract_confidence(text)
        label = self._extract_label(text, self._t1_labels) or self._t1_labels[-1]
        all_tasks_covered = False  # Only T1 covered; T2-T5 pending
        handoff = self._routing_decision(
            confidence=confidence,
            backtrack_count=backtrack_count,
            all_tasks_covered=all_tasks_covered,
            handoff_menu=self.HANDOFF_MENU,
            default_forward="PathogenAgent",
        )
        return {"T1": label}, confidence, handoff
