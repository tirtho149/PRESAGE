"""
agents/pathogen_agent.py
========================
PathogenAgent — T2 pathogen class and T3 disease name (PlantSwarm Table: tasks).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent, ContextEntry
from utils.vllm_client import VLLMClient


class PathogenAgent(BaseAgent):
    """Agent 3 of 5. Scope: T2, T3."""

    AGENT_NAME = "PathogenAgent"
    TASK_IDS = ["T2", "T3"]
    HANDOFF_MENU = ["MorphologyAgent", "SymptomAgent", "SeverityAgent", "DiagnosisAgent"]

    SYSTEM_PROMPT = (
        "You are PathogenAgent. Task T2: pathogen class from the fixed label set. "
        "Task T3: specific disease name from the label set (use 'Other' if uncertain). "
        "Read all prior messages. Report confidence (high / medium / low). "
        "Use hedge words if cues are ambiguous. "
        "If confidence=low AND no prior backtrack: handoff to MorphologyAgent. "
        "Otherwise route forward to SymptomAgent, SeverityAgent, or DiagnosisAgent as appropriate. "
        "Name the next agent explicitly."
    )

    def __init__(self, client: VLLMClient, label_space: Dict[str, List[str]], **kwargs: Any):
        super().__init__(client, label_space, **kwargs)
        self._t3_labels = label_space.get("T3", ["Other"])

    def _parse_response(
        self,
        text: str,
        context: List[ContextEntry],
        backtrack_count: int,
    ) -> Tuple[Dict[str, str], str, Optional[str]]:
        confidence = self._extract_confidence(text)
        t2_labels = self.label_space.get("T2", [])
        pathogen = self._extract_label(text, t2_labels) or (t2_labels[-1] if t2_labels else "Other")
        disease = self._extract_label(text, self._t3_labels) or "Other"
        all_tasks_covered = False  # T1-T3 covered; T4-T5 pending
        handoff = self._routing_decision(
            confidence=confidence,
            backtrack_count=backtrack_count,
            all_tasks_covered=all_tasks_covered,
            handoff_menu=self.HANDOFF_MENU,
            default_forward="SeverityAgent",
        )
        return {"T2": pathogen, "T3": disease}, confidence, handoff
