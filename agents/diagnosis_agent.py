"""
agents/diagnosis_agent.py
=========================
DiagnosisAgent — final JSON synthesis + TERMINATE (PlantSwarm §4).
Excluded from calibration ensemble (not an independent observer).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent, ContextEntry
from utils.vllm_client import VLLMClient


class DiagnosisAgent(BaseAgent):
    """Agent 5 of 5 — terminal aggregation."""

    AGENT_NAME = "DiagnosisAgent"
    TASK_IDS = ["T1", "T2", "T3", "T4", "T5"]
    HANDOFF_MENU: List[str] = []

    SYSTEM_PROMPT = (
        "You are DiagnosisAgent. Read all prior messages. Resolve contradictions by preferring "
        "later, more-informed predictions. Output valid JSON with keys: "
        "symptom_type (T1), pathogen_class (T2), disease_name (T3), severity_class (T4), "
        "crop_species (T5), confidence_t1 through confidence_t5 (high/medium/low), "
        "contradiction_resolved (bool), path_summary (one sentence). "
        "Then output TERMINATE on a new line. Do not output anything after TERMINATE."
    )

    def __init__(self, client: VLLMClient, label_space: Dict[str, List[str]], **kwargs: Any):
        super().__init__(client, label_space, **kwargs)

    def _parse_response(
        self,
        text: str,
        context: List[ContextEntry],
        backtrack_count: int,
    ) -> Tuple[Dict[str, str], str, Optional[str]]:
        predictions = self._extract_json_predictions(text)
        confidence = predictions.get("confidence_t1", "medium")
        return predictions, confidence, None

    def _extract_json_predictions(self, text: str) -> Dict[str, Any]:
        text = text.split("TERMINATE")[0].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {
            "symptom_type": "Other",
            "pathogen_class": "Other",
            "disease_name": "Other",
            "severity_class": "Early",
            "crop_species": "Other",
            "confidence_t1": "low",
            "confidence_t2": "low",
            "confidence_t3": "low",
            "confidence_t4": "low",
            "confidence_t5": "low",
            "contradiction_resolved": False,
            "path_summary": "Synthesis failed; fallback to defaults.",
        }

    def _score_all_tasks(self, context_text: str, image_b64: str) -> Dict[str, Dict[str, float]]:
        return {}
