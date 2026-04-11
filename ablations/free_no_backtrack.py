"""
ablations/free_no_backtrack.py
==============================
Free routing without MorphologyAgent backtrack (PlantSwarm ablation).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.base_agent import AgentOutput, ContextEntry
from agents.diagnosis_agent import DiagnosisAgent
from agents.morphology_agent import MorphologyAgent
from agents.pathogen_agent import PathogenAgent
from agents.severity_agent import SeverityAgent
from agents.symptom_agent import SymptomAgent
from calibration.ensemble import argmax_label, ensemble_probabilities
from plantswarm.pipeline import _core_diag_covered, _tasks_covered
from utils.vllm_client import VLLMClient


def _early_no_bt(kappa: str, current: str, outputs: List[AgentOutput]) -> bool:
    if kappa != "high":
        return False
    if _tasks_covered(outputs):
        return True
    if current == "PathogenAgent" and _core_diag_covered(outputs):
        return True
    return False


FORWARD_DEFAULT = {
    "MorphologyAgent": "SymptomAgent",
    "SymptomAgent": "PathogenAgent",
    "PathogenAgent": "SeverityAgent",
    "SeverityAgent": "DiagnosisAgent",
}


@dataclass
class FreeNoBacktrackTrace:
    image_id: str
    path: List[str] = field(default_factory=list)
    total_tokens: int = 0
    agent_outputs: List[AgentOutput] = field(default_factory=list)
    final_predictions: Dict[str, str] = field(default_factory=dict)
    ensemble_probs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    backtrack_count: int = 0
    early_terminated: bool = False
    wall_time_s: float = 0.0


class FreeNoBacktrackAblation:
    """Confidence-gated routing but no regrounding to MorphologyAgent."""

    def __init__(
        self,
        client: VLLMClient,
        label_space: Dict[str, List[str]],
        Tmax: int = 15,
        confidence_weights: Optional[Dict[str, int]] = None,
    ):
        self.Tmax = Tmax
        self.label_space = label_space
        self.confidence_weights = confidence_weights or {"high": 3, "medium": 2, "low": 1}
        self.agents: Dict[str, Any] = {
            "MorphologyAgent": MorphologyAgent(client, label_space),
            "SymptomAgent": SymptomAgent(client, label_space),
            "PathogenAgent": PathogenAgent(client, label_space),
            "SeverityAgent": SeverityAgent(client, label_space),
            "DiagnosisAgent": DiagnosisAgent(client, label_space),
        }

    def run(self, image_id: str, image_b64: str) -> FreeNoBacktrackTrace:
        t0 = time.time()
        context: List[ContextEntry] = []
        agent_outputs: List[AgentOutput] = []
        agent_log_probs: Dict[str, Dict[str, Dict[str, float]]] = {}
        agent_confidences: Dict[str, str] = {}
        Omega = 0
        path: List[str] = []
        bt = 0
        early_terminated = False
        current = "MorphologyAgent"
        t = 0

        while current != "DiagnosisAgent" and t < self.Tmax:
            agent = self.agents[current]
            output = agent(image_b64=image_b64, context=context, backtrack_count=bt)
            Omega += output.tokens_used
            context.append(
                ContextEntry(
                    agent_name=current,
                    message=output.message,
                    confidence=output.confidence,
                    log_probs=output.log_probs,
                )
            )
            agent_outputs.append(output)
            path.append(current)
            agent_log_probs[current] = output.log_probs
            agent_confidences[current] = output.confidence
            t += 1

            kappa = output.confidence
            declared = output.handoff_target

            if _early_no_bt(kappa, current, agent_outputs):
                current = "DiagnosisAgent"
                early_terminated = True
            elif kappa == "low":
                if declared and declared != "MorphologyAgent":
                    current = declared
                else:
                    current = FORWARD_DEFAULT.get(current, "DiagnosisAgent")
            else:
                if declared and declared != "MorphologyAgent":
                    current = declared
                else:
                    current = FORWARD_DEFAULT.get(current, "DiagnosisAgent")

        synth = self.agents["DiagnosisAgent"]
        synth_out = synth(image_b64=image_b64, context=context, backtrack_count=bt)
        Omega += synth_out.tokens_used
        path.append("DiagnosisAgent")
        agent_outputs.append(synth_out)

        ensemble_probs: Dict[str, Dict[str, float]] = {}
        for task_id, labels in self.label_space.items():
            ensemble_probs[task_id] = ensemble_probabilities(
                agent_log_probs=agent_log_probs,
                agent_confidences=agent_confidences,
                task_id=task_id,
                label_list=labels,
                confidence_weights=self.confidence_weights,
            )

        sp = synth_out.predictions
        final_predictions = {
            "T1": sp.get("symptom_type", argmax_label(ensemble_probs.get("T1", {}))),
            "T2": sp.get("pathogen_class", argmax_label(ensemble_probs.get("T2", {}))),
            "T3": sp.get("disease_name", argmax_label(ensemble_probs.get("T3", {}))),
            "T4": sp.get("severity_class", argmax_label(ensemble_probs.get("T4", {}))),
            "T5": sp.get("crop_species", argmax_label(ensemble_probs.get("T5", {}))),
        }

        return FreeNoBacktrackTrace(
            image_id=image_id,
            path=path,
            total_tokens=Omega,
            agent_outputs=agent_outputs,
            final_predictions=final_predictions,
            ensemble_probs=ensemble_probs,
            backtrack_count=bt,
            early_terminated=early_terminated,
            wall_time_s=time.time() - t0,
        )

    def run_batch(self, records) -> List[FreeNoBacktrackTrace]:
        return [self.run(r.image_id, r.image_b64) for r in records]
