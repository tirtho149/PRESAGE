"""
plantswarm/pipeline.py
======================
PlantSwarm: confidence-gated free-routing (Algorithm in PlantSwarm paper).

Runtime orchestration uses ``AutoGenPlantSwarmPipeline`` in ``autogen_pipeline.py``
(AutoGen AgentChat Swarm). ``PlantSwarmPipeline`` remains for ``RoutingTrace``,
task-coverage helpers, and alignment with the paper's routing semantics.
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
from utils.vllm_client import VLLMClient


@dataclass
class RoutingTrace:
    """Routing trace for one image."""

    image_id: str
    path: List[str] = field(default_factory=list)
    path_length: int = 0
    backtrack_count: int = 0
    loop_rate: float = 0.0
    early_terminated: bool = False
    total_tokens: int = 0
    agent_outputs: List[AgentOutput] = field(default_factory=list)
    final_predictions: Dict[str, str] = field(default_factory=dict)
    ensemble_probs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    wall_time_s: float = 0.0
    entropy_field: List[Dict[str, Any]] = field(default_factory=list)
    entropy_gradients: List[float] = field(default_factory=list)
    routing_signal: str = "kappa"  # "kappa" | "entropy"


TASK_COVERAGE: Dict[str, str] = {
    "T1": "SymptomAgent",
    "T2": "PathogenAgent",
    "T3": "PathogenAgent",
    "T4": "SeverityAgent",
    "T5": "SeverityAgent",
}

ALL_TASKS = {"T1", "T2", "T3", "T4", "T5"}
CORE_DIAG_TASKS = {"T1", "T2", "T3"}


def _tasks_covered(outputs: List[AgentOutput]) -> bool:
    covered = set()
    for out in outputs:
        covered.update(out.predictions.keys())
    return ALL_TASKS.issubset(covered)


def _core_diag_covered(outputs: List[AgentOutput]) -> bool:
    covered = set()
    for out in outputs:
        covered.update(out.predictions.keys())
    return CORE_DIAG_TASKS.issubset(covered)


def _early_termination(kappa: str, current_agent_name: str, agent_outputs: List[AgentOutput]) -> bool:
    """High confidence + (all tasks covered, or paper-style T1–T3 after PathogenAgent)."""
    if kappa != "high":
        return False
    if _tasks_covered(agent_outputs):
        return True
    if current_agent_name == "PathogenAgent" and _core_diag_covered(agent_outputs):
        return True
    return False


class PlantSwarmPipeline:
    """Confidence-gated free-routing multi-agent pipeline."""

    AGENT_ORDER = [
        "MorphologyAgent",
        "SymptomAgent",
        "PathogenAgent",
        "SeverityAgent",
        "DiagnosisAgent",
    ]

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

    def run(self, image_id: str, image_b64: str) -> RoutingTrace:
        t0 = time.time()
        context: List[ContextEntry] = []
        agent_outputs: List[AgentOutput] = []
        agent_log_probs: Dict[str, Dict[str, Dict[str, float]]] = {}
        agent_confidences: Dict[str, str] = {}

        t = 0
        Omega = 0
        bt = 0
        current_agent_name = "MorphologyAgent"
        path: List[str] = []
        early_terminated = False

        while current_agent_name != "DiagnosisAgent" and t < self.Tmax:
            agent = self.agents[current_agent_name]
            output: AgentOutput = agent(
                image_b64=image_b64,
                context=context,
                backtrack_count=bt,
            )
            Omega += output.tokens_used
            context.append(
                ContextEntry(
                    agent_name=current_agent_name,
                    message=output.message,
                    confidence=output.confidence,
                    log_probs=output.log_probs,
                )
            )
            agent_outputs.append(output)
            path.append(current_agent_name)
            agent_log_probs[current_agent_name] = output.log_probs
            agent_confidences[current_agent_name] = output.confidence
            t += 1

            kappa = output.confidence

            if kappa == "low" and bt == 0:
                next_agent_name = "MorphologyAgent"
                bt += 1
            elif _early_termination(kappa, current_agent_name, agent_outputs):
                next_agent_name = "DiagnosisAgent"
                early_terminated = True
            else:
                next_agent_name = output.handoff_target or "DiagnosisAgent"

            current_agent_name = next_agent_name

        diagnosis_agent: DiagnosisAgent = self.agents["DiagnosisAgent"]
        diag_output: AgentOutput = diagnosis_agent(
            image_b64=image_b64,
            context=context,
            backtrack_count=bt,
        )
        Omega += diag_output.tokens_used
        path.append("DiagnosisAgent")
        agent_outputs.append(diag_output)

        ensemble_probs: Dict[str, Dict[str, float]] = {}
        for task_id, labels in self.label_space.items():
            ensemble_probs[task_id] = ensemble_probabilities(
                agent_log_probs=agent_log_probs,
                agent_confidences=agent_confidences,
                task_id=task_id,
                label_list=labels,
                confidence_weights=self.confidence_weights,
            )

        synth_preds = diag_output.predictions
        final_predictions = {
            "T1": synth_preds.get("symptom_type", argmax_label(ensemble_probs.get("T1", {}))),
            "T2": synth_preds.get("pathogen_class", argmax_label(ensemble_probs.get("T2", {}))),
            "T3": synth_preds.get("disease_name", argmax_label(ensemble_probs.get("T3", {}))),
            "T4": synth_preds.get("severity_class", argmax_label(ensemble_probs.get("T4", {}))),
            "T5": synth_preds.get("crop_species", argmax_label(ensemble_probs.get("T5", {}))),
        }

        revisits = len(path) - len(set(path))
        loop_rate = revisits / max(len(path), 1)

        return RoutingTrace(
            image_id=image_id,
            path=path,
            path_length=len(path),
            backtrack_count=bt,
            loop_rate=loop_rate,
            early_terminated=early_terminated,
            total_tokens=Omega,
            agent_outputs=agent_outputs,
            final_predictions=final_predictions,
            ensemble_probs=ensemble_probs,
            wall_time_s=time.time() - t0,
            routing_signal="kappa",
        )
