"""
baselines/fixed_chain.py
========================
Fixed Sequential Chain baseline — PlantSwarm paper (vs free-routing).
κ is stored but not used for routing.
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
class FixedChainTrace:
    image_id: str
    path: List[str] = field(default_factory=list)
    total_tokens: int = 0
    agent_outputs: List[AgentOutput] = field(default_factory=list)
    final_predictions: Dict[str, str] = field(default_factory=dict)
    ensemble_probs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    stored_confidences: Dict[str, str] = field(default_factory=dict)
    wall_time_s: float = 0.0


class FixedChainBaseline:
    """Fixed order Morphology → Symptom → Pathogen → Severity → Diagnosis."""

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
        confidence_weights: Optional[Dict[str, int]] = None,
    ):
        self.label_space = label_space
        self.confidence_weights = confidence_weights or {"high": 3, "medium": 2, "low": 1}

        self.agents: Dict[str, Any] = {
            "MorphologyAgent": MorphologyAgent(client, label_space),
            "SymptomAgent": SymptomAgent(client, label_space),
            "PathogenAgent": PathogenAgent(client, label_space),
            "SeverityAgent": SeverityAgent(client, label_space),
            "DiagnosisAgent": DiagnosisAgent(client, label_space),
        }

    def run(self, image_id: str, image_b64: str) -> FixedChainTrace:
        t0 = time.time()
        context: List[ContextEntry] = []
        agent_outputs: List[AgentOutput] = []
        agent_log_probs: Dict[str, Dict[str, Dict[str, float]]] = {}
        agent_confidences: Dict[str, str] = {}
        total_tokens = 0
        path = []

        for agent_name in self.AGENT_ORDER[:-1]:
            agent = self.agents[agent_name]
            output: AgentOutput = agent(
                image_b64=image_b64,
                context=context,
                backtrack_count=0,
            )
            total_tokens += output.tokens_used
            context.append(
                ContextEntry(
                    agent_name=agent_name,
                    message=output.message,
                    confidence=output.confidence,
                    log_probs=output.log_probs,
                )
            )
            agent_outputs.append(output)
            agent_log_probs[agent_name] = output.log_probs
            agent_confidences[agent_name] = output.confidence
            path.append(agent_name)

        diag_agent: DiagnosisAgent = self.agents["DiagnosisAgent"]
        diag_output = diag_agent(
            image_b64=image_b64,
            context=context,
            backtrack_count=0,
        )
        total_tokens += diag_output.tokens_used
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

        return FixedChainTrace(
            image_id=image_id,
            path=path,
            total_tokens=total_tokens,
            agent_outputs=agent_outputs,
            final_predictions=final_predictions,
            ensemble_probs=ensemble_probs,
            stored_confidences=agent_confidences,
            wall_time_s=time.time() - t0,
        )

    def run_batch(self, records) -> List[FixedChainTrace]:
        return [self.run(r.image_id, r.image_b64) for r in records]
