"""
plantswarm/hf_pipeline.py
=========================
Confidence-gated routing pipeline (Algorithm 1) using HFClient.

Functionally identical to AutoGenPlantSwarmPipeline but:
  - No AutoGen dependency
  - No vLLM HTTP server required
  - Runs the model in-process via utils.hf_client.HFClient

Use this on single-GPU Nova nodes where you cannot run a separate vLLM
server alongside the main script.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from agents.base_agent import AgentOutput, ContextEntry
from agents.diagnosis_agent import DiagnosisAgent
from agents.morphology_agent import MorphologyAgent
from agents.pathogen_agent import PathogenAgent
from agents.severity_agent import SeverityAgent
from agents.symptom_agent import SymptomAgent
from calibration.ensemble import argmax_label, ensemble_probabilities
from .pipeline import RoutingTrace


class HFDirectPipeline:
    """
    Algorithm 1 confidence-gated routing running entirely in-process.
    Drop-in replacement for AutoGenPlantSwarmPipeline.

    Uses HFClient (utils/hf_client.py) — no AutoGen, no vLLM server.
    """

    _AGENT_SEQUENCE = [
        "MorphologyAgent",
        "SymptomAgent",
        "PathogenAgent",
        "SeverityAgent",
        "DiagnosisAgent",
    ]

    def __init__(
        self,
        client,          # HFClient (or VLLMClient — same interface)
        label_space: Dict[str, List[str]],
        Tmax: int = 15,
        confidence_weights: Optional[Dict[str, int]] = None,
    ):
        self.label_space = label_space
        self.Tmax = Tmax
        self.confidence_weights = confidence_weights or {"high": 3, "medium": 2, "low": 1}

        self.agents: Dict[str, any] = {
            "MorphologyAgent": MorphologyAgent(client, label_space),
            "SymptomAgent":    SymptomAgent(client, label_space),
            "PathogenAgent":   PathogenAgent(client, label_space),
            "SeverityAgent":   SeverityAgent(client, label_space),
            "DiagnosisAgent":  DiagnosisAgent(client, label_space),
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, image_id: str, image_b64: str) -> RoutingTrace:
        t0 = time.time()

        context: List[ContextEntry] = []
        path: List[str] = []
        agent_outputs: List[AgentOutput] = []
        agent_log_probs: Dict[str, Dict] = {}
        agent_confidences: Dict[str, str] = {}
        backtrack_count = 0
        total_tokens = 0
        early_terminated = False

        current = "MorphologyAgent"

        for _step in range(self.Tmax):
            output = self.agents[current](image_b64, context, backtrack_count)

            context.append(ContextEntry(
                agent_name=current,
                message=output.message,
                confidence=output.confidence,
                log_probs=output.log_probs,
                mean_entropy_H=output.mean_entropy_H,
                entropy_dispersion_D=output.entropy_dispersion_D,
                token_entropies=output.token_entropies,
            ))
            path.append(current)
            agent_outputs.append(output)
            total_tokens += output.tokens_used
            agent_log_probs[current] = output.log_probs
            agent_confidences[current] = output.confidence

            if output.handoff_target == "MorphologyAgent" and current != "MorphologyAgent":
                backtrack_count += 1

            if current == "DiagnosisAgent":
                early_terminated = True
                break

            next_agent = output.handoff_target or self._default_next(current)
            current = next_agent or "DiagnosisAgent"

        # Guarantee a DiagnosisAgent output
        synth = next(
            (o for o in reversed(agent_outputs) if o.agent_name == "DiagnosisAgent"),
            None,
        )
        if synth is None:
            synth = self.agents["DiagnosisAgent"](image_b64, context, backtrack_count)
            path.append("DiagnosisAgent")
            agent_outputs.append(synth)
            agent_log_probs["DiagnosisAgent"] = synth.log_probs
            agent_confidences["DiagnosisAgent"] = synth.confidence
            total_tokens += synth.tokens_used

        # Weighted confidence ensemble (Eq. 3)
        ensemble_probs: Dict[str, Dict[str, float]] = {}
        for task_id, labels in self.label_space.items():
            ensemble_probs[task_id] = ensemble_probabilities(
                agent_log_probs=agent_log_probs,
                agent_confidences=agent_confidences,
                task_id=task_id,
                label_list=labels,
                confidence_weights=self.confidence_weights,
            )

        sp = synth.predictions
        final_predictions = {
            "T1": sp.get("symptom_type",   argmax_label(ensemble_probs.get("T1", {}))),
            "T2": sp.get("pathogen_class", argmax_label(ensemble_probs.get("T2", {}))),
            "T3": sp.get("disease_name",   argmax_label(ensemble_probs.get("T3", {}))),
            "T4": sp.get("severity_class", argmax_label(ensemble_probs.get("T4", {}))),
            "T5": sp.get("crop_species",   argmax_label(ensemble_probs.get("T5", {}))),
        }

        revisits = len(path) - len(set(path))
        return RoutingTrace(
            image_id=image_id,
            path=path,
            path_length=len(path),
            backtrack_count=backtrack_count,
            loop_rate=revisits / max(len(path), 1),
            early_terminated=early_terminated,
            total_tokens=total_tokens,
            agent_outputs=agent_outputs,
            final_predictions=final_predictions,
            ensemble_probs=ensemble_probs,
            wall_time_s=time.time() - t0,
            routing_signal="hf_direct",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _default_next(self, current: str) -> Optional[str]:
        """Next agent in the fixed sequence when handoff_target is None."""
        try:
            idx = self._AGENT_SEQUENCE.index(current)
        except ValueError:
            return None
        if idx + 1 < len(self._AGENT_SEQUENCE):
            return self._AGENT_SEQUENCE[idx + 1]
        return None
