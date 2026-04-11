"""
plantswarm/entropy_pipeline.py
==============================
Entropy-driven free-routing swarm (logprob / predictive entropy), parallel to κ-routing.

Uses per-token entropies from vLLM chat ``logprobs``, builds the entropy field
``E_t``, gradient ``G_t``, and routes with thresholds ``delta_1`` (backtrack) and
``delta_2`` (early synthesis) per Algorithm ``entropy_swarm``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agents.base_agent import AgentOutput, ContextEntry
from agents.diagnosis_agent import DiagnosisAgent
from agents.morphology_agent import MorphologyAgent
from agents.pathogen_agent import PathogenAgent
from agents.severity_agent import SeverityAgent
from agents.symptom_agent import SymptomAgent
from calibration.ensemble import argmax_label, ensemble_probabilities
from utils.sequence_entropy import entropy_gradient_G
from utils.vllm_client import VLLMClient

from .pipeline import RoutingTrace


class EntropyPlantSwarmPipeline:
    """
    Same agent stack as :class:`plantswarm.pipeline.PlantSwarmPipeline`, but routing
    uses mean sequence entropy H_t and gradient G_t instead of κ alone.
    """

    def __init__(
        self,
        client: VLLMClient,
        label_space: Dict[str, List[str]],
        Tmax: int = 15,
        confidence_weights: Optional[Dict[str, int]] = None,
        delta1: float = 0.05,
        delta2: float = 0.35,
    ):
        self.client = client
        if hasattr(client, "top_logprobs"):
            client.top_logprobs = max(getattr(client, "top_logprobs", 20), 10)
        self.label_space = label_space
        self.Tmax = Tmax
        self.confidence_weights = confidence_weights or {"high": 3, "medium": 2, "low": 1}
        self.delta1 = delta1
        self.delta2 = delta2

        kw = {"sequence_entropy": True}
        self.agents: Dict[str, Any] = {
            "MorphologyAgent": MorphologyAgent(client, label_space, **kw),
            "SymptomAgent": SymptomAgent(client, label_space, **kw),
            "PathogenAgent": PathogenAgent(client, label_space, **kw),
            "SeverityAgent": SeverityAgent(client, label_space, **kw),
            "DiagnosisAgent": DiagnosisAgent(client, label_space, **kw),
        }

    def run(self, image_id: str, image_b64: str) -> RoutingTrace:
        t0 = time.time()
        context: List[ContextEntry] = []
        agent_outputs: List[AgentOutput] = []
        agent_log_probs: Dict[str, Dict[str, Dict[str, float]]] = {}
        agent_confidences: Dict[str, str] = {}

        H_sequence: List[float] = []
        entropy_field: List[Dict[str, Any]] = []
        entropy_gradients: List[float] = []

        bt = 0
        Omega = 0
        current_agent_name = "MorphologyAgent"
        path: List[str] = []
        early_terminated = False
        t_step = 0

        while current_agent_name != "DiagnosisAgent" and t_step < self.Tmax:
            agent = self.agents[current_agent_name]
            output: AgentOutput = agent(
                image_b64=image_b64,
                context=context,
                backtrack_count=bt,
            )
            Omega += output.tokens_used
            H_t = output.mean_entropy_H
            if H_t is None:
                H_t = 0.0
            D_t = output.entropy_dispersion_D
            h_seq = output.token_entropies or []

            context.append(
                ContextEntry(
                    agent_name=current_agent_name,
                    message=output.message,
                    confidence=output.confidence,
                    log_probs=output.log_probs,
                    mean_entropy_H=H_t,
                    entropy_dispersion_D=D_t,
                    token_entropies=h_seq if h_seq else None,
                )
            )
            agent_outputs.append(output)
            path.append(current_agent_name)
            agent_log_probs[current_agent_name] = output.log_probs
            agent_confidences[current_agent_name] = output.confidence

            H_sequence.append(H_t)
            entropy_field.append(
                {
                    "agent": current_agent_name,
                    "H_t": H_t,
                    "D_t": D_t,
                    "h_len": len(h_seq),
                    "H_disease": output.targeted_disease_entropy,
                }
            )
            G_t = entropy_gradient_G(H_sequence)
            entropy_gradients.append(G_t)

            default_forward = output.handoff_target or "DiagnosisAgent"

            if t_step >= 1 and G_t > self.delta1:
                next_agent_name = "MorphologyAgent"
                bt += 1
            elif H_t < self.delta2:
                next_agent_name = "DiagnosisAgent"
                early_terminated = True
            else:
                next_agent_name = default_forward

            current_agent_name = next_agent_name
            t_step += 1

            if early_terminated:
                break

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
            entropy_field=entropy_field,
            entropy_gradients=entropy_gradients,
            routing_signal="entropy",
        )
