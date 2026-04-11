"""
ablations/three_agent_swarm.py
==============================
3-Agent ablation: MorphologyAgent + MultiTaskAgent (T1–T3) + DiagnosisAgent.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import AgentOutput, BaseAgent, ContextEntry
from agents.diagnosis_agent import DiagnosisAgent
from agents.morphology_agent import MorphologyAgent
from calibration.ensemble import argmax_label, ensemble_probabilities
from utils.vllm_client import VLLMClient


MULTITASK_PROMPT = """You are PathoSymptomAgent, combining symptom classification (T1) and
pathogen/disease identification (T2,T3) for plant disease images.
Predict T1 symptom_type, T2 pathogen_class, T3 disease_name using the label sets
from configuration / prior context.
Report overall_confidence (high/medium/low).
If overall_confidence=low AND no prior backtrack: handoff MorphologyAgent.
If overall_confidence=high: handoff DiagnosisAgent.
Otherwise handoff DiagnosisAgent or MorphologyAgent as appropriate.
Output JSON: {symptom_type, pathogen_class, disease_name, overall_confidence, handoff}.
"""


class MultiTaskAgent(BaseAgent):
    AGENT_NAME = "MultiTaskAgent"
    TASK_IDS = ["T1", "T2", "T3"]
    HANDOFF_MENU = ["MorphologyAgent", "DiagnosisAgent"]
    SYSTEM_PROMPT = MULTITASK_PROMPT

    def __init__(self, client: VLLMClient, label_space: Dict[str, List[str]]):
        super().__init__(client, label_space)

    def _parse_response(
        self,
        text: str,
        context: List[ContextEntry],
        backtrack_count: int,
    ) -> Tuple[Dict[str, str], str, Optional[str]]:
        preds = self._extract_json(text)
        confidence = preds.get("overall_confidence", self._extract_confidence(text))
        if confidence == "low" and backtrack_count == 0:
            handoff = "MorphologyAgent"
        elif confidence == "high":
            handoff = "DiagnosisAgent"
        else:
            handoff = preds.get("handoff", "DiagnosisAgent") or "DiagnosisAgent"
        task_preds = {
            "T1": preds.get("symptom_type", "Other"),
            "T2": preds.get("pathogen_class", "Other"),
            "T3": preds.get("disease_name", "Other"),
        }
        return task_preds, confidence, handoff

    def _extract_json(self, text: str) -> Dict:
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}


CORE = {"T1", "T2", "T3"}


@dataclass
class ThreeAgentTrace:
    image_id: str
    path: List[str] = field(default_factory=list)
    total_tokens: int = 0
    agent_outputs: List[AgentOutput] = field(default_factory=list)
    final_predictions: Dict[str, str] = field(default_factory=dict)
    ensemble_probs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    backtrack_count: int = 0
    early_terminated: bool = False
    wall_time_s: float = 0.0


class ThreeAgentSwarmAblation:
    VARIANT_NAME = "3-Agent Swarm"
    Tmax = 10

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
            "MultiTaskAgent": MultiTaskAgent(client, label_space),
            "DiagnosisAgent": DiagnosisAgent(client, label_space),
        }

    def run(self, image_id: str, image_b64: str) -> ThreeAgentTrace:
        t0 = time.time()
        context: List[ContextEntry] = []
        agent_outputs: List[AgentOutput] = []
        agent_log_probs: Dict[str, Dict[str, Dict[str, float]]] = {}
        agent_confidences: Dict[str, str] = {}
        total_tokens = 0
        path: List[str] = []
        bt = 0
        early_terminated = False
        current = "MorphologyAgent"
        t = 0

        while current != "DiagnosisAgent" and t < self.Tmax:
            agent = self.agents[current]
            output = agent(image_b64=image_b64, context=context, backtrack_count=bt)
            total_tokens += output.tokens_used
            context.append(
                ContextEntry(
                    agent_name=current,
                    message=output.message,
                    confidence=output.confidence,
                    log_probs=output.log_probs,
                )
            )
            agent_outputs.append(output)
            agent_log_probs[current] = output.log_probs
            agent_confidences[current] = output.confidence
            path.append(current)
            t += 1

            kappa = output.confidence
            covered = set()
            for out in agent_outputs:
                covered.update(out.predictions.keys())
            tasks_done = CORE.issubset(covered)

            if kappa == "low" and bt == 0 and current != "MorphologyAgent":
                current = "MorphologyAgent"
                bt += 1
            elif kappa == "high" and tasks_done:
                current = "DiagnosisAgent"
                early_terminated = True
            elif current == "MorphologyAgent":
                current = "MultiTaskAgent"
            else:
                current = output.handoff_target or "DiagnosisAgent"

        synth = self.agents["DiagnosisAgent"]
        synth_out = synth(image_b64=image_b64, context=context, backtrack_count=bt)
        total_tokens += synth_out.tokens_used
        path.append("DiagnosisAgent")
        agent_outputs.append(synth_out)

        ensemble_probs: Dict[str, Dict[str, float]] = {}
        for task_id, labels in self.label_space.items():
            mt_probs = agent_log_probs.get("MultiTaskAgent", {}).get(task_id, {})
            if mt_probs:
                ensemble_probs[task_id] = mt_probs
            else:
                uniform = 1.0 / len(labels)
                ensemble_probs[task_id] = {lbl: uniform for lbl in labels}

        sp = synth_out.predictions
        final_predictions = {
            "T1": sp.get("symptom_type", argmax_label(ensemble_probs.get("T1", {}))),
            "T2": sp.get("pathogen_class", argmax_label(ensemble_probs.get("T2", {}))),
            "T3": sp.get("disease_name", argmax_label(ensemble_probs.get("T3", {}))),
            "T4": sp.get("severity_class", argmax_label(ensemble_probs.get("T4", {}))),
            "T5": sp.get("crop_species", argmax_label(ensemble_probs.get("T5", {}))),
        }

        return ThreeAgentTrace(
            image_id=image_id,
            path=path,
            total_tokens=total_tokens,
            agent_outputs=agent_outputs,
            final_predictions=final_predictions,
            ensemble_probs=ensemble_probs,
            backtrack_count=bt,
            early_terminated=early_terminated,
            wall_time_s=time.time() - t0,
        )

    def run_batch(self, records) -> List[ThreeAgentTrace]:
        return [self.run(r.image_id, r.image_b64) for r in records]
