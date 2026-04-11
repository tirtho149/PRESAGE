"""
baselines/multi_agent_debate.py
================================
Multi-Agent Debate baseline (§6, Table 4).

Reference: Chen et al. (2023a) "Large language models are visual reasoning
coordinators." arXiv:2310.15166.

Paper §6:
    "Debate — fixed debate structure" (Table 4).
    "Prior work falls into two camps: fixed sequential chains and fixed
     debate structures." (§1)
    "Fixed debate structures" cannot adapt routing based on confidence.

Implements a standard 2-agent debate:
    Round 1: Two independent agents predict T1–T5 separately.
    Round 2: Each agent sees the other's prediction and updates.
    Synthesis: A judge agent resolves disagreements.

All agents use Qwen3-VL-8B-Instruct with constrained decoding (§6).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from data.loader import PlantRecord
from utils.vllm_client import VLLMClient


AGENT_A_PROMPT = """You are Debater A, an expert plant pathologist assistant.
Predict all five PlantDiagBench tasks from the image.
Output JSON: {symptom_type, pathogen_class, disease_name, severity_class, crop_species,
confidence_t1..confidence_t5 (high/medium/low), reasoning (one sentence)}.
"""

AGENT_B_PROMPT = """You are Debater B, a second independent plant pathology expert.
Predict all five tasks from the image.
Output JSON: {symptom_type, pathogen_class, disease_name, severity_class, crop_species,
confidence_t1..confidence_t5 (high/medium/low), reasoning (one sentence)}.
"""

DEBATE_ROUND2_PROMPT = """You are {agent_name}. You made an initial prediction.
Debater {other_name} predicted: {other_prediction}.
Review your initial prediction in light of {other_name}'s reasoning.
Output updated JSON: {symptom_type, pathogen_class, disease_name, severity_class, crop_species,
confidence_t1..confidence_t5 (high/medium/low), changed (bool), change_reason (str)}.
"""

JUDGE_PROMPT = """You are the Judge. Two debaters have provided predictions.
Debater A final: {pred_a}
Debater B final: {pred_b}
Resolve disagreements by confidence and visual evidence.
Output final JSON: {symptom_type, pathogen_class, disease_name, severity_class, crop_species,
confidence_t1..confidence_t5 (high/medium/low), agreement_rate (float 0-1)}.
"""


@dataclass
class DebateTrace:
    image_id: str
    round1_a: Dict = field(default_factory=dict)
    round1_b: Dict = field(default_factory=dict)
    round2_a: Dict = field(default_factory=dict)
    round2_b: Dict = field(default_factory=dict)
    final: Dict = field(default_factory=dict)
    total_tokens: int = 0
    final_predictions: Dict[str, str] = field(default_factory=dict)
    probs: Dict[str, Dict[str, float]] = field(default_factory=dict)


class MultiAgentDebateBaseline:
    """
    Multi-Agent Debate with 2 debaters + 1 judge.
    Chen et al. (2023a), arXiv:2310.15166.
    Fixed topology: no confidence-gated routing.
    Paper §1: "fixed debate structure" baseline.
    """

    def __init__(
        self,
        client: VLLMClient,
        label_space: Dict[str, List[str]],
    ):
        self.client = client
        self.label_space = label_space

    def predict(self, record: PlantRecord) -> DebateTrace:
        total_tokens = 0
        img = record.image_b64

        # Round 1: independent predictions
        r1a, tok = self._call(img, AGENT_A_PROMPT, "Make your initial prediction.")
        total_tokens += tok
        r1b, tok = self._call(img, AGENT_B_PROMPT, "Make your initial prediction.")
        total_tokens += tok
        p_a1 = self._parse(r1a)
        p_b1 = self._parse(r1b)

        # Round 2: debaters see each other's predictions
        r2a_prompt = DEBATE_ROUND2_PROMPT.format(
            agent_name="Debater A", other_name="B",
            other_prediction=json.dumps(p_b1, indent=2),
        )
        r2b_prompt = DEBATE_ROUND2_PROMPT.format(
            agent_name="Debater B", other_name="A",
            other_prediction=json.dumps(p_a1, indent=2),
        )
        r2a, tok = self._call(img, r2a_prompt, f"Initial: {r1a}\nNow update.")
        total_tokens += tok
        r2b, tok = self._call(img, r2b_prompt, f"Initial: {r1b}\nNow update.")
        total_tokens += tok
        p_a2 = self._parse(r2a)
        p_b2 = self._parse(r2b)

        # Judge: resolve disagreements
        judge_prompt = JUDGE_PROMPT.format(
            pred_a=json.dumps(p_a2, indent=2),
            pred_b=json.dumps(p_b2, indent=2),
        )
        r_judge, tok = self._call(img, judge_prompt, "Resolve the debate.")
        total_tokens += tok
        p_final = self._parse(r_judge)

        final_predictions = {
            "T1": p_final.get("symptom_type", "Other"),
            "T2": p_final.get("pathogen_class", "Other"),
            "T3": p_final.get("disease_name", "Other"),
            "T4": p_final.get("severity_class", "Early"),
            "T5": p_final.get("crop_species", "Other"),
        }
        probs = self._score(img, r_judge)

        return DebateTrace(
            image_id=record.image_id,
            round1_a=p_a1, round1_b=p_b1,
            round2_a=p_a2, round2_b=p_b2,
            final=p_final,
            total_tokens=total_tokens,
            final_predictions=final_predictions,
            probs=probs,
        )

    def _call(self, image_b64: str, system: str, user_text: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        return self.client.chat(messages=messages, system_prompt=system)

    def _parse(self, text: str) -> Dict:
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

    def _score(self, image_b64: str, context: str) -> Dict[str, Dict[str, float]]:
        result = {}
        for task_id, labels in self.label_space.items():
            prefix = f"Context:\n{context}\nPredict {task_id}: "
            try:
                probs = self.client.score_labels(prefix, labels, image_b64)
            except Exception:
                uniform = 1.0 / len(labels)
                probs = {lbl: uniform for lbl in labels}
            result[task_id] = probs
        return result

    def predict_batch(self, records: List[PlantRecord]) -> List[DebateTrace]:
        return [self.predict(r) for r in records]
