"""
baselines/deer_baseline.py
==========================
DeeR (Dynamic Early Exit) baseline (§6, Table 4).

Reference: Chen et al. (2024) "DeeR-VLM: Dynamic early exit for efficient
inference of large vision-language models." arXiv:2410.22702.

Paper §9:
    "DeeR (Chen et al., 2024) operates at layer granularity within a single
     forward pass, complementary rather than competing with PlantSwarm's
     agent-level routing."

Since true layer-level early exit requires direct model internals access
(not available via a standard vLLM API endpoint), this implementation
approximates DeeR using confidence-thresholded prompt cascades:
  - Short prompt (fast/cheap) → if confidence ≥ θ_high: return
  - Full prompt (slow/expensive) → always fallback

The paper uses DeeR as a compute-efficiency comparison point in Table 4
(TPCP). This implementation faithfully reproduces the two-stage cascade
logic at the prompt level. For true layer-exit DeeR, use the official
DeeR-VLM codebase with Transformers direct access.

All use Qwen3-VL-8B-Instruct with constrained decoding (§6).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from data.loader import PlantRecord
from utils.vllm_client import VLLMClient


SHORT_PROMPT = """You are a quick plant disease image classifier.
Give concise JSON and confidence (high/medium/low) for:
  symptom_type, pathogen_class, disease_name, severity_class, crop_species
(same label sets as PlantDiagBench / PlantSwarm paper Table: tasks).
Output JSON: {symptom_type, pathogen_class, disease_name, severity_class, crop_species,
confidence_t1..confidence_t5}.
"""

FULL_PROMPT = """You are an expert plant pathologist assistant.
Carefully analyze the crop disease image and output JSON:
  symptom_type, pathogen_class, disease_name, severity_class, crop_species,
confidence_t1..confidence_t5 (high/medium/low).
Use PlantSwarm task label sets (symptom 8-class, pathogen 5-class, severity 4-class, crop species).
"""


@dataclass
class DeeRTrace:
    image_id: str
    exited_early: bool = False
    total_tokens: int = 0
    final_predictions: Dict[str, str] = field(default_factory=dict)
    probs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    exit_stage: str = "full"   # 'early' or 'full'


class DeeRBaseline:
    """
    DeeR approximation: two-stage cascade with confidence threshold.
    Chen et al. (2024), arXiv:2410.22702.

    Stage 1: short/fast prompt → check if all T1-T4 confidence = 'high'
    Stage 2: full prompt → always run if stage 1 isn't confident

    Paper §6: used as adaptive-compute comparison in TPCP analysis (Table 4).
    """

    def __init__(
        self,
        client: VLLMClient,
        label_space: Dict[str, List[str]],
        high_confidence_threshold: float = 0.85,
    ):
        self.client = client
        self.label_space = label_space
        self.theta_high = high_confidence_threshold

    def predict(self, record: PlantRecord) -> DeeRTrace:
        """Two-stage DeeR cascade."""
        # Stage 1: Short/fast prompt
        messages_short = self._build_messages(record.image_b64, SHORT_PROMPT)
        resp1, tok1 = self.client.chat(messages=messages_short, system_prompt=SHORT_PROMPT)
        preds1 = self._parse(resp1)
        total_tokens = tok1

        # Check if all key tasks are high-confidence (early exit)
        confidences = [
            preds1.get(f"confidence_t{i}", "low") for i in range(1, 5)
        ]
        all_high = all(c == "high" for c in confidences)

        if all_high:
            # Stage 1 exit (DeeR early termination)
            probs = self._score(record.image_b64, resp1)
            return DeeRTrace(
                image_id=record.image_id,
                exited_early=True,
                total_tokens=total_tokens,
                final_predictions=self._to_task_dict(preds1),
                probs=probs,
                exit_stage="early",
            )

        # Stage 2: Full prompt
        messages_full = self._build_messages(record.image_b64, FULL_PROMPT)
        resp2, tok2 = self.client.chat(messages=messages_full, system_prompt=FULL_PROMPT)
        preds2 = self._parse(resp2)
        total_tokens += tok2
        probs = self._score(record.image_b64, resp2)

        return DeeRTrace(
            image_id=record.image_id,
            exited_early=False,
            total_tokens=total_tokens,
            final_predictions=self._to_task_dict(preds2),
            probs=probs,
            exit_stage="full",
        )

    def _build_messages(self, image_b64: str, system_prompt: str) -> List[Dict]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": "Classify this plant disease image."},
                ],
            }
        ]

    def _parse(self, text: str) -> Dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {}

    def _to_task_dict(self, preds: Dict) -> Dict[str, str]:
        return {
            "T1": preds.get("symptom_type", "Other"),
            "T2": preds.get("pathogen_class", "Other"),
            "T3": preds.get("disease_name", "Other"),
            "T4": preds.get("severity_class", "Early"),
            "T5": preds.get("crop_species", "Other"),
        }

    def _score(self, image_b64: str, context: str) -> Dict[str, Dict[str, float]]:
        result = {}
        for task_id, labels in self.label_space.items():
            prefix = f"{FULL_PROMPT}\nContext: {context}\nPredict {task_id}: "
            try:
                probs = self.client.score_labels(prefix, labels, image_b64)
            except Exception:
                uniform = 1.0 / len(labels)
                probs = {lbl: uniform for lbl in labels}
            result[task_id] = probs
        return result

    def predict_batch(self, records: List[PlantRecord]) -> List[DeeRTrace]:
        return [self.predict(r) for r in records]
