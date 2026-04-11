"""
baselines/single_vlm.py
=======================
Single VLM direct inference (PlantSwarm baselines).
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

from data.loader import PlantRecord
from utils.vllm_client import VLLMClient


DIRECT_SYSTEM_PROMPT = """You are an expert plant pathologist assistant.
Given a crop disease image, predict all five tasks using ONLY these label sets:
  T1 symptom_type: Lesion / Blight / Mosaic / Wilt / Rot / Canker / Rust / Powdery mildew
  T2 pathogen_class: Fungal / Bacterial / Viral / Nutrient deficiency / Pest damage
  T3 disease_name: use the most specific disease name from training vocabulary or 'Other'
  T4 severity_class: Healthy / Early / Moderate / Severe
  T5 crop_species: tomato-scale crop name or 'Other'

Output ONLY valid JSON with keys: symptom_type, pathogen_class, disease_name,
severity_class, crop_species,
confidence_t1, confidence_t2, confidence_t3, confidence_t4, confidence_t5
(each confidence: high / medium / low).
"""


class SingleVLMBaseline:
    def __init__(
        self,
        client: VLLMClient,
        label_space: Dict[str, List[str]],
    ):
        self.client = client
        self.label_space = label_space

    def predict(self, record: PlantRecord) -> Dict:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{record.image_b64}"},
                    },
                    {"type": "text", "text": "Analyze this image and predict all five tasks."},
                ],
            }
        ]

        response_text, tokens = self.client.chat(
            messages=messages,
            system_prompt=DIRECT_SYSTEM_PROMPT,
        )

        predictions = self._parse_response(response_text)
        probs = self._score_all_tasks(record.image_b64, response_text)

        return {
            "predictions": predictions,
            "probs": probs,
            "tokens": tokens,
            "raw_response": response_text,
            "method": "single_vlm_direct",
        }

    def _parse_response(self, text: str) -> Dict[str, str]:
        text = text.split("TERMINATE")[0].strip()
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    d = json.loads(match.group())
                except json.JSONDecodeError:
                    d = {}
            else:
                d = {}
        return {
            "T1": d.get("symptom_type", "Other"),
            "T2": d.get("pathogen_class", "Other"),
            "T3": d.get("disease_name", "Other"),
            "T4": d.get("severity_class", "Early"),
            "T5": d.get("crop_species", "Other"),
        }

    def _score_all_tasks(self, image_b64: str, context_text: str) -> Dict[str, Dict[str, float]]:
        result = {}
        for task_id, labels in self.label_space.items():
            prefix = f"{DIRECT_SYSTEM_PROMPT}\nContext: {context_text}\nPredict {task_id}: "
            try:
                probs = self.client.score_labels(
                    prompt_prefix=prefix,
                    label_list=labels,
                    image_b64=image_b64,
                )
            except Exception:
                uniform = 1.0 / len(labels)
                probs = {lbl: uniform for lbl in labels}
            result[task_id] = probs
        return result

    def predict_batch(self, records: List[PlantRecord]) -> List[Dict]:
        return [self.predict(r) for r in records]
