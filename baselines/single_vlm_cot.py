"""
baselines/single_vlm_cot.py
===========================
Single VLM + Chain-of-Thought (PlantSwarm baselines).
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

from data.loader import PlantRecord
from utils.vllm_client import VLLMClient


COT_SYSTEM_PROMPT = """You are an expert plant pathologist assistant.
Reason step-by-step from the crop disease image, then output predictions.

Steps: (1) morphology (2) symptom pattern (3) pathogen class & disease (4) severity & crop.

Output after "FINAL_JSON:" ONLY valid JSON:
  symptom_type, pathogen_class, disease_name, severity_class, crop_species,
  confidence_t1..confidence_t5 (high/medium/low).

Label spaces:
  T1: Lesion / Blight / Mosaic / Wilt / Rot / Canker / Rust / Powdery mildew
  T2: Fungal / Bacterial / Viral / Nutrient deficiency / Pest damage
  T3: specific disease or Other
  T4: Healthy / Early / Moderate / Severe
  T5: crop species or Other
"""


class SingleVLMCoTBaseline:
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
                    {
                        "type": "text",
                        "text": "Reason step-by-step, then output FINAL_JSON.",
                    },
                ],
            }
        ]

        response_text, tokens = self.client.chat(
            messages=messages,
            system_prompt=COT_SYSTEM_PROMPT,
        )

        predictions = self._parse_cot_response(response_text)
        probs = self._score_all_tasks(record.image_b64, response_text)

        return {
            "predictions": predictions,
            "probs": probs,
            "tokens": tokens,
            "raw_response": response_text,
            "method": "single_vlm_cot",
        }

    def _parse_cot_response(self, text: str) -> Dict[str, str]:
        if "FINAL_JSON:" in text:
            after = text.split("FINAL_JSON:", 1)[1].strip()
            try:
                d = json.loads(after)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", after, re.DOTALL)
                d = json.loads(match.group()) if match else {}
        else:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            d = json.loads(match.group()) if match else {}
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
            prefix = f"{COT_SYSTEM_PROMPT}\nContext: {context_text}\nPredict {task_id}: "
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
