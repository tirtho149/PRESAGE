"""
baselines/random_baseline.py
=============================
Random classifier baseline (§6, Table 4).

"Baselines: Random, Majority Class, Single VLM (direct and +CoT),
 adaptive-compute baselines DeeR (Chen et al., 2024) and
 Multi-Agent Debate (Chen et al., 2023a)."

Random assigns a uniformly random label from the valid label set.
Expected Macro-F1 ≈ 1/|C| per task (Table 4: T1≈16.7, T2≈14.3).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from data.loader import PlantRecord


class RandomBaseline:
    """
    Uniform random classifier over the valid label space.

    Paper §6, Table 4:
        T1 F1 ≈ 16.7  (6 classes)
        T2 F1 ≈ 14.3  (7 classes)
    """

    def __init__(self, label_space: Dict[str, List[str]], seed: int = 42):
        self.label_space = label_space
        self.rng = random.Random(seed)

    def predict(self, record: PlantRecord) -> Dict[str, str]:
        """Return random label for each task. Image is ignored."""
        return {
            task_id: self.rng.choice(labels)
            for task_id, labels in self.label_space.items()
        }

    def predict_batch(self, records: List[PlantRecord]) -> List[Dict[str, str]]:
        return [self.predict(r) for r in records]

    def predict_probs(self, record: PlantRecord) -> Dict[str, Dict[str, float]]:
        """Uniform probability distribution over labels (for ECE computation)."""
        return {
            task_id: {lbl: 1.0 / len(labels) for lbl in labels}
            for task_id, labels in self.label_space.items()
        }
