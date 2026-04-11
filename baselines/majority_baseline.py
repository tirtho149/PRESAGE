"""
baselines/majority_baseline.py
================================
Majority-class classifier baseline (§6, Table 4).

"Majority Class ≈28 (T1), ≈42 (T2)" (Table 4).

Predicts the most frequent class in the training distribution
for every input, regardless of image content.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import pandas as pd

from data.loader import PlantRecord


class MajorityClassBaseline:
    """
    Predicts the majority class per task, fitted on training data.

    Paper §6, Table 4: Majority Class T1 F1 ≈ 28, T2 F1 ≈ 42.
    """

    def __init__(self, label_space: Dict[str, List[str]]):
        self.label_space = label_space
        self.majority: Dict[str, str] = {}

    def fit(self, df: pd.DataFrame, label_cols: Dict[str, str]) -> "MajorityClassBaseline":
        """
        Fit majority labels from a DataFrame.

        Parameters
        ----------
        df         : training DataFrame (PlantDiagBench split)
        label_cols : mapping {task_id: column_name}
        """
        for task_id, col in label_cols.items():
            if col in df.columns:
                counts = Counter(df[col].dropna().tolist())
                self.majority[task_id] = counts.most_common(1)[0][0]
            else:
                # Fallback: first label in space
                self.majority[task_id] = self.label_space[task_id][0]
        return self

    def fit_from_records(
        self, records: List[PlantRecord]
    ) -> "MajorityClassBaseline":
        """Fit from PlantRecord list."""
        task_values: Dict[str, List[str]] = {t: [] for t in self.label_space}
        for r in records:
            if r.symptom_type:
                task_values["T1"].append(r.symptom_type)
            if r.pathogen_class:
                task_values["T2"].append(r.pathogen_class)
            if r.disease_name:
                task_values["T3"].append(r.disease_name)
            if r.severity_class:
                task_values["T4"].append(r.severity_class)
            if r.crop_species:
                task_values["T5"].append(r.crop_species)
        for task_id, vals in task_values.items():
            if vals:
                self.majority[task_id] = Counter(vals).most_common(1)[0][0]
            else:
                self.majority[task_id] = self.label_space[task_id][0]
        return self

    def predict(self, record: PlantRecord) -> Dict[str, str]:
        """Always return the majority class regardless of input image."""
        return {
            task_id: self.majority.get(task_id, self.label_space[task_id][0])
            for task_id in self.label_space
        }

    def predict_batch(self, records: List[PlantRecord]) -> List[Dict[str, str]]:
        return [self.predict(r) for r in records]

    def predict_probs(self, record: PlantRecord) -> Dict[str, Dict[str, float]]:
        """One-hot probability for majority class."""
        result = {}
        for task_id, labels in self.label_space.items():
            maj = self.majority.get(task_id, labels[0])
            result[task_id] = {lbl: (1.0 if lbl == maj else 0.0) for lbl in labels}
        return result
