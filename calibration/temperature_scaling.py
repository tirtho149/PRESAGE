"""
calibration/temperature_scaling.py
====================================
Post-hoc temperature scaling calibration (Appendix B).

"T* is trained on the 500-image calibration split (disjoint from test)
 via arg min_T NLL(P̂/T, Ŷ). ECE reported before and after scaling for
 all conditions (Guo et al., 2017)."

Reference: Guo et al. (2017) ICML.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from typing import Dict, List, Optional, Tuple

from calibration.ece import compute_ece_from_probs


def _nll(T: float, logits: np.ndarray, true_idx: np.ndarray) -> float:
    """Negative log-likelihood after temperature scaling."""
    scaled = logits / max(T, 1e-8)
    # Numerically stable softmax
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    log_softmax = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    nll = -log_softmax[np.arange(len(true_idx)), true_idx].mean()
    return float(nll)


class TemperatureScaler:
    """
    Learns a single scalar temperature T* on a calibration split,
    then applies it to test probabilities (Appendix B).

    Usage
    -----
    scaler = TemperatureScaler()
    scaler.fit(logits_cal, true_labels_cal, label_list)
    probs_scaled = scaler.transform(logits_test)
    """

    def __init__(self):
        self.T_star: float = 1.0

    def fit(
        self,
        logits: np.ndarray,
        true_labels,
        label_list: List[str],
    ) -> "TemperatureScaler":
        """
        Find T* = arg min_T NLL(P̂/T, Ŷ) on calibration data.

        Parameters
        ----------
        logits : shape (N, C) — raw logits (pre-softmax)
        true_labels : shape (N,) — integer indices or label strings
        label_list : list of C label strings
        """
        if isinstance(true_labels[0], str):
            label_to_idx = {lbl: i for i, lbl in enumerate(label_list)}
            true_idx = np.array([label_to_idx.get(lbl, 0) for lbl in true_labels])
        else:
            true_idx = true_labels.astype(int)

        result = minimize_scalar(
            fun=lambda T: _nll(T, logits, true_idx),
            bounds=(0.01, 10.0),
            method="bounded",
        )
        self.T_star = float(result.x)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature T* and return calibrated probabilities."""
        scaled = logits / self.T_star
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def fit_transform(
        self,
        logits_cal: np.ndarray,
        true_labels_cal,
        logits_test: np.ndarray,
        label_list: List[str],
    ) -> Tuple[np.ndarray, float]:
        """
        Convenience method: fit on calibration, transform test.
        Returns (calibrated_probs_test, T_star).
        """
        self.fit(logits_cal, true_labels_cal, label_list)
        return self.transform(logits_test), self.T_star

    def report_ece(
        self,
        logits: np.ndarray,
        true_labels,
        label_list: List[str],
        n_bins: int = 15,
    ) -> Dict[str, float]:
        """Report ECE before and after temperature scaling."""
        # Before scaling
        exp_raw = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs_raw = exp_raw / exp_raw.sum(axis=1, keepdims=True)
        ece_before, _ = compute_ece_from_probs(probs_raw, true_labels, label_list, n_bins)

        # After scaling
        probs_scaled = self.transform(logits)
        ece_after, _ = compute_ece_from_probs(probs_scaled, true_labels, label_list, n_bins)

        return {
            "T_star": self.T_star,
            "ece_before": ece_before,
            "ece_after": ece_after,
        }
