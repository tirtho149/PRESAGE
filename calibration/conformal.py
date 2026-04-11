"""
calibration/conformal.py
=========================
Split Conformal Prediction (§3 / Appendix B).

"Split Conformal prediction (Deutschmann et al., 2025) coverage is reported
 at α=0.1 as a complementary distribution-free measure."

"Prediction sets contain all labels c for which
 P̂(Y=c | X, τ, T_k) ≥ q̂_{1−α}, the (1−α)-quantile of non-conformity
 scores on the calibration split."
Target coverage: ≥ 0.90 (α=0.1).

Reference: Deutschmann et al. (2025), arXiv:2402.01464.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Set, Tuple


def nonconformity_score(probs: np.ndarray, true_idx: int) -> float:
    """
    Non-conformity score = 1 - P̂(Y=true_class).
    Lower score = more conforming to ground truth.
    """
    return float(1.0 - probs[true_idx])


class SplitConformalPredictor:
    """
    Split conformal predictor for multi-class classification (§3, Appendix B).

    Usage
    -----
    predictor = SplitConformalPredictor(alpha=0.1)
    predictor.calibrate(probs_cal, true_labels_cal, label_list)
    prediction_sets = predictor.predict(probs_test)
    coverage = predictor.empirical_coverage(prediction_sets, true_labels_test, label_list)
    """

    def __init__(self, alpha: float = 0.1):
        """
        Parameters
        ----------
        alpha : miscoverage level; target coverage = 1 - alpha = 0.90 (§3)
        """
        self.alpha = alpha
        self.q_hat: Optional[float] = None

    def calibrate(
        self,
        probs_cal: np.ndarray,
        true_labels,
        label_list: List[str],
    ) -> float:
        """
        Compute q̂_{1-α} from calibration non-conformity scores (Appendix B).

        Parameters
        ----------
        probs_cal   : shape (N_cal, C) — predicted probabilities on calibration split
        true_labels : shape (N_cal,)   — integer indices or label strings
        label_list  : C label strings

        Returns
        -------
        q_hat : float — the (1-α) quantile threshold
        """
        if isinstance(true_labels[0], str):
            label_to_idx = {lbl: i for i, lbl in enumerate(label_list)}
            true_idx = np.array([label_to_idx.get(lbl, 0) for lbl in true_labels])
        else:
            true_idx = true_labels.astype(int)

        scores = np.array([
            nonconformity_score(probs_cal[i], true_idx[i])
            for i in range(len(true_idx))
        ])

        n_cal = len(scores)
        # Finite-sample corrected quantile: ceil((n+1)(1-α))/n
        level = np.ceil((n_cal + 1) * (1.0 - self.alpha)) / n_cal
        level = min(level, 1.0)
        self.q_hat = float(np.quantile(scores, level))
        return self.q_hat

    def predict(self, probs_test: np.ndarray) -> List[List[int]]:
        """
        Return prediction sets for test samples.
        Set contains indices c where P̂(Y=c) ≥ (1 - q̂).

        Parameters
        ----------
        probs_test : shape (N_test, C)

        Returns
        -------
        List of lists of class indices (one per test sample)
        """
        if self.q_hat is None:
            raise RuntimeError("Must call calibrate() before predict().")

        threshold = 1.0 - self.q_hat
        prediction_sets = []
        for probs in probs_test:
            s = [c for c, p in enumerate(probs) if p >= threshold]
            if not s:
                s = [int(np.argmax(probs))]  # always include top-1
            prediction_sets.append(s)
        return prediction_sets

    def empirical_coverage(
        self,
        prediction_sets: List[List[int]],
        true_labels,
        label_list: List[str],
    ) -> float:
        """
        Compute empirical marginal coverage = P(Y ∈ C(X)).
        Should be ≥ 0.90 for α=0.10.
        """
        if isinstance(true_labels[0], str):
            label_to_idx = {lbl: i for i, lbl in enumerate(label_list)}
            true_idx = [label_to_idx.get(lbl, 0) for lbl in true_labels]
        else:
            true_idx = list(true_labels)

        covered = sum(
            1 for pset, ti in zip(prediction_sets, true_idx) if ti in pset
        )
        return covered / len(true_idx)

    def set_size_stats(self, prediction_sets: List[List[int]]) -> Dict[str, float]:
        """Return mean/median/max prediction set size (efficiency metric)."""
        sizes = [len(s) for s in prediction_sets]
        return {
            "mean_set_size": float(np.mean(sizes)),
            "median_set_size": float(np.median(sizes)),
            "max_set_size": float(np.max(sizes)),
        }
