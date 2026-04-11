"""
calibration/ece.py
==================
Expected Calibration Error (ECE) with B=15 equal-width bins (§3).

Reference: Guo et al. (2017) "On calibration of modern neural networks."
           ICML 2017, pages 1321–1330.

"ECE is computed with B=15 equal-width bins (Guo et al., 2017)."
"ECE is the primary metric for the systems claim." (§7)

Also provides reliability diagram data for Supplementary figures.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


def ece(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute ECE with equal-width bins (Guo et al., 2017).

    Parameters
    ----------
    confidences : shape (N,) — max predicted probability per sample
    correctness : shape (N,) — 1 if prediction is correct, else 0
    n_bins      : number of equal-width bins (default 15, §3)

    Returns
    -------
    float : ECE ∈ [0, 1]
    """
    assert len(confidences) == len(correctness), "Length mismatch"
    n = len(confidences)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        n_b = mask.sum()
        if n_b == 0:
            continue
        acc_b = correctness[mask].mean()
        conf_b = confidences[mask].mean()
        ece_val += (n_b / n) * abs(acc_b - conf_b)

    return float(ece_val)


def reliability_diagram_data(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, List[float]]:
    """
    Return bin-level data for reliability diagrams (§7 supplementary figures).

    Returns dict with keys:
        bin_centers, mean_confidence, mean_accuracy, bin_counts
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, mean_conf, mean_acc, bin_counts = [], [], [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        n_b = int(mask.sum())
        bin_centers.append(float((lo + hi) / 2))
        bin_counts.append(n_b)
        if n_b == 0:
            mean_conf.append(float((lo + hi) / 2))
            mean_acc.append(0.0)
        else:
            mean_conf.append(float(confidences[mask].mean()))
            mean_acc.append(float(correctness[mask].mean()))

    return {
        "bin_centers": bin_centers,
        "mean_confidence": mean_conf,
        "mean_accuracy": mean_acc,
        "bin_counts": bin_counts,
    }


def compute_ece_from_probs(
    probs_matrix: np.ndarray,
    true_labels: np.ndarray,
    label_list: List[str],
    n_bins: int = 15,
) -> Tuple[float, Dict]:
    """
    Convenience wrapper: compute ECE from a (N, C) probability matrix.

    Parameters
    ----------
    probs_matrix : shape (N, C) — predicted probabilities per class
    true_labels  : shape (N,)  — integer class indices OR label strings
    label_list   : list of label strings (defines column ordering)
    n_bins       : ECE bins (default 15)

    Returns
    -------
    (ece_value, reliability_dict)
    """
    # Convert string labels to indices if needed
    if isinstance(true_labels[0], str):
        label_to_idx = {lbl: i for i, lbl in enumerate(label_list)}
        true_idx = np.array([label_to_idx.get(lbl, 0) for lbl in true_labels])
    else:
        true_idx = true_labels.astype(int)

    pred_idx = probs_matrix.argmax(axis=1)
    confidences = probs_matrix.max(axis=1)
    correctness = (pred_idx == true_idx).astype(float)

    ece_val = ece(confidences, correctness, n_bins=n_bins)
    rel_diag = reliability_diagram_data(confidences, correctness, n_bins=n_bins)
    return ece_val, rel_diag
