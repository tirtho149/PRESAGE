"""
utils/metrics.py
================
Evaluation metrics for PlantSwarm (§6, §7).

Metrics:
    - Macro-F1 with bootstrap CI (bootstrap n=1000, §7)
    - TPCP: Tokens-per-correct-prediction (§6 RQ2)
    - McNemar's test on per-image correctness (§6, Bonferroni corrected)
    - κ calibration: correctness stratified by confidence level (§7)

Paper §6:
    "TPCP = Ω̄ × N_images / N_correct"
    "McNemar's test (α=0.05, Bonferroni corrected) assesses significance
     on per-image correctness."
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# Macro-F1 with bootstrap CI
# ---------------------------------------------------------------------------

def macro_f1(
    preds: List[str],
    targets: List[str],
    labels: List[str],
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute Macro-F1 with 95% bootstrap confidence interval.
    §7: "Macro-F1 (mean±95% CI, bootstrap n=1000)"

    Returns
    -------
    (f1_mean, (ci_lo, ci_hi))  — multiplied by 100 (percentage points)
    """
    preds_arr = np.array(preds)
    targets_arr = np.array(targets)

    f1_mean = f1_score(targets_arr, preds_arr, labels=labels,
                       average="macro", zero_division=0) * 100.0

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    n = len(preds)
    boot_scores = []
    for _ in range(bootstrap_n):
        idx = rng.integers(0, n, size=n)
        score = f1_score(
            targets_arr[idx], preds_arr[idx],
            labels=labels, average="macro", zero_division=0
        ) * 100.0
        boot_scores.append(score)

    ci_lo = float(np.percentile(boot_scores, 2.5))
    ci_hi = float(np.percentile(boot_scores, 97.5))
    return float(f1_mean), (ci_lo, ci_hi)


# ---------------------------------------------------------------------------
# TPCP: Tokens-per-correct-prediction
# ---------------------------------------------------------------------------

def tpcp(tokens_per_image: List[int], correctness: List[int]) -> float:
    """
    Tokens-Per-Correct-Prediction (§6 RQ2):
        TPCP = Ω̄ × N_images / N_correct

    Lower TPCP = more compute-efficient correct predictions.

    Parameters
    ----------
    tokens_per_image : list of token counts per image
    correctness      : list of 0/1 per image (1 = correct)

    Returns
    -------
    float : TPCP (↓ is better)
    """
    n = len(tokens_per_image)
    n_correct = sum(correctness)
    if n_correct == 0:
        return float("inf")
    omega_bar = np.mean(tokens_per_image)
    return float(omega_bar * n / n_correct)


# ---------------------------------------------------------------------------
# Bootstrap CI (generic)
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: List[float],
    stat_fn=np.mean,
    n: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Return (2.5th, 97.5th) percentile bootstrap CI for stat_fn(values)."""
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    boot = [stat_fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------

def mcnemar_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    continuity_correction: bool = True,
) -> float:
    """
    McNemar's test on per-image correctness arrays (§6).
    Returns p-value (two-tailed).

    H0: P(A correct, B wrong) = P(A wrong, B correct)
    """
    correct_a = np.asarray(correct_a, dtype=int)
    correct_b = np.asarray(correct_b, dtype=int)
    assert len(correct_a) == len(correct_b), "Arrays must be same length"

    # Discordant pairs
    b = int(((correct_a == 1) & (correct_b == 0)).sum())  # A right, B wrong
    c = int(((correct_a == 0) & (correct_b == 1)).sum())  # A wrong, B right

    if b + c == 0:
        return 1.0  # No discordant pairs → no evidence of difference

    if continuity_correction:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        stat = (b - c) ** 2 / (b + c)

    from scipy.stats import chi2 as _chi2
    p_val = 1.0 - _chi2.cdf(stat, df=1)
    return float(p_val)


# ---------------------------------------------------------------------------
# κ calibration analysis (§7)
# ---------------------------------------------------------------------------

def kappa_calibration_report(
    confidences: List[str],  # 'high' | 'medium' | 'low'
    correctness: List[int],
) -> Dict[str, Dict]:
    """
    Per-agent κ calibration: correctness stratified by κ ∈ {H, M, L}.

    §7: "We report per-agent correctness stratified by κ (H/M/L) and Spearman ρ
         between ordinal κ and token log-probability magnitude."

    Returns dict: {level: {count, accuracy, ci_lo, ci_hi}}
    Failure of monotonic ordering acc(H) > acc(M) > acc(L) reported as
    agent-level miscalibration.
    """
    from scipy.stats import spearmanr

    levels = ["high", "medium", "low"]
    ordinal = {"high": 2, "medium": 1, "low": 0}

    report = {}
    for level in levels:
        mask = [c == level for c in confidences]
        subset = [correctness[i] for i, m in enumerate(mask) if m]
        n = len(subset)
        if n == 0:
            report[level] = {"count": 0, "accuracy": None, "ci_lo": None, "ci_hi": None}
            continue
        acc = np.mean(subset)
        ci = bootstrap_ci(subset, stat_fn=np.mean)
        report[level] = {"count": n, "accuracy": float(acc), "ci_lo": ci[0], "ci_hi": ci[1]}

    # Spearman ρ between ordinal κ and correctness
    kappa_ordinal = np.array([ordinal.get(c, 1) for c in confidences])
    correct_arr = np.array(correctness)
    if len(set(kappa_ordinal)) > 1:
        rho, p = spearmanr(kappa_ordinal, correct_arr)
    else:
        rho, p = 0.0, 1.0

    # Check monotonic ordering
    accs = [report[l].get("accuracy") for l in levels]
    monotonic = all(
        (a is None or b is None or a >= b)
        for a, b in zip(accs, accs[1:])
    )

    report["spearman_rho"] = float(rho)
    report["spearman_p"] = float(p)
    report["monotonic_ordering"] = monotonic
    report["miscalibrated"] = not monotonic
    return report


# ---------------------------------------------------------------------------
# Accuracy per group (for bias analysis helpers)
# ---------------------------------------------------------------------------

def accuracy_by_group(
    preds: List[str],
    targets: List[str],
    groups: List[str],
) -> Dict[str, float]:
    """Return per-group accuracy dict."""
    group_set = set(groups)
    result = {}
    for g in group_set:
        mask = [grp == g for grp in groups]
        p_g = [preds[i] for i, m in enumerate(mask) if m]
        t_g = [targets[i] for i, m in enumerate(mask) if m]
        if not p_g:
            result[g] = 0.0
        else:
            result[g] = float(np.mean([int(p == t) for p, t in zip(p_g, t_g)]))
    return result
