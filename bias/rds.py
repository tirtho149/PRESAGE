"""
bias/rds.py
===========
Routing Disparity Score (RDS) — §8 / Eq. (4).

Eq. (4):
    RDS(g, M) = M̄_g − M̄_global

where M ∈ {path_length L, loop_rate λ, backtrack_rate β}
and g is a demographic group (gender, region, sector).

Paper §8:
    "RDS requires only routing traces — no ground-truth labels —
     enabling deployment as a real-time production audit metric."
    "Positive RDS = more deliberation than global mean." (Table 6)

T5 as negative control (§8):
    "Since T5 (funding timing) is non-visual, RDS for T5 should be near zero
     across all demographic groups. Nonzero T5 RDS would indicate baseline
     routing bias artifacts that must be subtracted from T1–T4 RDS values."

Cross-model consistency (§7):
    "Kendall's τ across backbone RDS rankings is reported;
     τ>0.70 would confirm model-agnostic routing-level bias."
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


# ---------------------------------------------------------------------------
# Core RDS computation
# ---------------------------------------------------------------------------

def compute_rds(
    traces: List[Dict],
    group_labels: List[str],
    metric: str = "path_length",
) -> Dict[str, float]:
    """
    Compute RDS(g, M) = M̄_g − M̄_global for all groups g.

    Parameters
    ----------
    traces       : list of trace dicts (with keys: path_length, loop_rate, backtrack_count)
    group_labels : demographic group per trace (same order as traces)
    metric       : one of 'path_length' (L), 'loop_rate' (λ), 'backtrack_rate' (β)

    Returns
    -------
    {group_name: RDS_value}  — positive = more deliberation than global mean
    """
    assert len(traces) == len(group_labels), "Traces and group_labels must align"

    # Extract metric values
    metric_key_map = {
        "path_length": lambda t: t.get("path_length", len(t.get("path", []))),
        "loop_rate": lambda t: t.get("loop_rate", 0.0),
        "backtrack_rate": lambda t: float(t.get("backtrack_count", 0) > 0),
    }
    if metric not in metric_key_map:
        raise ValueError(f"Unknown metric '{metric}'. Use: path_length, loop_rate, backtrack_rate")

    extract = metric_key_map[metric]
    values = np.array([extract(t) for t in traces], dtype=float)
    global_mean = values.mean()

    groups = np.array(group_labels)
    rds = {}
    for g in sorted(set(group_labels)):
        mask = groups == g
        g_mean = values[mask].mean() if mask.sum() > 0 else global_mean
        rds[g] = float(g_mean - global_mean)

    return rds


def compute_rds_table(
    traces: List[Dict],
    group_labels: List[str],
    accuracy_by_group: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute full RDS table for all three metrics (Table 6).

    Parameters
    ----------
    traces          : trace dicts
    group_labels    : demographic group per trace
    accuracy_by_group : optional {group: accuracy} for ΔAcc column

    Returns
    -------
    DataFrame with columns: Group, RDS(L), RDS(λ), RDS(β), ΔAcc
    """
    rds_L = compute_rds(traces, group_labels, metric="path_length")
    rds_lam = compute_rds(traces, group_labels, metric="loop_rate")
    rds_beta = compute_rds(traces, group_labels, metric="backtrack_rate")

    groups = sorted(set(group_labels))
    rows = []
    for g in groups:
        row = {
            "Group": g,
            "RDS(L)": rds_L.get(g, 0.0),
            "RDS(λ)": rds_lam.get(g, 0.0),
            "RDS(β)": rds_beta.get(g, 0.0),
        }
        if accuracy_by_group:
            global_acc = np.mean(list(accuracy_by_group.values()))
            row["Δacc"] = accuracy_by_group.get(g, global_acc) - global_acc
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RDS–accuracy correlation (§8)
# ---------------------------------------------------------------------------

def rds_accuracy_correlation(
    rds_values: Dict[str, float],
    accuracy_gaps: Dict[str, float],
) -> Tuple[float, float]:
    """
    Spearman ρ(RDS(g, L), Δacc(g)) across groups (§8).
    Paper: "ρ ≈ −0.68 across groups, confirmed after confounder control."

    Parameters
    ----------
    rds_values    : {group: RDS(L) value}
    accuracy_gaps : {group: accuracy − global_mean_accuracy}

    Returns
    -------
    (spearman_rho, p_value)
    """
    common = sorted(set(rds_values) & set(accuracy_gaps))
    if len(common) < 3:
        return 0.0, 1.0
    rds_arr = np.array([rds_values[g] for g in common])
    acc_arr = np.array([accuracy_gaps[g] for g in common])
    rho, p = spearmanr(rds_arr, acc_arr)
    return float(rho), float(p)


# ---------------------------------------------------------------------------
# T5 negative control (§8)
# ---------------------------------------------------------------------------

def t5_negative_control_check(
    traces_t5: List[Dict],
    group_labels: List[str],
    threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Check that T5 (non-visual) RDS ≈ 0 across all groups (§8).
    "Nonzero T5 RDS would indicate baseline routing bias artifacts."

    Returns dict with: rds_table, all_near_zero (bool), max_abs_rds
    """
    rds_L = compute_rds(traces_t5, group_labels, metric="path_length")
    max_abs = max(abs(v) for v in rds_L.values()) if rds_L else 0.0
    all_near_zero = max_abs < threshold

    return {
        "rds_t5": rds_L,
        "max_abs_rds": max_abs,
        "all_near_zero": all_near_zero,
        "threshold": threshold,
        "interpretation": (
            "T5 RDS ≈ 0: no baseline routing artifact detected."
            if all_near_zero else
            f"T5 RDS exceeds threshold {threshold}: baseline routing artifact present. "
            f"Subtract T5 RDS from T1–T4 values."
        ),
    }


# ---------------------------------------------------------------------------
# Cross-model Kendall's τ (§7)
# ---------------------------------------------------------------------------

def cross_model_rds_consistency(
    rds_model_a: Dict[str, float],
    rds_model_b: Dict[str, float],
) -> Tuple[float, float]:
    """
    Kendall's τ between RDS group rankings across two backbone models (§7).
    "τ>0.70 would confirm model-agnostic routing-level bias."

    Parameters
    ----------
    rds_model_a, rds_model_b : {group: RDS(L) value} for two models

    Returns
    -------
    (kendall_tau, p_value)
    """
    common = sorted(set(rds_model_a) & set(rds_model_b))
    if len(common) < 3:
        return 0.0, 1.0
    x = np.array([rds_model_a[g] for g in common])
    y = np.array([rds_model_b[g] for g in common])
    tau, p = kendalltau(x, y)
    return float(tau), float(p)
