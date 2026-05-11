"""
calibration/ensemble.py
========================
Weighted confidence ensemble (§3 / Eq. 3).

Eq. (3):
    P̂(Y=c | X, τ, T_k) = Σ_{j∈A(T_k)} v(κ_j) · p_c^(j)
                          ─────────────────────────────────
                                    Σ_j v(κ_j)

where v(H)=3, v(M)=2, v(L)=1 (§3).

DiagnosisAgent is excluded (§3):
    "excluded from the calibration ensemble as it is not an independent observer."

A(T_k) = task-scoped upstream agents active for task T_k (PlantSwarm Table 1).
"""

from __future__ import annotations

from typing import Dict, List, Optional

CONFIDENCE_WEIGHTS = {"high": 3, "medium": 2, "low": 1}

# Task → agents that cover it (excluding DiagnosisAgent)
TASK_AGENT_SCOPE: Dict[str, List[str]] = {
    "T1": ["SymptomAgent"],
    "T2": ["PathogenAgent"],
    "T3": ["PathogenAgent"],
    "T4": ["SeverityAgent"],
    "T5": ["SeverityAgent"],
}


def ensemble_probabilities(
    agent_log_probs: Dict[str, Dict[str, Dict[str, float]]],
    agent_confidences: Dict[str, str],
    task_id: str,
    label_list: List[str],
    confidence_weights: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """
    Compute weighted ensemble probabilities for task_id (Eq. 3).

    Parameters
    ----------
    agent_log_probs : {agent_name: {task_id: {label: prob}}}
    agent_confidences : {agent_name: 'high'|'medium'|'low'}
    task_id : one of T1..T5
    label_list : complete label vocabulary for task_id
    confidence_weights : override default {H:3, M:2, L:1}

    Returns
    -------
    {label: ensemble_probability}
    """
    weights = confidence_weights or CONFIDENCE_WEIGHTS
    in_scope = TASK_AGENT_SCOPE.get(task_id, [])

    weighted_probs: Dict[str, float] = {lbl: 0.0 for lbl in label_list}
    total_weight = 0.0

    for agent_name in in_scope:
        if agent_name not in agent_log_probs:
            continue
        task_probs = agent_log_probs[agent_name].get(task_id, {})
        if not task_probs:
            continue
        conf = agent_confidences.get(agent_name, "medium")
        w = weights.get(conf, 2)
        total_weight += w
        for lbl in label_list:
            weighted_probs[lbl] += w * task_probs.get(lbl, 0.0)

    if total_weight == 0.0:
        uniform = 1.0 / len(label_list) if label_list else 1.0
        return {lbl: uniform for lbl in label_list}

    return {lbl: v / total_weight for lbl, v in weighted_probs.items()}


def argmax_label(probs: Dict[str, float]) -> str:
    """Return the label with highest ensemble probability, or "" if no
    probabilities are available (e.g., task missing from the ensemble or
    label_list empty)."""
    if not probs:
        return ""
    return max(probs, key=lambda k: probs[k])
