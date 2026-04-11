"""
utils/sequence_entropy.py
=========================
Token- and sequence-level entropy from model log-probabilities (predictive entropy).

Implements the entropy field, gradient, dispersion, and targeted disease-token
uncertainty used for entropy-driven routing (Algorithm ``entropy_swarm``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _log_softmax_from_logprobs(logprob_items: List[Tuple[str, float]]) -> Dict[str, float]:
    """Stable softmax over (token, logprob) pairs (natural log)."""
    if not logprob_items:
        return {}
    max_lp = max(lp for _, lp in logprob_items)
    exps = {tok: math.exp(lp - max_lp) for tok, lp in logprob_items}
    s = sum(exps.values())
    if s <= 0:
        n = len(exps)
        return {tok: 1.0 / n for tok in exps}
    return {tok: v / s for tok, v in exps.items()}


def entropy_from_distribution(probs: Dict[str, float]) -> float:
    """Shannon entropy (nats): -sum p log p."""
    h = 0.0
    for p in probs.values():
        if p > 0:
            h -= p * math.log(p)
    return float(h)


def token_entropy_from_openai_token_item(item: Dict[str, Any]) -> float:
    """
    Per-token predictive entropy from one OpenAI/vLLM ``logprobs.content[]`` entry.

    Uses the selected token logprob plus ``top_logprobs`` to approximate the full
    vocabulary distribution (mass outside top-k is lumped as residual).
    """
    lp_sel = item.get("logprob")
    tops = item.get("top_logprobs") or []
    pairs: List[Tuple[str, float]] = []
    if lp_sel is not None:
        tok = item.get("token", "")
        pairs.append((str(tok), float(lp_sel)))
    for alt in tops:
        if isinstance(alt, dict):
            t = alt.get("token", "")
            lp = alt.get("logprob")
            if lp is not None:
                pairs.append((str(t), float(lp)))
    if not pairs:
        return 0.0
    # De-duplicate by token string, keep max logprob
    best: Dict[str, float] = {}
    for tok, lp in pairs:
        if tok not in best or lp > best[tok]:
            best[tok] = lp
    merged = list(best.items())
    probs = _log_softmax_from_logprobs(merged)
    # Residual mass for unseen mass (Laplace-style) if we only have top-k
    p_sum = sum(probs.values())
    if p_sum < 0.999:
        probs["_residual"] = max(1.0 - p_sum, 1e-12)
    return entropy_from_distribution(probs)


def sequence_mean_entropy(token_entropies: Sequence[float]) -> float:
    """H_t = (1/N) sum_i h_i^(t)."""
    if not token_entropies:
        return 0.0
    return float(np.mean(np.asarray(token_entropies, dtype=np.float64)))


def entropy_dispersion(token_entropies: Sequence[float]) -> float:
    """D_t = Var(h^(t))."""
    if len(token_entropies) < 2:
        return 0.0
    return float(np.var(np.asarray(token_entropies, dtype=np.float64), ddof=0))


def entropy_gradient_G(H_sequence: Sequence[float]) -> float:
    """
    G_t = (1/(t-1)) sum_{k=2}^{t} (H_k - H_{k-1}) = (H_t - H_1) / (t-1) for t >= 2.
    For t < 2, returns 0.
    """
    hs = list(H_sequence)
    t = len(hs)
    if t < 2:
        return 0.0
    return float((hs[-1] - hs[0]) / (t - 1))


def targeted_disease_entropy(
    token_entropies: Sequence[float],
    message: str,
    disease_label: str,
    token_strings: Optional[Sequence[str]] = None,
) -> Optional[float]:
    """
    H_t^(disease): entropy at tokens overlapping the predicted disease string (heuristic).

    If ``token_strings`` is provided with same length as ``token_entropies``, uses
    mean h_i over tokens whose decoded text appears in ``disease_label`` (case-insensitive).
    Otherwise falls back to sequence mean entropy.
    """
    if not token_entropies:
        return None
    dl = (disease_label or "").strip().lower()
    if not dl:
        return sequence_mean_entropy(token_entropies)
    if token_strings is not None and len(token_strings) == len(token_entropies):
        hits = []
        for h, ts in zip(token_entropies, token_strings):
            ts_l = str(ts).lower() if ts else ""
            if not ts_l:
                continue
            if ts_l in dl or dl in ts_l or any(
                part in ts_l for part in dl.replace(",", " ").split() if len(part) > 3
            ):
                hits.append(h)
        if hits:
            return float(np.mean(hits))
    return sequence_mean_entropy(token_entropies)


def pds_by_class(
    disease_labels: Sequence[str],
    disease_entropies: Sequence[float],
) -> Dict[str, float]:
    """
    Pathogen Difficulty Score: PDS(c) = E[H^(disease)(x) | x in class c].
    """
    buckets: Dict[str, List[float]] = {}
    for c, h in zip(disease_labels, disease_entropies):
        key = str(c).strip() or "Unknown"
        buckets.setdefault(key, []).append(float(h))
    return {c: float(np.mean(vs)) for c, vs in buckets.items() if vs}
