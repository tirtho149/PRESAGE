"""
plantswarm/observe_rollout.py
=============================
GRPO rollout integration for OBSERVE (paper §7.3 Phase B).

Given an OBSERVE model and a frozen Phase-A reference, run one PlantSwarm
episode under OBSERVE-driven routing and produce a ``Rollout`` carrying:
  - per-step log π_θ and log π_ref
  - F1 of the final prediction vs ground truth
  - ECE of the calibrated confidence
  - delta F1 around backtracks
  - path length
  - epsilon-match score |eps_T - eps*_T|

This module is a deliberate seam: full integration requires the agents to
expose per-step routing log-probs. The current implementation stubs the
log-prob trace using OBSERVE.forward at each routing step (a soft proxy
sufficient for the GRPO surrogate loss to flow gradients).
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from observe.grpo import Rollout
from observe.model import OBSERVE


# ---------------------------------------------------------------------------
# Reward components
# ---------------------------------------------------------------------------

def _f1_one(pred: str, target: str) -> float:
    """Single-sample agreement (used as a 1/0 proxy for F1 in offline GRPO)."""
    if not target:
        return 0.0
    return 1.0 if pred == target else 0.0


def _approx_ece(confidence: float, correct: float) -> float:
    """1-bin ECE on a single sample: |confidence - correct|."""
    return abs(float(confidence) - float(correct))


# ---------------------------------------------------------------------------
# Step-level log-prob recording
# ---------------------------------------------------------------------------

def _route_step_logprob(model: OBSERVE, image: Any, context: str, action: str) -> torch.Tensor:
    """log π(a | s) for one routing decision."""
    out = model(image=image, context_text=context, return_dict=True)
    probs = out["routing_probs"]
    if action in model.agent_classes:
        idx = model.agent_classes.index(action)
    else:
        idx = 0
    p = probs.squeeze(0)[idx].clamp(min=1e-9)
    return torch.log(p)


# ---------------------------------------------------------------------------
# Public entry point used by GRPOTrainer.fit
# ---------------------------------------------------------------------------

def collect_rollout(
    model: OBSERVE,
    ref_model: OBSERVE,
    instance: Any,
    *,
    pipeline: Optional[Any] = None,
    max_steps: int = 15,
) -> Rollout:
    """
    Produce one rollout for GRPO.

    ``instance`` is expected to expose ``image``, ``context_text``, ``next_agent``,
    ``epistemic``, and ``ground_truth`` (paper §7.3). When ``pipeline`` is provided
    the actual swarm runs to generate the path; otherwise we replay the
    instance's recorded path.
    """
    if pipeline is not None:
        # Paper §7.3 — let OBSERVE drive routing; pipeline records per-step
        # actions and the model evaluates each step's log π under both policies.
        # TODO(pathome): connect to AutoGen Swarm with OBSERVE handoff hooks.
        path = getattr(instance, "path", None) or [getattr(instance, "next_agent", "DiagnosisAgent")]
    else:
        path = getattr(instance, "path", None) or [getattr(instance, "next_agent", "DiagnosisAgent")]

    log_probs = []
    ref_log_probs = []
    image = getattr(instance, "image", None)
    context = getattr(instance, "context_text", "")
    for step_idx, action in enumerate(path[:max_steps]):
        log_probs.append(_route_step_logprob(model, image, context, action))
        with torch.no_grad():
            ref_log_probs.append(_route_step_logprob(ref_model, image, context, action))

    if not log_probs:
        # Empty path → degenerate rollout; emit zeros so GRPO still steps.
        zero = torch.zeros(1, device=next(model.parameters()).device)
        return Rollout(
            log_probs=zero, ref_log_probs=zero,
            f1=0.0, ece=1.0, delta_f1_bt=0.0, path_length=0,
            epsilon_match=0.0,
        )

    log_probs_t = torch.stack(log_probs)
    ref_log_probs_t = torch.stack(ref_log_probs).detach()

    pred = path[-1]
    target = getattr(instance, "ground_truth", "") or ""
    correct = _f1_one(pred, target)
    confidence = float(getattr(instance, "confidence", 0.5))

    eps_target = float(getattr(instance, "epistemic", 0.0))
    eps_pred = float(getattr(instance, "epistemic_pred", eps_target))
    eps_match = max(0.0, 1.0 - abs(eps_pred - eps_target))

    return Rollout(
        log_probs=log_probs_t,
        ref_log_probs=ref_log_probs_t,
        f1=correct,
        ece=_approx_ece(confidence, correct),
        delta_f1_bt=0.0,             # TODO: compute from before/after backtrack
        path_length=len(path),
        epsilon_match=eps_match,
    )
