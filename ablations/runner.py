"""
ablations/runner.py
====================
Factorial ablation driver (§6 RQ3, Table 3).

Runs all six ablation variants on the PlantDiagBench test split
and records Macro-F1, ECE, and TPCP for each.

Table 3 variants:
    1. Fixed Chain             (Ctx=✓, C-Gate=✗, BT=✗, Nag=5)
    2. Fixed Chain + Full Ctx  (Ctx=✓, C-Gate=✗, BT=✗, Nag=5)
    3. Free, No Conf-Gate      (Ctx=✓, C-Gate=✗, BT=✓, Nag=5)
    4. Free, No Backtrack      (Ctx=✓, C-Gate=✓, BT=✗, Nag=5)
    5. 3-Agent Swarm           (Ctx=✓, C-Gate=✓, BT=✓, Nag=3)
    6. PlantSwarm Full         (Ctx=✓, C-Gate=✓, BT=✓, Nag=5)

McNemar's test (α=0.05, Bonferroni corrected) assesses significance
on per-image correctness vs. Fixed Chain (§6).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2

from ablations.free_no_backtrack import FreeNoBacktrackAblation
from ablations.free_no_conf_gate import FreeNoConfGateAblation
from ablations.three_agent_swarm import ThreeAgentSwarmAblation
from baselines.fixed_chain import FixedChainBaseline
from baselines.fixed_chain_ctx import FixedChainCtxBaseline
from calibration.ece import compute_ece_from_probs
from data.loader import PlantRecord
from plantswarm.autogen_pipeline import AutoGenPlantSwarmPipeline
from utils.metrics import macro_f1, tpcp, bootstrap_ci, mcnemar_test
from utils.vllm_client import VLLMClient


VARIANT_NAMES = [
    "Fixed Chain",
    "Fixed Chain + Full Ctx",
    "Free, No Conf-Gate",
    "Free, No Backtrack",
    "3-Agent Swarm",
    "PlantSwarm Full",
]


def build_variants(
    client: VLLMClient,
    label_space: Dict[str, List[str]],
    Tmax: int = 15,
    confidence_weights: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Instantiate all six ablation variants (Table 3). PlantSwarm Full uses AutoGen Swarm."""
    cw = confidence_weights or {"high": 3, "medium": 2, "low": 1}
    return {
        "Fixed Chain": FixedChainBaseline(client, label_space),
        "Fixed Chain + Full Ctx": FixedChainCtxBaseline(client, label_space),
        "Free, No Conf-Gate": FreeNoConfGateAblation(client, label_space),
        "Free, No Backtrack": FreeNoBacktrackAblation(client, label_space),
        "3-Agent Swarm": ThreeAgentSwarmAblation(client, label_space),
        "PlantSwarm Full": AutoGenPlantSwarmPipeline(
            client, label_space, Tmax=Tmax, confidence_weights=cw
        ),
    }


def _extract_predictions(trace) -> Dict[str, str]:
    """Unified prediction extractor across trace types."""
    if hasattr(trace, "final_predictions"):
        return trace.final_predictions
    if hasattr(trace, "predictions"):
        return trace.predictions
    return {}


def _extract_probs(trace) -> Dict[str, Dict[str, float]]:
    if hasattr(trace, "ensemble_probs"):
        return trace.ensemble_probs
    if hasattr(trace, "probs"):
        return trace.probs
    return {}


def _extract_tokens(trace) -> int:
    if hasattr(trace, "total_tokens"):
        return trace.total_tokens
    if hasattr(trace, "tokens"):
        return trace.tokens
    return 0


def run_variant(
    variant_name: str,
    variant,
    records: List[PlantRecord],
    label_space: Dict[str, List[str]],
    task_id: str = "T1",
) -> Dict:
    """
    Run a single variant on all records and return metrics dict.

    Returns
    -------
    dict with keys: variant, macro_f1, ece, tpcp, correctness_array
    """
    task_label_col = {
        "T1": "symptom_type",
        "T2": "pathogen_class",
        "T3": "disease_name",
        "T4": "severity_class",
        "T5": "crop_species",
    }
    gt_col = task_label_col[task_id]
    labels = label_space[task_id]

    all_preds, all_gt, all_probs_max, all_correct, all_tokens = [], [], [], [], []

    for record in records:
        gt = getattr(record, gt_col, None)
        if gt is None:
            continue

        trace = variant.run(record.image_id, record.image_b64)
        preds = _extract_predictions(trace)
        probs = _extract_probs(trace)
        tokens = _extract_tokens(trace)

        pred = preds.get(task_id, labels[0])
        all_preds.append(pred)
        all_gt.append(gt)
        all_correct.append(int(pred == gt))
        all_tokens.append(tokens)

        task_probs = probs.get(task_id, {})
        all_probs_max.append(max(task_probs.values()) if task_probs else 0.5)

    # Macro-F1 with bootstrap CI (§6, §7)
    f1, (f1_lo, f1_hi) = macro_f1(all_preds, all_gt, labels, bootstrap_n=1000)

    # ECE
    probs_matrix = _build_probs_matrix(all_preds, labels)
    ece_val, _ = compute_ece_from_probs(probs_matrix, np.array(all_gt), labels)

    # TPCP (§6 / Eq. in paper): Ω̄ × N / N_correct
    tpcp_val = tpcp(all_tokens, all_correct)

    return {
        "variant": variant_name,
        f"macro_f1_{task_id}": f1,
        f"macro_f1_{task_id}_ci_lo": f1_lo,
        f"macro_f1_{task_id}_ci_hi": f1_hi,
        f"ece_{task_id}": ece_val,
        f"tpcp_{task_id}": tpcp_val,
        "correctness": np.array(all_correct),
        "n": len(all_correct),
    }


def _build_probs_matrix(
    preds: List[str], labels: List[str]
) -> np.ndarray:
    """Build (N, C) one-hot matrix from predictions for ECE fallback."""
    label_idx = {lbl: i for i, lbl in enumerate(labels)}
    N, C = len(preds), len(labels)
    mat = np.zeros((N, C))
    for i, p in enumerate(preds):
        idx = label_idx.get(p, 0)
        mat[i, idx] = 1.0
    return mat


def run_all_ablations(
    client: VLLMClient,
    records: List[PlantRecord],
    label_space: Dict[str, List[str]],
    task_id: str = "T1",
    results_dir: str = "results/",
    bonferroni_correct: bool = True,
    Tmax: int = 15,
    confidence_weights: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Run all six ablation variants. Returns a DataFrame (Table 3 equivalent).
    McNemar's test vs. Fixed Chain baseline (§6).
    """
    os.makedirs(results_dir, exist_ok=True)
    variants = build_variants(
        client, label_space, Tmax=Tmax, confidence_weights=confidence_weights
    )

    results = []
    baseline_correct = None

    for name in VARIANT_NAMES:
        print(f"Running ablation: {name} ...")
        variant = variants[name]
        result = run_variant(name, variant, records, label_space, task_id)
        results.append(result)
        if name == "Fixed Chain":
            baseline_correct = result["correctness"]

    # McNemar's test vs. Fixed Chain (§6, Bonferroni corrected)
    n_comparisons = len(VARIANT_NAMES) - 1  # all vs. Fixed Chain
    alpha = 0.05 / n_comparisons if bonferroni_correct else 0.05

    for result in results:
        if result["variant"] == "Fixed Chain":
            result["mcnemar_p"] = None
            result["significant"] = None
            continue
        p_val = mcnemar_test(baseline_correct, result["correctness"])
        result["mcnemar_p"] = p_val
        result["significant"] = p_val < alpha

    # Build summary DataFrame
    rows = []
    for r in results:
        rows.append({
            "Variant": r["variant"],
            f"Macro-F1 {task_id}": f"{r[f'macro_f1_{task_id}']:.1f} "
                                    f"[{r[f'macro_f1_{task_id}_ci_lo']:.1f},"
                                    f"{r[f'macro_f1_{task_id}_ci_hi']:.1f}]",
            f"ECE {task_id}": f"{r[f'ece_{task_id}']:.4f}",
            f"TPCP {task_id}": f"{r[f'tpcp_{task_id}']:.1f}",
            "McNemar p": r.get("mcnemar_p"),
            "Significant": r.get("significant"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "ablation_results.csv"), index=False)

    serializable = []
    for r in results:
        row = {}
        for k, v in r.items():
            if hasattr(v, "tolist"):
                row[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                row[k] = float(v) if isinstance(v, np.floating) else int(v)
            else:
                row[k] = v
        serializable.append(row)
    with open(os.path.join(results_dir, f"ablation_metrics_{task_id}.json"), "w") as f:
        json.dump(
            {"task_id": task_id, "variants": serializable},
            f,
            indent=2,
        )

    print(df.to_string(index=False))
    return df
