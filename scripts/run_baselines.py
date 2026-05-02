"""
scripts/run_baselines.py
=========================
Run all baselines defined in §6 / Table 4.

Baselines:
    1. Random
    2. Majority Class
    3. Single VLM (direct)
    4. Single VLM + CoT
    5. DeeR (Chen et al., 2024)
    6. Fixed Chain (Algorithm 2)
    7. Fixed Chain + Full Ctx
    8. Multi-Agent Debate (Chen et al., 2023a)

Usage:
    python scripts/run_baselines.py --config configs/default.yaml [--subset N]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from tqdm import tqdm

from baselines.random_baseline import RandomBaseline
from baselines.majority_baseline import MajorityClassBaseline
from baselines.single_vlm import SingleVLMBaseline
from baselines.single_vlm_cot import SingleVLMCoTBaseline
from baselines.deer_baseline import DeeRBaseline
from baselines.fixed_chain import FixedChainBaseline
from baselines.fixed_chain_ctx import FixedChainCtxBaseline
from baselines.multi_agent_debate import MultiAgentDebateBaseline
from calibration.ece import compute_ece_from_probs
from data.loader import PlantDiagBenchLoader
from utils.metrics import macro_f1, tpcp, mcnemar_test
from utils.vllm_client import VLLMClient, configure_vllm_client_from_yaml, validate_model_server_matches_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--baselines", nargs="+", default=None,
                        help="Subset of baselines to run (default: all)")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _extract_task_pred(result, task_id: str, labels) -> str:
    """Unified prediction extractor across baseline result types."""
    if hasattr(result, "final_predictions"):
        return result.final_predictions.get(task_id, labels[0])
    if isinstance(result, dict):
        preds = result.get("predictions", {})
        if isinstance(preds, dict):
            key_map = {
                "T1": "symptom_type",
                "T2": "pathogen_class",
                "T3": "disease_name",
                "T4": "severity_class",
                "T5": "crop_species",
            }
            return preds.get(task_id, preds.get(key_map.get(task_id, task_id), labels[0]))
    return labels[0]


def _extract_task_probs(result, task_id: str, labels):
    """Unified probs extractor."""
    if hasattr(result, "probs"):
        return result.probs.get(task_id, {})
    if isinstance(result, dict):
        probs = result.get("probs", {})
        return probs.get(task_id, {})
    return {}


def _extract_tokens(result) -> int:
    if hasattr(result, "total_tokens"):
        return result.total_tokens
    if isinstance(result, dict):
        return result.get("tokens", 0)
    return 0


def run_baseline(name, baseline, records, label_space, task_id="T1"):
    """Run one baseline on all records; return metrics dict."""
    labels = label_space[task_id]
    gt_col = {
        "T1": "symptom_type",
        "T2": "pathogen_class",
        "T3": "disease_name",
        "T4": "severity_class",
        "T5": "crop_species",
    }[task_id]

    preds, gts, tokens_list, correct_list, probs_list = [], [], [], [], []

    for record in tqdm(records, desc=name, leave=False):
        gt = getattr(record, gt_col, None)
        if gt is None:
            continue

        # Handle non-VLM baselines
        if hasattr(baseline, "predict_probs"):
            pred_dict = baseline.predict(record)
            probs = baseline.predict_probs(record).get(task_id, {})
            pred = pred_dict.get(task_id, labels[0])
            tok = 0
        else:
            result = baseline.predict(record)
            pred = _extract_task_pred(result, task_id, labels)
            probs = _extract_task_probs(result, task_id, labels)
            tok = _extract_tokens(result)

        preds.append(pred)
        gts.append(gt)
        tokens_list.append(tok)
        correct_list.append(int(pred == gt))
        probs_list.append([probs.get(lbl, 1.0 / len(labels)) for lbl in labels])

    if not preds:
        return None

    f1, (ci_lo, ci_hi) = macro_f1(preds, gts, labels, bootstrap_n=1000)
    tpcp_val = tpcp(tokens_list, correct_list)
    probs_matrix = np.array(probs_list)
    ece_val, _ = compute_ece_from_probs(probs_matrix, np.array(gts), labels)

    return {
        "name": name,
        f"macro_f1_{task_id}": f1,
        f"macro_f1_{task_id}_ci": [ci_lo, ci_hi],
        f"ece_{task_id}": ece_val,
        f"tpcp_{task_id}": tpcp_val,
        "correctness": np.array(correct_list),
        "n": len(preds),
        "n_correct": sum(correct_list),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)

    results_dir = args.output_dir or cfg["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    print("Loading PlantDiagBench...")
    loader = PlantDiagBenchLoader(cfg["data"], split="test")
    label_space = loader.label_space
    records = list(loader)
    if args.subset:
        records = records[:args.subset]
    print(f"  {len(records)} test images")

    client = VLLMClient(
        base_url=cfg["model"]["vllm_base_url"],
        model=cfg["model"]["backbone"],
        temperature=cfg["model"]["temperature"],
        seed=cfg["model"]["seed"],
        max_new_tokens=cfg["model"]["max_new_tokens"],
    )
    configure_vllm_client_from_yaml(client, cfg.get("model"), orchestrator="autogen_swarm")
    validate_model_server_matches_config(cfg)

    # Instantiate baselines
    baselines = {
        "Random": RandomBaseline(label_space, seed=cfg["model"]["seed"]),
        "Majority Class": MajorityClassBaseline(label_space),
        "Single VLM": SingleVLMBaseline(client, label_space),
        "Single VLM+CoT": SingleVLMCoTBaseline(client, label_space),
        "DeeR": DeeRBaseline(client, label_space),
        "Fixed Chain": FixedChainBaseline(client, label_space),
        "Fixed Chain+Ctx": FixedChainCtxBaseline(client, label_space),
        "Debate": MultiAgentDebateBaseline(client, label_space),
    }

    # Fit majority baseline
    baselines["Majority Class"].fit_from_records(records)

    if args.baselines:
        baselines = {k: v for k, v in baselines.items() if k in args.baselines}

    # Run all baselines across all tasks
    all_results = {}
    fixed_chain_correct = {}
    task_ids = ["T1", "T2", "T3", "T4", "T5"]

    for name, baseline in baselines.items():
        print(f"\nRunning baseline: {name}")
        baseline_results = {}

        for task_id in task_ids:
            result = run_baseline(name, baseline, records, label_space, task_id=task_id)
            if result is None:
                continue
            baseline_results[task_id] = result

            if name == "Fixed Chain":
                if task_id not in fixed_chain_correct:
                    fixed_chain_correct[task_id] = result["correctness"]

            f1_key = f"macro_f1_{task_id}"
            ci_key = f"macro_f1_{task_id}_ci"
            ece_key = f"ece_{task_id}"
            tpcp_key = f"tpcp_{task_id}"

            if f1_key in result:
                print(f"  {task_id}: F1={result[f1_key]:.1f} "
                      f"[{result[ci_key][0]:.1f},{result[ci_key][1]:.1f}] "
                      f"ECE={result[ece_key]:.4f} TPCP={result[tpcp_key]:.1f}")

        if baseline_results:
            all_results[name] = baseline_results

    # McNemar's test vs. Fixed Chain (§6, Bonferroni corrected)
    if fixed_chain_correct:
        n_comparisons = max(1, len(all_results) - 1) * len(task_ids)
        alpha_corrected = cfg["eval"]["mcnemar_alpha"] / n_comparisons \
            if cfg["eval"]["bonferroni_correct"] else cfg["eval"]["mcnemar_alpha"]

        print(f"\nMcNemar's test vs. Fixed Chain (α={alpha_corrected:.4f} Bonferroni-corrected):")
        for name, baseline_results in all_results.items():
            if name == "Fixed Chain":
                continue
            for task_id in task_ids:
                if task_id not in baseline_results or task_id not in fixed_chain_correct:
                    continue
                p = mcnemar_test(fixed_chain_correct[task_id], baseline_results[task_id]["correctness"])
                sig = "***" if p < alpha_corrected / 10 else "**" if p < alpha_corrected else ""
                print(f"  {name} {task_id}: p={p:.4f} {sig}")
                baseline_results[task_id]["mcnemar_p_vs_fixed_chain"] = float(p)
                baseline_results[task_id]["significant"] = p < alpha_corrected

    # Save
    saveable = {}
    for name, baseline_results in all_results.items():
        saveable[name] = {}
        for task_id, res in baseline_results.items():
            saveable[name][task_id] = {k: v.tolist() if hasattr(v, "tolist") else v
                                       for k, v in res.items() if k != "correctness"}
    out_path = os.path.join(results_dir, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(saveable, f, indent=2)
    print(f"\nBaseline results saved to {out_path}")


if __name__ == "__main__":
    main()
