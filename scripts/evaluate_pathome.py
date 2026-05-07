"""
scripts/evaluate_pathome.py
===========================
Held-out evaluation on the COMPLETE PlantVillage and/or PlantWild dataset
(paper §5.2-5.3). Drives PlantSwarm + OBSERVE through the existing eval
loop and reports T3 F1, ECE, TPCP, and the seen/unseen slice for PV.

Usage:
    python scripts/evaluate_pathome.py --config configs/plantvillage_full_eval.yaml
    python scripts/evaluate_pathome.py --config configs/plantwild_full_eval.yaml --observe-ckpt observe/checkpoints/observe_grpo_epoch_10.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml
from tqdm import tqdm

from data.loader import PlantDiagBenchLoader
from pathome import PathomeDB
from utils.metrics import macro_f1, tpcp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--observe-ckpt", default=None,
                   help="OBSERVE checkpoint path; if absent, runs PlantSwarm only")
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--unseen-classes", type=str, default=None,
                   help="comma-separated list of T3 classes counted as unseen")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results_dir = cfg["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading test set per {args.config}")
    test_loader = PlantDiagBenchLoader(cfg["data"], split="test")
    label_space = test_loader.label_space
    records = list(test_loader)
    if args.subset:
        records = records[: args.subset]
    print(f"  test records: {len(records)}")

    # Optional PathomeDB
    pathome_db = None
    if cfg.get("pathome", {}).get("load_dir"):
        pdir = cfg["pathome"]["load_dir"]
        if os.path.isdir(pdir):
            print(f"Loading PathomeDB from {pdir}")
            pathome_db = PathomeDB.load(pdir)

    # Build a pipeline (PlantSwarm) using the same construction as run_plantswarm.py
    from utils.vllm_client import VLLMClient, configure_vllm_client_from_yaml
    from plantswarm.autogen_pipeline import AutoGenPlantSwarmPipeline

    client = VLLMClient(
        base_url=cfg["model"]["vllm_base_url"],
        model=cfg["model"]["backbone"],
        temperature=cfg["model"].get("temperature", 0.0),
        seed=cfg["model"].get("seed", 42),
        max_new_tokens=cfg["model"].get("max_new_tokens", 512),
    )
    configure_vllm_client_from_yaml(client, cfg.get("model"), orchestrator="autogen_swarm")
    pipeline = AutoGenPlantSwarmPipeline(
        client=client,
        label_space=label_space,
        Tmax=cfg["routing"]["Tmax"],
        confidence_weights=cfg["routing"]["confidence_weights"],
    )
    if pathome_db is not None:
        pipeline.pathome_db = pathome_db

    # Optional: OBSERVE-driven scoring (paper §7)
    observe_inference = None
    if args.observe_ckpt:
        from observe.inference import OBSERVEInference
        observe_inference = OBSERVEInference(args.observe_ckpt)

    unseen_classes = set()
    if args.unseen_classes:
        unseen_classes = {x.strip() for x in args.unseen_classes.split(",") if x.strip()}

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    all_preds, all_gts, all_correct, all_tokens = {}, {}, {}, []
    all_probs = {}
    seen_unseen = []
    for tid in ("T1", "T2", "T3", "T4", "T5"):
        all_preds[tid] = []
        all_gts[tid] = []
        all_correct[tid] = []
        all_probs[tid] = []

    for rec in tqdm(records):
        try:
            trace = pipeline.run(rec.image_id, rec.image_b64)
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"  [skip] {rec.image_id}: {type(e).__name__}: {e}")
            continue

        for tid in ("T1", "T2", "T3", "T4", "T5"):
            attr = {
                "T1": "symptom_type", "T2": "pathogen_class", "T3": "disease_name",
                "T4": "severity_class", "T5": "crop_species",
            }[tid]
            gt = getattr(rec, attr, None)
            labels = label_space.get(tid, [])
            pred = trace.final_predictions.get(tid, labels[0] if labels else "")
            if gt is None:
                continue
            all_preds[tid].append(pred)
            all_gts[tid].append(gt)
            all_correct[tid].append(int(pred == gt))
            probs = trace.ensemble_probs.get(tid, {})
            all_probs[tid].append([probs.get(l, 1.0 / max(len(labels), 1)) for l in labels])

        all_tokens.append(trace.total_tokens)
        if rec.disease_name:
            seen_unseen.append("unseen" if rec.disease_name in unseen_classes else "seen")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metrics = {}
    for tid in ("T1", "T2", "T3", "T4", "T5"):
        labels = label_space.get(tid, [])
        if not all_preds[tid]:
            continue
        f1, (lo, hi) = macro_f1(all_preds[tid], all_gts[tid], labels,
                                bootstrap_n=cfg["eval"]["bootstrap_n"])
        tpcp_v = tpcp(all_tokens[: len(all_correct[tid])], all_correct[tid])
        from calibration.ece import compute_ece_from_probs
        probs_mat = np.array(all_probs[tid])
        ece, _ = compute_ece_from_probs(
            probs_mat, np.array(all_gts[tid]), labels,
            n_bins=cfg["calibration"]["ece_bins"],
        )
        metrics[tid] = {
            "macro_f1": float(f1),
            "macro_f1_ci": [float(lo), float(hi)],
            "ece": float(ece),
            "tpcp": float(tpcp_v),
            "n": len(all_preds[tid]),
        }

    # Seen / unseen slice (paper §5.2 P5)
    if seen_unseen and unseen_classes:
        seen_idx = [i for i, s in enumerate(seen_unseen) if s == "seen"]
        unseen_idx = [i for i, s in enumerate(seen_unseen) if s == "unseen"]
        for slice_name, idx in [("seen", seen_idx), ("unseen", unseen_idx)]:
            if not idx:
                continue
            preds = [all_preds["T3"][i] for i in idx]
            gts = [all_gts["T3"][i] for i in idx]
            labels = label_space.get("T3", [])
            f1, _ = macro_f1(preds, gts, labels, bootstrap_n=cfg["eval"]["bootstrap_n"])
            metrics.setdefault("T3_slices", {})[slice_name] = {
                "macro_f1": float(f1), "n": len(preds),
            }

    out_path = os.path.join(results_dir, "pathome_eval.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
