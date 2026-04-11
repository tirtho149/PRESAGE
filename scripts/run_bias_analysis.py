"""
scripts/run_bias_analysis.py
=============================
Run RDS + mixed-effects bias analysis (§8, Appendix D).

Loads saved routing traces and demographic metadata, computes:
    - RDS(g, M) for all groups and metrics (Table 6)
    - Spearman ρ(RDS(g, L), Δacc(g)) — expected ≈ −0.68 (§8)
    - T5 negative control (§8)
    - Mixed-effects regression β_g (Eq. 5, Appendix D)
    - Cross-model RDS consistency Kendall's τ (§7)

Usage:
    python scripts/run_bias_analysis.py --config configs/default.yaml
        --traces results/traces/plantswarm_traces.jsonl
        --predictions results/plantswarm_predictions.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml

from bias.mixed_effects import build_regression_dataframe, fit_mixed_effects, sensitivity_analyses
from bias.rds import (
    compute_rds_table,
    rds_accuracy_correlation,
    t5_negative_control_check,
    cross_model_rds_consistency,
)
from data.loader import PlantDiagBenchLoader
from utils.routing_trace import load_traces


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--traces",
                        default="results/traces/plantswarm_traces.jsonl",
                        help="PlantSwarm routing traces JSONL")
    parser.add_argument("--predictions",
                        default="results/plantswarm_predictions.jsonl",
                        help="PlantSwarm predictions JSONL")
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_predictions(path: str):
    preds = []
    with open(path) as f:
        for line in f:
            preds.append(json.loads(line.strip()))
    return preds


def main():
    args = parse_args()
    cfg = load_config(args.config)
    results_dir = args.output_dir or cfg["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # --- Load traces and predictions ---
    print("Loading routing traces...")
    traces = load_traces(args.traces)
    print(f"  {len(traces)} traces loaded")

    print("Loading predictions...")
    predictions = load_predictions(args.predictions)
    pred_by_id = {p["image_id"]: p for p in predictions}

    # --- Load demographic / group metadata from PlantDiagBench ---
    print("Loading PlantDiagBench metadata...")
    loader = PlantDiagBenchLoader(cfg["data"], split="test")
    df = loader.dataframe

    # Align traces with demographic data
    demographic_cols = cfg["data"].get("demographic_cols") or {}
    gender_col = demographic_cols.get("gender")
    region_col = demographic_cols.get("region")

    id_to_gender = {}
    id_to_region = {}
    id_to_sector = {}
    id_to_quality = {}
    id_to_complexity = {}

    id_col = cfg["data"].get("id_col", "id")
    sector_col = cfg["data"]["label_cols"].get("T1", "symptom_type")
    for _, row in df.iterrows():
        img_id = str(row.get(id_col, row.name))
        if gender_col and gender_col in row.index and pd.notna(row.get(gender_col)):
            id_to_gender[img_id] = row.get(gender_col)
        else:
            id_to_gender[img_id] = "unknown"
        if region_col and region_col in row.index and pd.notna(row.get(region_col)):
            id_to_region[img_id] = row.get(region_col)
        else:
            id_to_region[img_id] = "Other"
        id_to_sector[img_id] = row.get(sector_col, "Other")
        id_to_quality[img_id] = float(row.get("img_quality_score", 0.5) or 0.5)
        id_to_complexity[img_id] = float(row.get("complexity_edge_density", 0.5) or 0.5)

    # Build aligned arrays
    trace_ids = [t["image_id"] for t in traces]
    gender_labels = [id_to_gender.get(i, "unknown") for i in trace_ids]
    region_labels = [id_to_region.get(i, "Other") for i in trace_ids]
    sector_labels = [id_to_sector.get(i, "Other") for i in trace_ids]
    quality_scores = np.array([id_to_quality.get(i, 0.5) for i in trace_ids])
    complexity_scores = np.array([id_to_complexity.get(i, 0.5) for i in trace_ids])

    # --- Accuracy per group for Δacc ---
    label_space = loader.label_space
    task_id = "T1"

    acc_by_gender = {}
    acc_by_region = {}

    for gender in set(gender_labels):
        mask = [g == gender for g in gender_labels]
        correct = []
        for t, m, gend in zip(traces, mask, gender_labels):
            if not m:
                continue
            pred_rec = pred_by_id.get(t["image_id"], {})
            pred = pred_rec.get("predictions", {}).get(task_id)
            gt = pred_rec.get("ground_truth", {}).get(task_id)
            if pred and gt:
                correct.append(int(pred == gt))
        acc_by_gender[gender] = float(np.mean(correct)) if correct else 0.0

    for region in set(region_labels):
        mask = [r == region for r in region_labels]
        correct = []
        for t, m in zip(traces, mask):
            if not m:
                continue
            pred_rec = pred_by_id.get(t["image_id"], {})
            pred = pred_rec.get("predictions", {}).get(task_id)
            gt = pred_rec.get("ground_truth", {}).get(task_id)
            if pred and gt:
                correct.append(int(pred == gt))
        acc_by_region[region] = float(np.mean(correct)) if correct else 0.0

    global_acc = float(np.mean(list(acc_by_gender.values())))
    acc_gap_gender = {g: acc - global_acc for g, acc in acc_by_gender.items()}
    acc_gap_region = {r: acc - global_acc for r, acc in acc_by_region.items()}

    # --- RDS Table (Table 6) ---
    print("\nComputing RDS by gender (Table 6)...")
    rds_gender_df = compute_rds_table(traces, gender_labels, acc_by_gender)
    print(rds_gender_df.to_string(index=False))

    print("\nComputing RDS by region (Table 6)...")
    rds_region_df = compute_rds_table(traces, region_labels, acc_by_region)
    print(rds_region_df.to_string(index=False))

    # --- RDS–accuracy correlation (§8) ---
    from bias.rds import compute_rds
    rds_L_gender = compute_rds(traces, gender_labels, metric="path_length")
    rho_g, p_g = rds_accuracy_correlation(rds_L_gender, acc_gap_gender)
    print(f"\nSpearman ρ(RDS(g,L), Δacc(g)) by gender: ρ={rho_g:.3f}, p={p_g:.4f}")
    print(f"  Paper predicts ρ ≈ −0.68 (§8)")

    rds_L_region = compute_rds(traces, region_labels, metric="path_length")
    rho_r, p_r = rds_accuracy_correlation(rds_L_region, acc_gap_region)
    print(f"Spearman ρ(RDS(g,L), Δacc(g)) by region: ρ={rho_r:.3f}, p={p_r:.4f}")

    # --- T5 negative control (§8) ---
    print("\nT5 negative control check (§8)...")
    t5_check_gender = t5_negative_control_check(traces, gender_labels, threshold=0.05)
    print(f"  {t5_check_gender['interpretation']}")

    # --- Mixed-effects regression (Eq. 5, Appendix D) ---
    print("\nFitting mixed-effects regression (Eq. 5)...")
    reg_df = build_regression_dataframe(
        traces=traces,
        group_labels=gender_labels,
        sector_labels=sector_labels,
        region_labels=region_labels,
        complexity_scores=complexity_scores,
        quality_scores=quality_scores,
        reference_gender=cfg["bias"].get("reference_gender", "male"),
        reference_region=cfg["bias"].get("reference_region", "Americas"),
    )

    n_groups = reg_df["group"].nunique()
    bonferroni_n = n_groups - 1 if cfg["bias"].get("bonferroni_correct", True) else None

    try:
        me_result = fit_mixed_effects(
            df=reg_df,
            outcome_col="path_length",
            group_col="group",
            fixed_covariates=["complexity", "quality"],
            random_effects_col="sector",
            bonferroni_n=bonferroni_n,
            reml=cfg["bias"].get("reml", True),
        )
        print("\n  β_g table (significant groups after confounder control):")
        print(me_result["beta_g_table"].to_string(index=False))
        print(f"  Significant groups: {me_result['significant_groups']}")

        # Sensitivity analyses (Appendix D)
        print("\n  Running sensitivity analyses (Appendix D)...")
        sens = sensitivity_analyses(reg_df, confounders=["complexity", "quality"],
                                    bonferroni_n=bonferroni_n, reml=True)
        for model_name, sens_result in sens.items():
            sig = sens_result.get("significant_groups", [])
            print(f"    {model_name}: significant groups = {sig}")

    except ImportError as e:
        print(f"  [SKIP] statsmodels not available: {e}")
        me_result = {}

    # --- Save results ---
    bias_output = {
        "rds_by_gender": rds_gender_df.to_dict(orient="records"),
        "rds_by_region": rds_region_df.to_dict(orient="records"),
        "rds_accuracy_correlation_gender": {"rho": rho_g, "p": p_g},
        "rds_accuracy_correlation_region": {"rho": rho_r, "p": p_r},
        "t5_negative_control": {k: v for k, v in t5_check_gender.items()
                                 if k != "rds_t5"},
    }

    out_path = os.path.join(results_dir, "bias_analysis.json")
    with open(out_path, "w") as f:
        json.dump(bias_output, f, indent=2, default=str)

    rds_gender_df.to_csv(os.path.join(results_dir, "rds_gender.csv"), index=False)
    rds_region_df.to_csv(os.path.join(results_dir, "rds_region.csv"), index=False)
    print(f"\nBias analysis saved to {results_dir}")


if __name__ == "__main__":
    main()
