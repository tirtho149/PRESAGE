#!/usr/bin/env python
"""
scripts/collect_metrics.py
===========================
Aggregate metrics from all experiments (PlantSwarm, OBSERVE, ablations, baselines)
into a unified results JSON for LaTeX sync.

Usage:
    python scripts/collect_metrics.py \
      --results-dir results/plant_village_tfds/ \
      --output results/unified_metrics.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json_safe(path: Path, default=None):
    """Safely load JSON file, return default if missing."""
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse {path}")
            return default
    return default


def collect_plantswarm_metrics(results_dir: Path) -> dict:
    """Collect PlantSwarm metrics from results directory."""
    metrics_file = results_dir / "plantswarm_metrics.json"
    data = load_json_safe(metrics_file, {})

    return {
        "plantswarm": data.get("T2", {}),
        "plantswarm_t3": data.get("T3", {}),
        "plantswarm_by_benchmark": data.get("by_benchmark", {}),
    }


def collect_baseline_metrics(results_dir: Path) -> dict:
    """Collect baseline comparison metrics."""
    baseline_file = results_dir / "baseline_results.json"
    data = load_json_safe(baseline_file, {})

    baselines = {}
    for method_name, metrics in data.items():
        baselines[method_name] = {
            "t3_f1": metrics.get("T3", {}).get("macro_f1"),
            "t2_f1": metrics.get("T2", {}).get("macro_f1"),
            "ece": metrics.get("calibration", {}).get("ece"),
            "tpcp": metrics.get("efficiency", {}).get("tokens_per_correct"),
        }

    return {"baselines": baselines}


def collect_observe_metrics(results_dir: Path) -> dict:
    """Collect OBSERVE evaluation metrics."""
    observe_file = results_dir / "observe_evaluation.json"
    data = load_json_safe(observe_file, {})

    return {
        "observe_seen": {
            "ece": data.get("calibration", {}).get("ece"),
            "agent_accuracy": data.get("agent_accuracy"),
            "backtrack_f1": data.get("backtrack_f1"),
        }
    }


def collect_ablation_metrics(results_dir: Path) -> dict:
    """Collect ablation study results."""
    ablations = {}

    # Standard ablation variants
    variants = [
        "fixed_chain",
        "fixed_chain_ctx",
        "free_no_backtrack",
        "free_no_conf_gate",
        "three_agent_swarm",
    ]

    for variant in variants:
        metric_file = results_dir / f"ablation_metrics_{variant}.json"
        data = load_json_safe(metric_file, {})
        ablations[variant] = {
            "t3_f1": data.get("T3", {}).get("macro_f1"),
            "t2_f1": data.get("T2", {}).get("macro_f1"),
            "ece": data.get("calibration", {}).get("ece"),
        }

    return {"ablations": ablations}


def collect_calibration_metrics(results_dir: Path) -> dict:
    """Collect calibration analysis results."""
    calib_file = results_dir / "calibration_report.json"
    data = load_json_safe(calib_file, {})

    return {
        "calibration": {
            "ece": data.get("ece"),
            "ece_after_temp_scaling": data.get("ece_after_temperature_scaling"),
            "kappa_calibration": data.get("kappa_calibration_report", {}),
        }
    }


def collect_routing_metrics(results_dir: Path) -> dict:
    """Collect routing analysis results (P1-P4)."""
    routing_file = results_dir / "routing_analysis.json"
    data = load_json_safe(routing_file, {})

    return {
        "routing_analysis": {
            "p1_path_entropy_correlation": data.get("p1", {}).get("spearman_rho"),
            "p2_backtrack_improvement": data.get("p2", {}).get("delta_accuracy"),
            "p3_early_termination": data.get("p3", {}).get("delta_accuracy"),
            "p4_observe_ood_ece": data.get("p4", {}).get("observe_ood_ece"),
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate all experiment metrics for LaTeX sync"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Results directory containing all metric files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/unified_metrics.json"),
        help="Output path for unified metrics",
    )

    args = parser.parse_args()

    logger.info(f"Collecting metrics from {args.results_dir}")

    # Aggregate all metrics
    unified = {}
    unified.update(collect_plantswarm_metrics(args.results_dir))
    unified.update(collect_baseline_metrics(args.results_dir))
    unified.update(collect_observe_metrics(args.results_dir))
    unified.update(collect_ablation_metrics(args.results_dir))
    unified.update(collect_calibration_metrics(args.results_dir))
    unified.update(collect_routing_metrics(args.results_dir))

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(unified, f, indent=2)

    logger.info(f"Saved unified metrics to {args.output}")


if __name__ == "__main__":
    main()
