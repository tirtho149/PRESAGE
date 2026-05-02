#!/usr/bin/env python
"""
scripts/evaluate_observe.py
============================
Evaluate OBSERVE model on PlantWild (OOD) routing traces.

Usage:
    # Evaluate on PlantWild
    python scripts/evaluate_observe.py \
      --model observe/checkpoints/observe_final.pt \
      --traces results/plantwild/traces/plantswarm_traces.jsonl \
      --output results/plantwild/observe_evaluation.json

    # Evaluate on PlantVillage validation split
    python scripts/evaluate_observe.py \
      --model observe/checkpoints/observe_final.pt \
      --traces results/plant_village_tfds/traces/plantswarm_traces.jsonl \
      --output results/plant_village_tfds/observe_evaluation.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from observe.inference import OBSERVEInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_image_loader():
    """Build a function to load images from image_id (stub for now)."""

    def load_image(image_id: str):
        from PIL import Image

        # In real deployment, load image from database or filesystem
        # For now, return dummy image
        return Image.new("RGB", (224, 224), color="white")

    return load_image


def build_context_extractor():
    """Build a function to extract context from trace."""

    def extract_context(trace: dict) -> str:
        path = trace.get("path", [])
        context_parts = []
        for agent in path[:-1]:
            context_parts.append(f"[{agent}] processed image")
        return "\n".join(context_parts) if context_parts else ""

    return extract_context


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OBSERVE model on routing traces"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained OBSERVE model weights",
    )
    parser.add_argument(
        "--traces",
        type=Path,
        required=True,
        help="Path to plantswarm_traces.jsonl for evaluation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("observe_evaluation.json"),
        help="Output path for evaluation report",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to evaluate on",
    )

    args = parser.parse_args()

    # Load model
    logger.info(f"Loading OBSERVE model from {args.model}")
    inference = OBSERVEInference(args.model, device=args.device)

    # Build loaders
    image_loader = build_image_loader()
    context_extractor = build_context_extractor()

    # Evaluate
    logger.info(f"Evaluating on {args.traces}")
    eval_metrics = inference.evaluate_on_traces(
        args.traces,
        image_loader,
        context_extractor,
    )

    # Add metadata
    eval_metrics["model_path"] = str(args.model)
    eval_metrics["trace_file"] = str(args.traces)
    eval_metrics["device"] = args.device

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    logger.info(f"Evaluation complete. Saved to {args.output}")

    # Print summary
    logger.info("\n=== Evaluation Summary ===")
    logger.info(f"Agent Accuracy: {eval_metrics['agent_accuracy']:.4f}")
    logger.info(f"Backtrack F1: {eval_metrics['backtrack_f1']:.4f}")
    logger.info(f"Mean Epistemic: {eval_metrics['mean_epistemic']:.4f}")
    logger.info(f"Mean Aleatoric: {eval_metrics['mean_aleatoric']:.4f}")
    logger.info(f"Mean Confidence: {eval_metrics['mean_confidence']:.4f}")


if __name__ == "__main__":
    main()
