#!/usr/bin/env python
"""
scripts/train_observe.py
========================
Train OBSERVE model on PlantSwarm routing traces.

Usage:
    python scripts/train_observe.py \
      --traces results/plant_village_tfds/traces/plantswarm_traces.jsonl \
      --output observe/checkpoints/observe_final.pt \
      --epochs 50 \
      --batch-size 8
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split

from observe.model import OBSERVE
from observe.trainer import OBSERVETrainer, RoutingTraceDataset, TraceAnnotation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_annotations_from_traces(
    trace_file: Path,
) -> list[TraceAnnotation]:
    """
    Convert PlantSwarm routing traces to OBSERVE training annotations.

    Args:
        trace_file: Path to plantswarm_traces.jsonl

    Returns:
        List of TraceAnnotation objects with labeled epistemic/aleatoric uncertainty
    """
    import base64
    import io
    from PIL import Image

    annotations = []

    with open(trace_file) as f:
        for line in f:
            trace = json.loads(line.strip())

            image_id = trace.get("image_id", "")
            path_length = trace.get("path_length", 5)
            backtrack_count = trace.get("backtrack_count", 0)
            ground_truth = trace.get("ground_truth", {})
            final_predictions = trace.get("final_predictions", {})

            # Heuristic epistemic uncertainty: based on path length
            # Short path (2-3) → low epistemic (good evidence)
            # Long path (4-5) → high epistemic (needed evidence gathering)
            epistemic = min(max((path_length - 2) / 3.0, 0.0), 1.0)

            # Heuristic aleatoric uncertainty: based on task difficulty
            # Correct T3 prediction → low aleatoric
            # Incorrect → higher aleatoric (inherent difficulty)
            t3_correct = final_predictions.get("T3") == ground_truth.get("T3", "")
            aleatoric = 0.2 if t3_correct else 0.6

            # Backtrack decision based on actual backtrack count
            backtrack = backtrack_count > 0

            # Confidence: higher if path converged early and correct
            confidence = 0.9 if (path_length <= 3 and t3_correct) else 0.6 if t3_correct else 0.3

            # Extract next agent (from path, typically 2nd agent after MorphologyAgent)
            path = trace.get("path", ["MorphologyAgent", "SymptomAgent", "DiagnosisAgent"])
            next_agent = path[1] if len(path) > 1 else "SymptomAgent"

            # Create dummy base64 image (in real training, load from image_id)
            dummy_image = Image.new("RGB", (224, 224), color="white")
            img_bytes = io.BytesIO()
            dummy_image.save(img_bytes, format="PNG")
            image_b64 = base64.b64encode(img_bytes.getvalue()).decode()

            # Build context text from prior predictions in path
            context_parts = []
            for agent in path[:-1]:  # Exclude final agent
                context_parts.append(f"[{agent}] processed image")
            context_text = "\n".join(context_parts) if context_parts else ""

            belief_state = f"Path: {' → '.join(path)}. T3: {final_predictions.get('T3', 'N/A')}"

            ann = TraceAnnotation(
                image_id=image_id,
                image_b64=image_b64,
                context_text=context_text,
                next_agent=next_agent,
                backtrack=backtrack,
                epistemic=epistemic,
                aleatoric=aleatoric,
                confidence=confidence,
                belief_state=belief_state,
            )
            annotations.append(ann)

    logger.info(f"Loaded {len(annotations)} annotations from {trace_file}")
    return annotations


def main():
    parser = argparse.ArgumentParser(
        description="Train OBSERVE model on PlantSwarm routing traces"
    )
    parser.add_argument(
        "--traces",
        type=Path,
        required=True,
        help="Path to plantswarm_traces.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("observe/checkpoints/observe_final.pt"),
        help="Output path for trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Load annotations
    logger.info(f"Loading annotations from {args.traces}")
    annotations = load_annotations_from_traces(args.traces)

    if not annotations:
        logger.error(f"No annotations found in {args.traces}")
        return

    # Create dataset
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    dataset = RoutingTraceDataset(annotations, processor)

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    logger.info(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Batch size: {args.batch_size}"
    )

    # Initialize model and trainer
    logger.info("Initializing OBSERVE model")
    model = OBSERVE()
    trainer = OBSERVETrainer(model, device=args.device, lr=args.lr)

    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    best_val_loss = float("inf")
    train_history = {"epochs": [], "train_loss": [], "val_loss": []}

    for epoch in range(args.epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        logger.info(f"Epoch {epoch + 1}/{args.epochs} - Train loss: {train_metrics['total_loss']:.4f}")

        # Validate
        val_metrics = trainer.validate(val_loader)
        logger.info(f"Epoch {epoch + 1}/{args.epochs} - Val loss: {val_metrics['val_loss']:.4f}, Agent accuracy: {val_metrics['agent_accuracy']:.4f}")

        # Track history
        train_history["epochs"].append(epoch + 1)
        train_history["train_loss"].append(train_metrics["total_loss"])
        train_history["val_loss"].append(val_metrics["val_loss"])

        # Save best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            trainer.save(args.output)
            logger.info(f"Saved best model to {args.output}")

    # Save training history
    history_file = args.output.parent / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(train_history, f, indent=2)
    logger.info(f"Saved training history to {history_file}")

    logger.info("Training complete")


if __name__ == "__main__":
    main()
