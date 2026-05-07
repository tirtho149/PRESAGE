"""
observe/trainer.py
==================
OBSERVE training pipeline.

Trains on 10,000 PlantSwarm routing traces:
- Phase 1: Collect traces (run_plantswarm.py)
- Phase 2: Annotate (extract labels from traces)
- Phase 3: Fine-tune LoRA on traces
- Phase 4: Evaluate on PlantWild (OOD)

Training hyperparameters (paper §5):
- LoRA: r=16, α=32
- Optimizer: AdamW, lr=1e-4
- Loss: weighted combination of routing + calibration + consistency + belief
- Hardware: single A100 40GB, ~4-6 hours
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from observe.model import OBSERVE

logger = logging.getLogger(__name__)


@dataclass
class TraceAnnotation:
    """Labeled routing trace for OBSERVE training."""
    image_id: str
    image_b64: str
    context_text: str
    next_agent: str  # Ground truth next agent
    backtrack: bool  # Whether backtracking actually helped
    epistemic: float  # ∈ [0, 1]: did more evidence help?
    aleatoric: float  # ∈ [0, 1]: was this inherently hard?
    confidence: float  # Calibrated confidence in final prediction
    belief_state: str  # Summary of reasoning


class RoutingTraceDataset(Dataset):
    """PyTorch Dataset for routing trace annotations."""

    def __init__(
        self,
        annotations: list[TraceAnnotation],
        processor,
    ):
        self.annotations = annotations
        self.processor = processor

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        ann = self.annotations[idx]

        # Decode image from base64
        import base64
        import io
        from PIL import Image

        img_bytes = base64.b64decode(ann.image_b64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Prepare inputs
        inputs = self.processor(
            images=image,
            text=ann.context_text,
            return_tensors="pt",
            padding=True,
        )

        # Remove batch dimension from processor output
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor) and inputs[key].ndim > 1:
                inputs[key] = inputs[key].squeeze(0)

        # Map agent name to class index
        agent_classes = [
            "MorphologyAgent", "SymptomAgent", "PathogenAgent",
            "SeverityAgent", "DiagnosisAgent"
        ]
        next_agent_idx = agent_classes.index(ann.next_agent)

        return {
            "image": image,
            "context_text": ann.context_text,
            "next_agent": torch.tensor(next_agent_idx, dtype=torch.long),
            "backtrack": torch.tensor(float(ann.backtrack), dtype=torch.float32),
            "epistemic": torch.tensor(ann.epistemic, dtype=torch.float32),
            "aleatoric": torch.tensor(ann.aleatoric, dtype=torch.float32),
            "confidence": torch.tensor(ann.confidence, dtype=torch.float32),
            "belief_state": ann.belief_state,
        }


class OBSERVETrainer:
    """Trainer for OBSERVE model."""

    def __init__(
        self,
        model: OBSERVE,
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_weights: dict = None,
    ) -> dict:
        """Train for one epoch."""
        if loss_weights is None:
            loss_weights = {
                "routing": 1.0,
                "calibration": 0.4,
                "consistency": 0.2,
                "belief": 0.2,
            }

        self.model.train()
        total_loss = 0.0
        loss_components = {k: 0.0 for k in loss_weights}

        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            self.optimizer.zero_grad()

            # Move batch to device
            image = batch["image"].to(self.device)
            context = batch["context_text"]
            y_agent = batch["next_agent"].to(self.device)
            y_backtrack = batch["backtrack"].to(self.device)
            y_epistemic = batch["epistemic"].to(self.device)
            y_aleatoric = batch["aleatoric"].to(self.device)
            y_confidence = batch["confidence"].to(self.device)

            # Forward pass
            outputs = self.model(image, context[0], return_dict=True)

            # Compute losses
            loss_routing = F.cross_entropy(
                outputs["routing_probs"],
                y_agent.unsqueeze(0),
            )
            loss_calibration = F.mse_loss(
                outputs["confidence"],
                y_confidence.unsqueeze(0),
            )
            loss_consistency = F.mse_loss(
                outputs["epistemic"] + outputs["aleatoric"],
                torch.ones_like(outputs["epistemic"]),
            )

            # Belief text loss (simplified: use MSE on confidence)
            loss_belief = F.mse_loss(
                outputs["confidence"],
                y_confidence.unsqueeze(0),
            )

            # Weighted combination
            loss = (
                loss_weights["routing"] * loss_routing +
                loss_weights["calibration"] * loss_calibration +
                loss_weights["consistency"] * loss_consistency +
                loss_weights["belief"] * loss_belief
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loss_components["routing"] += loss_routing.item()
            loss_components["calibration"] += loss_calibration.item()
            loss_components["consistency"] += loss_consistency.item()
            loss_components["belief"] += loss_belief.item()

            pbar.set_postfix({"loss": loss.item()})

        n_batches = len(train_loader)
        return {
            "total_loss": total_loss / n_batches,
            **{k: v / n_batches for k, v in loss_components.items()},
        }

    def validate(self, val_loader: DataLoader) -> dict:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct_agent = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                image = batch["image"].to(self.device)
                context = batch["context_text"]
                y_agent = batch["next_agent"].to(self.device)
                y_confidence = batch["confidence"].to(self.device)

                outputs = self.model(image, context[0], return_dict=True)

                loss = F.cross_entropy(
                    outputs["routing_probs"],
                    y_agent.unsqueeze(0),
                )
                total_loss += loss.item()

                pred_agent = outputs["routing_probs"].argmax(dim=-1)
                correct_agent += (pred_agent == y_agent).sum().item()
                total += y_agent.numel()

        return {
            "val_loss": total_loss / len(val_loader),
            "agent_accuracy": correct_agent / total if total > 0 else 0.0,
        }

    def save(self, path: str | Path):
        """Save model weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.model.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path):
        """Load model weights."""
        self.model.model.from_pretrained(path)
        logger.info(f"Model loaded from {path}")
