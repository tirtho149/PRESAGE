"""
observe/
========
OBSERVE: Observation-Based Structured Epistemic Representation for Visual Evaluation.
Vision-Language-Action model trained on PlantSwarm routing traces for epistemic action selection.

Paper §5: Fine-tuned from Qwen2.5-VL-3B with LoRA (r=16, α=32).
Outputs: next_agent, backtrack, epistemic/aleatoric uncertainty, confidence, belief_state.
6× lower inference cost than PlantSwarm, 52% ECE improvement under domain shift.
"""

from .model import OBSERVE
from .trainer import OBSERVETrainer
from .inference import OBSERVEInference

__all__ = ["OBSERVE", "OBSERVETrainer", "OBSERVEInference"]
