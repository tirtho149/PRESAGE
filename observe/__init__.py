"""
observe/
========
OBSERVE: Vision-Language-Action model for epistemic action selection.

Paper §7 (pathome_final). Fine-tuned from Qwen2.5-VL-7B with LoRA (r=16, alpha=32).
Trained two-phase on PlantSwarm Bugwood traces:
  Phase A — Decision Transformer (return-conditioned sequence modeling)
  Phase B — GRPO (group-relative policy optimization, KL-anchored to Phase A)

Outputs: next_agent, backtrack b_t, epistemic eps_t, aleatoric alpha_t,
calibrated confidence c_t, overconfidence flag OC_t, belief state s_t.

6× lower inference cost than PlantSwarm; ECE 0.12 on full PlantVillage,
0.17 on full PlantWild (paper §8).
"""

from .model import OBSERVE, EpistemicAction
from .trainer import OBSERVETrainer
from .inference import OBSERVEInference

__all__ = ["OBSERVE", "EpistemicAction", "OBSERVETrainer", "OBSERVEInference"]
