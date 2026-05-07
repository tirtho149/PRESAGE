"""
observe/model.py
================
OBSERVE Vision-Language-Action model architecture.

Fine-tuned from Qwen2.5-VL-3B with LoRA:
- Per-step visual grounding (image present at every step)
- Outputs structured epistemic actions
- 56M trainable parameters (50M LoRA + 6M heads)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor


@dataclass
class EpistemicAction:
    """Structured action output from OBSERVE."""
    next_agent: str  # 5-class: MorphologyAgent, SymptomAgent, PathogenAgent, SeverityAgent, DiagnosisAgent
    backtrack: bool  # Whether to backtrack to MorphologyAgent
    epistemic_uncertainty: float  # ∈ [0, 1]: resolvable ambiguity (get better evidence)
    aleatoric_uncertainty: float  # ∈ [0, 1]: irreducible difficulty (escalate to human)
    confidence: float  # ∈ [0, 1]: calibrated confidence in prediction
    belief_state: str  # Natural language belief (what agent thinks now)


class OBSERVE(nn.Module):
    """
    Vision-Language-Action model for epistemic action selection.

    Architecture:
    - Backbone: Qwen2.5-VL-3B (frozen, ~2.95B params)
    - LoRA: r=16, α=32, applied to q/k/v/o_proj (~50M trainable)
    - Heads: routing (5-class), backtrack (binary), epistemic (scalar),
             aleatoric (scalar), confidence (scalar), belief_text (autoregressive)
    - Total trainable: ~56M / 3B (1.8%)
    """

    def __init__(
        self,
        backbone: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        agent_classes: Optional[list] = None,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.agent_classes = agent_classes or [
            "MorphologyAgent", "SymptomAgent", "PathogenAgent",
            "SeverityAgent", "DiagnosisAgent"
        ]

        # Load base model and processor
        self.processor = AutoProcessor.from_pretrained(backbone)
        self.model = AutoModelForVision2Seq.from_pretrained(
            backbone,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Get hidden dimension from model
        hidden_dim = self.model.config.hidden_size  # Usually 2048 for Qwen2.5-VL-3B

        # Apply LoRA to vision encoder and language model
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        # Shared representation head
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Task-specific heads
        self.routing_head = nn.Linear(512, len(self.agent_classes))  # 5-class softmax
        self.backtrack_head = nn.Linear(512, 1)  # Binary: sigmoid
        self.epistemic_head = nn.Linear(512, 1)  # Scalar: sigmoid [0, 1]
        self.aleatoric_head = nn.Linear(512, 1)  # Scalar: sigmoid [0, 1]
        self.confidence_head = nn.Linear(512, 1)  # Scalar: sigmoid [0, 1]

        # Belief text autoregressive head (uses model's decoder)
        # Belief is generated via model decoder, not a separate head

    def forward(
        self,
        image: torch.Tensor,
        context_text: str,
        return_dict: bool = True,
    ) -> dict | tuple:
        """
        Forward pass for epistemic action selection.

        Args:
            image: Input image tensor (after preprocessing)
            context_text: Prior agent messages and context
            return_dict: Return as dict (True) or tuple

        Returns:
            Dict with keys: next_agent, backtrack, epistemic, aleatoric, confidence, belief
        """

        # Prepare input: image + context text
        prompt = f"Prior context:\n{context_text}\n\nBased on this image and context, what is your next action?"
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Forward through model to get hidden states
        with torch.no_grad():
            # Get embeddings/hidden states from model
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Use last hidden state as representation
        hidden_state = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        pooled = hidden_state.mean(dim=1)  # [batch, hidden_dim]

        # Pass through shared head
        shared_repr = self.shared_head(pooled)  # [batch, 512]

        # Compute action outputs
        routing_logits = self.routing_head(shared_repr)  # [batch, 5]
        backtrack_logits = self.backtrack_head(shared_repr)  # [batch, 1]
        epistemic_logits = self.epistemic_head(shared_repr)  # [batch, 1]
        aleatoric_logits = self.aleatoric_head(shared_repr)  # [batch, 1]
        confidence_logits = self.confidence_head(shared_repr)  # [batch, 1]

        # Apply activations
        routing_probs = torch.softmax(routing_logits, dim=-1)  # [batch, 5]
        backtrack_prob = torch.sigmoid(backtrack_logits).squeeze(-1)  # [batch]
        epistemic = torch.sigmoid(epistemic_logits).squeeze(-1)  # [batch]
        aleatoric = torch.sigmoid(aleatoric_logits).squeeze(-1)  # [batch]
        confidence = torch.sigmoid(confidence_logits).squeeze(-1)  # [batch]

        # Generate belief text (autoregressive from decoder)
        belief_prompt = f"My belief state is: "
        belief_inputs = self.processor.tokenizer(
            belief_prompt,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        belief_outputs = self.model.generate(
            **belief_inputs,
            max_new_tokens=50,
            temperature=0.7,
        )
        belief_text = self.processor.batch_decode(
            belief_outputs,
            skip_special_tokens=True,
        )[0]

        if return_dict:
            return {
                "routing_probs": routing_probs,
                "routing_class": self.agent_classes[routing_probs.argmax(dim=-1).item()],
                "backtrack_prob": backtrack_prob,
                "epistemic": epistemic,
                "aleatoric": aleatoric,
                "confidence": confidence,
                "belief_text": belief_text,
            }
        else:
            return (routing_probs, backtrack_prob, epistemic, aleatoric, confidence, belief_text)

    def get_epistemic_action(
        self,
        image: torch.Tensor,
        context_text: str,
        backtrack_threshold: float = 0.5,
    ) -> EpistemicAction:
        """Get a single EpistemicAction from image and context."""
        outputs = self.forward(image, context_text, return_dict=True)

        return EpistemicAction(
            next_agent=outputs["routing_class"],
            backtrack=outputs["backtrack_prob"].item() > backtrack_threshold,
            epistemic_uncertainty=outputs["epistemic"].item(),
            aleatoric_uncertainty=outputs["aleatoric"].item(),
            confidence=outputs["confidence"].item(),
            belief_state=outputs["belief_text"],
        )
