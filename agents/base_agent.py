"""
agents/base_agent.py
====================
Abstract base class for all PlantSwarm agents.

Each agent:
  1. Reads the full shared context buffer C_t (§4)
  2. Generates a message m_t with prediction + confidence κ_t ∈ {H,M,L}
  3. Returns (m_t, κ_t, log-probs ℓ_t) for the ensemble (Eq. 3)
  4. Declares a handoff target based on routing logic (Algorithm 1)

Constrained-decoding calibration (Appendix B / Eq. 2) is implemented
via VLLMClient.score_labels().
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from utils.sequence_entropy import (
    entropy_dispersion,
    sequence_mean_entropy,
    targeted_disease_entropy,
    token_entropy_from_openai_token_item,
)
from utils.vllm_client import VLLMClient


# ---------------------------------------------------------------------------
# Context buffer entry (§4)
# ---------------------------------------------------------------------------

@dataclass
class ContextEntry:
    agent_name: str
    message: str
    confidence: str          # 'high' | 'medium' | 'low'
    log_probs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # log_probs[task_id][label] = probability
    mean_entropy_H: Optional[float] = None
    entropy_dispersion_D: Optional[float] = None
    token_entropies: Optional[List[float]] = None


# ---------------------------------------------------------------------------
# Agent output
# ---------------------------------------------------------------------------

@dataclass
class AgentOutput:
    agent_name: str
    message: str             # full text output
    confidence: str          # 'high' | 'medium' | 'low'
    predictions: Dict[str, str]          # {task_id: predicted_label}
    log_probs: Dict[str, Dict[str, float]]   # {task_id: {label: prob}}
    handoff_target: Optional[str]        # next agent name or None (TERMINATE)
    tokens_used: int
    token_entropies: Optional[List[float]] = None
    mean_entropy_H: Optional[float] = None
    entropy_dispersion_D: Optional[float] = None
    targeted_disease_entropy: Optional[float] = None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseAgent(abc.ABC):
    """
    Abstract PlantSwarm agent.

    Subclasses must define:
        - AGENT_NAME      : str
        - TASK_IDS        : List[str]   e.g. ['T1']
        - HANDOFF_MENU    : List[str]   valid handoff targets (Table 1)
        - SYSTEM_PROMPT   : str         (Appendix A)

    and implement:
        - _build_user_message(context, image_b64) → str
        - _parse_response(text) → (predictions, confidence, handoff)
    """

    AGENT_NAME: str = "BaseAgent"
    TASK_IDS: List[str] = []
    HANDOFF_MENU: List[str] = []
    SYSTEM_PROMPT: str = ""

    def __init__(
        self,
        client: VLLMClient,
        label_space: Dict[str, List[str]],
        *,
        sequence_entropy: bool = False,
    ):
        self.client = client
        self.label_space = label_space
        self.sequence_entropy = sequence_entropy

    # ------------------------------------------------------------------
    # Public call interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        image_b64: str,
        context: List[ContextEntry],
        backtrack_count: int = 0,
    ) -> AgentOutput:
        """
        Execute this agent given the current context buffer.
        Returns an AgentOutput with predictions, confidence, log-probs,
        handoff target, and token count.
        """
        messages = self._build_messages(context, image_b64)
        token_hs: Optional[List[float]] = None
        mean_h: Optional[float] = None
        disp_d: Optional[float] = None
        h_dis: Optional[float] = None
        tok_str: Optional[List[str]] = None

        if self.sequence_entropy:
            chat_res = self.client.chat_with_logprobs(
                messages=messages,
                system_prompt=self.SYSTEM_PROMPT,
            )
            response_text = chat_res.text
            tokens = chat_res.completion_tokens
            if chat_res.content_logprobs:
                token_hs = []
                for item in chat_res.content_logprobs:
                    if isinstance(item, dict):
                        token_hs.append(token_entropy_from_openai_token_item(item))
                tok_str = chat_res.token_strings or []
                mean_h = sequence_mean_entropy(token_hs)
                disp_d = entropy_dispersion(token_hs) if token_hs else None
            else:
                token_hs = []
                mean_h = 0.0
                disp_d = 0.0
        else:
            response_text, tokens = self.client.chat(
                messages=messages,
                system_prompt=self.SYSTEM_PROMPT,
            )

        predictions, confidence, handoff = self._parse_response(
            response_text, context, backtrack_count
        )

        # Constrained-decoding scoring pass for calibration (Appendix B)
        log_probs = self._score_all_tasks(
            context_text=self._context_to_text(context),
            image_b64=image_b64,
        )

        if self.sequence_entropy and token_hs is not None:
            dis_lab = predictions.get("T3") or predictions.get("disease_name") or ""
            h_dis = targeted_disease_entropy(
                token_hs,
                response_text,
                str(dis_lab),
                tok_str if tok_str and len(tok_str) == len(token_hs) else None,
            )

        return AgentOutput(
            agent_name=self.AGENT_NAME,
            message=response_text,
            confidence=confidence,
            predictions=predictions,
            log_probs=log_probs,
            handoff_target=handoff,
            tokens_used=tokens,
            token_entropies=token_hs,
            mean_entropy_H=mean_h,
            entropy_dispersion_D=disp_d,
            targeted_disease_entropy=h_dis,
        )

    # ------------------------------------------------------------------
    # Context serialisation
    # ------------------------------------------------------------------

    def _context_to_text(self, context: List[ContextEntry]) -> str:
        lines = []
        for entry in context:
            lines.append(
                f"[{entry.agent_name}] (confidence={entry.confidence})\n{entry.message}"
            )
        return "\n\n".join(lines)

    def _build_messages(
        self, context: List[ContextEntry], image_b64: str
    ) -> List[Dict]:
        """
        Build OpenAI-format message list.
        First user turn carries the image (§4).
        """
        ctx_text = self._context_to_text(context) if context else ""

        user_content: List[Dict] = []

        # Image is always in the first user message position
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            }
        )

        if ctx_text:
            user_content.append(
                {"type": "text", "text": f"Prior agent context:\n{ctx_text}\n\nNow perform your task."}
            )
        else:
            user_content.append(
                {"type": "text", "text": "Analyze the image and perform your task."}
            )

        return [{"role": "user", "content": user_content}]

    # ------------------------------------------------------------------
    # Constrained-decoding scoring (Appendix B / Eq. 2)
    # ------------------------------------------------------------------

    def _score_all_tasks(
        self, context_text: str, image_b64: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Run a constrained-decoding scoring pass for each task in TASK_IDS.
        Returns {task_id: {label: probability}}.

        Excludes DiagnosisAgent (§3: excluded from calibration ensemble
        as it is not an independent observer).
        """
        if self.AGENT_NAME == "DiagnosisAgent":
            return {}

        result: Dict[str, Dict[str, float]] = {}
        for task_id in self.TASK_IDS:
            labels = self.label_space.get(task_id, [])
            if not labels:
                continue
            prefix = (
                f"{self.SYSTEM_PROMPT}\n\n"
                f"Context:\n{context_text}\n\n"
                f"Predict {task_id}: "
            )
            try:
                probs = self.client.score_labels(
                    prompt_prefix=prefix,
                    label_list=labels,
                    image_b64=image_b64,
                )
            except Exception:
                uniform = 1.0 / len(labels)
                probs = {lbl: uniform for lbl in labels}
            result[task_id] = probs
        return result

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _parse_response(
        self,
        text: str,
        context: List[ContextEntry],
        backtrack_count: int,
    ) -> Tuple[Dict[str, str], str, Optional[str]]:
        """
        Parse model text output.
        Returns (predictions_dict, confidence_str, handoff_target_or_None).
        """
        ...

    # ------------------------------------------------------------------
    # Shared parsing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_confidence(text: str) -> str:
        """Extract confidence level from agent text output."""
        text_lower = text.lower()
        for kw in ["high", "medium", "low"]:
            if kw in text_lower:
                return kw
        return "medium"  # safe default

    @staticmethod
    def _extract_label(text: str, label_list: List[str]) -> Optional[str]:
        """Return the first matching label found in text (case-insensitive)."""
        text_lower = text.lower()
        for lbl in label_list:
            if lbl.lower() in text_lower:
                return lbl
        return None

    @staticmethod
    def _routing_decision(
        confidence: str,
        backtrack_count: int,
        all_tasks_covered: bool,
        handoff_menu: List[str],
        default_forward: Optional[str],
    ) -> Optional[str]:
        """
        Implement routing logic §4 / Algorithm 1:

        1. If κ=L and no prior backtrack → MorphologyAgent (regrounding)
        2. If κ=L and already backtracked → proceed forward (prevent loops)
        3. If κ=H and all tasks resolved → DiagnosisAgent (early termination)
        4. Otherwise → default forward target
        """
        if confidence == "low" and backtrack_count == 0:
            return "MorphologyAgent"
        if confidence == "high" and all_tasks_covered:
            return "DiagnosisAgent"
        return default_forward
