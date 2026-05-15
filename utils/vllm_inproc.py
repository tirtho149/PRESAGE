"""
utils/vllm_inproc.py
====================
In-process vLLM backend for the Phase 0R swarm.

Why this exists
---------------
The original ``utils/vllm_client.py`` talks to a separately-launched
``vllm serve`` over OpenAI-compatible HTTP. On Nova that produced
``HTTPError: 400 Client Error`` on every specialist call after the
server returned non-OpenAI responses for some image+prompt combinations,
and the failure mode left every (crop, disease, state) tuple with zero
deltas. Phase 0R is a single-node, single-GPU pipeline — there is no
reason to keep an HTTP boundary inside the same job.

This module loads ``vllm.LLM`` directly in the same Python process, so
the swarm calls the model via a Python method instead of a socket.
There is no server, no port, no JSON wire format, and no path where a
400 can silently zero out a run.

API surface
-----------
``InProcessVLLMClient`` duck-types the subset of ``utils.vllm_client.VLLMClient``
that Phase 0R actually uses:

  - ``chat(messages, system_prompt=..., seed=..., temperature=...) -> (text, tokens)``
  - ``chat_with_logprobs(...) -> ChatResult``  (compat shim; logprobs disabled)
  - ``count_tokens(text)``

That is the entire contract observed by ``agents/base_agent.py`` and
``agents/diagnosis_agent.py``.

Concurrency
-----------
``vllm.LLM`` is a single batched engine — calling ``.chat()`` from
multiple Python threads concurrently is unsafe. The swarm currently
fans out 24 specialists through ``ThreadPoolExecutor``. Each thread's
call to ``InProcessVLLMClient.chat`` acquires a process-wide lock and
hands one prompt to ``LLM.chat()``. The lock is short enough that
threads behave like a serial queue feeding the engine, which is also
what vLLM expects for the single-LLM offline-batching style.

A batched-fanout variant (collect all 24 round-1 prompts, hand them
to ``LLM.chat()`` in one call, dispatch results back) is a bigger
refactor of ``delta_pipeline._run_single_pass`` and is left for a
follow-up — correctness first.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    """Drop-in shape match for utils.vllm_client.ChatResult."""

    text: str
    completion_tokens: int
    content_logprobs: Optional[List[Dict[str, Any]]] = None
    token_strings: List[str] = field(default_factory=list)


class InProcessVLLMClient:
    """vLLM running in-process via the ``vllm.LLM`` library API.

    The first call to ``chat`` lazily loads Qwen2.5-VL-7B-Instruct (or
    whatever ``VLLM_MODEL`` is set to) onto the local GPU. Subsequent
    calls reuse the same engine instance, serialised through
    ``_engine_lock``.
    """

    # Single engine shared across all instances + threads in this process.
    _engine = None
    _engine_lock = threading.Lock()
    _init_lock = threading.Lock()
    # Cache a failed init so we FAIL FAST instead of re-attempting a
    # full LLM() boot on every one of the N*agents calls (that produced
    # the EngineCore retry-storm).
    _engine_init_error: Optional[BaseException] = None

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        temperature: float = 0.8,
        seed: int = 42,
        max_new_tokens: int = 512,
        max_model_len: int = 32768,
        min_image_pixels: int = 50176,
        max_image_pixels: int = 1003520,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.90,
        **_ignored: Any,
    ):
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len
        self.min_image_pixels = min_image_pixels
        self.max_image_pixels = max_image_pixels
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization

        # Compat flags expected by VLLMClient callers — all no-ops here
        # since this client doesn't do guided/logprob scoring.
        self.chat_request_logprobs: bool = False
        self.prefer_structured_outputs: bool = False
        self.guided_scoring_enabled: bool = False
        self.top_logprobs: int = 0

    # ------------------------------------------------------------------
    # Lazy engine init
    # ------------------------------------------------------------------

    @staticmethod
    def _force_inprocess_engine_env() -> None:
        """vLLM v1 spawns an EngineCore SUBPROCESS by default. When the
        parent has already touched CUDA (our pipeline imports torch via
        agents/cache) the spawned core dies with::

            RuntimeError: CUDA unknown error ... Setting the available
            devices to be zero.

        and because init is lazy + per-call it retry-storms. Run the
        engine core IN-PROCESS (no subprocess) and use spawn for any
        residual worker. Must be set BEFORE importing vllm.
        """
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    def warmup(self) -> None:
        """Build the engine NOW, on the calling thread. The pipeline
        calls this on the MAIN thread before any ThreadPoolExecutor —
        vLLM must not be constructed inside a worker thread."""
        self._ensure_engine()

    def _ensure_engine(self):
        """Load the LLM once; subsequent calls return the cached engine.
        A failed init is cached and re-raised immediately (fail fast —
        no per-call EngineCore retry-storm)."""
        if InProcessVLLMClient._engine is not None:
            return InProcessVLLMClient._engine
        if InProcessVLLMClient._engine_init_error is not None:
            raise InProcessVLLMClient._engine_init_error
        with InProcessVLLMClient._init_lock:
            if InProcessVLLMClient._engine is not None:
                return InProcessVLLMClient._engine
            if InProcessVLLMClient._engine_init_error is not None:
                raise InProcessVLLMClient._engine_init_error
            try:
                self._force_inprocess_engine_env()
                # Lazy import — vllm is heavyweight and not available on
                # CPU-only LOCAL hosts.
                from vllm import LLM  # type: ignore

                logger.info(
                    "[vllm_inproc] loading %s (max_model_len=%d, "
                    "image pixels %d..%d, gpu_mem=%.2f) — engine core "
                    "in-process",
                    self.model, self.max_model_len,
                    self.min_image_pixels, self.max_image_pixels,
                    self.gpu_memory_utilization,
                )
                engine = LLM(
                    model=self.model,
                    trust_remote_code=True,
                    max_model_len=self.max_model_len,
                    limit_mm_per_prompt={"image": 1},
                    mm_processor_kwargs={
                        "min_pixels": self.min_image_pixels,
                        "max_pixels": self.max_image_pixels,
                    },
                    dtype=self.dtype,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                )
            except BaseException as e:  # noqa: BLE001
                InProcessVLLMClient._engine_init_error = e
                logger.error("[vllm_inproc] engine init FAILED (cached, "
                             "will fail fast): %s: %s",
                             type(e).__name__, e)
                raise
            InProcessVLLMClient._engine = engine
            logger.info("[vllm_inproc] engine ready")
            return engine

    # ------------------------------------------------------------------
    # Message preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_messages(
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str],
        image_b64: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Build the message list passed to ``LLM.chat()``.

        ``vllm.LLM.chat`` accepts OpenAI-style messages directly,
        including ``{"type": "image_url", "image_url": {"url": ...}}``
        with data URLs — so existing agent prompts pass through
        unchanged.
        """
        out: List[Dict[str, Any]] = []
        if system_prompt:
            out.append({"role": "system", "content": system_prompt})

        # If image_b64 is supplied as a separate arg (legacy code path
        # in VLLMClient), inject it into the first user message exactly
        # the way VLLMClient does.
        if image_b64 is not None:
            messages = [dict(m) for m in messages]
            for m in messages:
                if m.get("role") == "user":
                    original = m.get("content")
                    if isinstance(original, str):
                        original = [{"type": "text", "text": original}]
                    m["content"] = [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                            },
                        }
                    ] + list(original or [])
                    break

        out.extend(messages)
        return out

    # ------------------------------------------------------------------
    # Public API — matches VLLMClient
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        image_b64: Optional[str] = None,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[str, int]:
        result = self.chat_with_logprobs(
            messages=messages,
            image_b64=image_b64,
            system_prompt=system_prompt,
            seed=seed,
            temperature=temperature,
        )
        return result.text, result.completion_tokens

    def chat_with_logprobs(
        self,
        messages: List[Dict[str, Any]],
        image_b64: Optional[str] = None,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> ChatResult:
        from vllm import SamplingParams  # type: ignore

        engine = self._ensure_engine()
        prepared = self._prepare_messages(messages, system_prompt, image_b64)

        sp = SamplingParams(
            temperature=self.temperature if temperature is None else float(temperature),
            seed=self.seed if seed is None else int(seed),
            max_tokens=self.max_new_tokens,
        )

        # Serialise concurrent threads through the single engine.
        with InProcessVLLMClient._engine_lock:
            outputs = engine.chat(
                messages=prepared,
                sampling_params=sp,
                use_tqdm=False,
            )

        if not outputs:
            return ChatResult(text="", completion_tokens=0)

        first = outputs[0]
        comp_outputs = getattr(first, "outputs", None) or []
        if not comp_outputs:
            return ChatResult(text="", completion_tokens=0)
        comp = comp_outputs[0]

        text = getattr(comp, "text", None) or ""
        token_ids = getattr(comp, "token_ids", None) or []
        return ChatResult(text=text, completion_tokens=len(token_ids))

    def count_tokens(self, text: str) -> int:
        return len(text.split())


# ---------------------------------------------------------------------------
# Factory helper for callers that want one shared singleton
# ---------------------------------------------------------------------------

_GLOBAL: Optional[InProcessVLLMClient] = None
_GLOBAL_LOCK = threading.Lock()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def get_inproc_client() -> InProcessVLLMClient:
    """Return a process-wide singleton configured from env vars.

    Env knobs (matching the old VLLMClient knobs where they overlap):
        VLLM_MODEL                 Qwen/Qwen2.5-VL-7B-Instruct
        VLLM_TEMPERATURE           0.8
        VLLM_MAX_NEW_TOKENS        512
        VLLM_MAX_MODEL_LEN         32768
        VLLM_MIN_PIXELS            50176
        VLLM_MAX_PIXELS            1003520
        VLLM_DTYPE                 auto
        VLLM_GPU_MEMORY_UTIL       0.90
    """
    global _GLOBAL
    if _GLOBAL is not None:
        return _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is not None:
            return _GLOBAL
        _GLOBAL = InProcessVLLMClient(
            model=os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
            temperature=_float_env("VLLM_TEMPERATURE", 0.8),
            max_new_tokens=_int_env("VLLM_MAX_NEW_TOKENS", 512),
            max_model_len=_int_env("VLLM_MAX_MODEL_LEN", 32768),
            min_image_pixels=_int_env("VLLM_MIN_PIXELS", 50176),
            max_image_pixels=_int_env("VLLM_MAX_PIXELS", 1003520),
            dtype=os.environ.get("VLLM_DTYPE", "auto"),
            gpu_memory_utilization=_float_env("VLLM_GPU_MEMORY_UTIL", 0.90),
        )
        return _GLOBAL
