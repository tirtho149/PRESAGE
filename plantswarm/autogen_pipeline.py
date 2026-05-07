"""
plantswarm/autogen_pipeline.py
==============================
AutoGen AgentChat **Swarm** orchestration for PlantSwarm (Microsoft AutoGen pattern).

Follows the public Swarm design: a shared message context, next speaker chosen from the
latest ``HandoffMessage``, and handoffs implemented as tool calls (``transfer_to_<target>``).
Models must support function calling; we set ``parallel_tool_calls=False`` on
``OpenAIChatCompletionClient`` so only one handoff runs per turn (see AutoGen Swarm docs).

Participant order matches ``Swarm``: the **first** agent is the initial speaker; each
non-terminal agent must invoke exactly one handoff tool after its reply so routing
does not stall on the same speaker.

Also includes **local Hugging Face Qwen** (text-only) ``ChatCompletionClient`` and
``run_local_qwen_text_swarm_demo`` for Colab-style runs without vLLM; wired from
``scripts/run_plantswarm.py --local-qwen-text-demo``.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
)

from agents.base_agent import AgentOutput, ContextEntry
from agents.diagnosis_agent import DiagnosisAgent
from agents.morphology_agent import MorphologyAgent
from agents.pathogen_agent import PathogenAgent
from agents.severity_agent import SeverityAgent
from agents.symptom_agent import SymptomAgent
from calibration.ensemble import argmax_label, ensemble_probabilities
from .pipeline import RoutingTrace
from utils.vllm_client import VLLMClient

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
    from autogen_agentchat.messages import MultiModalMessage
    from autogen_agentchat.teams import Swarm
    from autogen_agentchat.ui import Console
    from autogen_ext.models.openai import OpenAIChatCompletionClient
except Exception:  # pragma: no cover
    AssistantAgent = None
    Console = None
    MaxMessageTermination = None
    MultiModalMessage = None
    OpenAIChatCompletionClient = None
    Swarm = None
    TextMentionTermination = None

try:
    from autogen_core import CancellationToken, FunctionCall, Image
    from autogen_core.models import (
        AssistantMessage,
        ChatCompletionClient,
        CreateResult,
        FunctionExecutionResultMessage,
        LLMMessage,
        ModelCapabilities,
        ModelFamily,
        ModelInfo,
        RequestUsage,
        SystemMessage,
        UserMessage,
    )
    from autogen_core.tools import Tool, ToolSchema
    from pydantic import BaseModel
except Exception:  # pragma: no cover
    BaseModel = None  # type: ignore[misc, assignment]
    CancellationToken = None  # type: ignore[misc, assignment]
    ChatCompletionClient = object  # type: ignore[misc, assignment]
    CreateResult = None  # type: ignore[misc, assignment]
    FunctionCall = None  # type: ignore[misc, assignment]
    Image = None  # type: ignore[misc, assignment]
    LLMMessage = None  # type: ignore[misc, assignment]
    ModelCapabilities = None  # type: ignore[misc, assignment]
    ModelFamily = None  # type: ignore[misc, assignment]
    ModelInfo = None  # type: ignore[misc, assignment]
    RequestUsage = None  # type: ignore[misc, assignment]
    SystemMessage = None  # type: ignore[misc, assignment]
    UserMessage = None  # type: ignore[misc, assignment]
    AssistantMessage = None  # type: ignore[misc, assignment]
    FunctionExecutionResultMessage = None  # type: ignore[misc, assignment]
    Tool = None  # type: ignore[misc, assignment]
    ToolSchema = None  # type: ignore[misc, assignment]


def _swarm_handoff_instruction(handoff_targets: List[str]) -> str:
    """Append to system prompt so the VLM uses AutoGen handoff tools, not text-only routing."""
    if not handoff_targets:
        return (
            "\n\n--- AutoGen Swarm (terminal agent) ---\n"
            "You have no handoff tools. Emit the required JSON, then a single line containing "
            "only TERMINATE. Do not call transfer tools or hand off to another agent."
        )
    examples = ", ".join(f"transfer_to_{t.lower()}" for t in handoff_targets[:3])
    if len(handoff_targets) > 3:
        examples += ", …"
    return (
        "\n\n--- AutoGen Swarm (routing) ---\n"
        "Swarm advances only via handoff tool calls (e.g. "
        f"{examples}), not by naming the next agent in prose alone. "
        "After your task output and reasoning, you MUST invoke exactly one handoff tool "
        f"that matches your intended next specialist. Allowed targets: {', '.join(handoff_targets)}. "
        "Use at most one handoff call per turn (parallel tool calls are disabled on the client)."
    )


class AutoGenPlantSwarmPipeline:
    """AutoGen Swarm version of PlantSwarm."""

    def __init__(
        self,
        client: VLLMClient,
        label_space: Dict[str, List[str]],
        Tmax: int = 15,
        confidence_weights: Optional[Dict[str, int]] = None,
        pathome_db: Optional[object] = None,
    ):
        if any(
            dep is None
            for dep in (
                AssistantAgent,
                MaxMessageTermination,
                MultiModalMessage,
                OpenAIChatCompletionClient,
                Swarm,
                TextMentionTermination,
            )
        ):
            raise ImportError(
                "AutoGen dependencies are missing. Install with: "
                "pip install autogen-agentchat autogen-ext[openai]"
            )

        self.client = client
        self.label_space = label_space
        self.Tmax = Tmax
        self.confidence_weights = confidence_weights or {"high": 3, "medium": 2, "low": 1}
        self.pathome_db = pathome_db

        ag_kwargs = {"pathome_db": pathome_db} if pathome_db is not None else {}
        self.parsers: Dict[str, Any] = {
            "MorphologyAgent": MorphologyAgent(client, label_space, **ag_kwargs),
            "SymptomAgent": SymptomAgent(client, label_space, **ag_kwargs),
            "PathogenAgent": PathogenAgent(client, label_space, **ag_kwargs),
            "SeverityAgent": SeverityAgent(client, label_space, **ag_kwargs),
            "DiagnosisAgent": DiagnosisAgent(client, label_space, **ag_kwargs),
        }

    def _build_swarm(self):
        model_client = OpenAIChatCompletionClient(
            model=self.client.model,
            base_url=self.client.base_url,
            api_key="EMPTY",
            temperature=self.client.temperature,
            max_tokens=self.client.max_new_tokens,
            # Swarm docs: disable parallel tool calls so multiple handoffs cannot fire at once.
            parallel_tool_calls=False,
            model_info={
                "vision": True,
                "function_calling": True,
                "json_output": True,
                "structured_output": False,
                "family": "unknown",
            },
        )

        # handoffs= matches each agent's HANDOFF_MENU (Algorithm 1); first participant = entry speaker.
        morph = AssistantAgent(
            "MorphologyAgent",
            model_client=model_client,
            handoffs=self.parsers["MorphologyAgent"].HANDOFF_MENU,
            system_message=self.parsers["MorphologyAgent"].SYSTEM_PROMPT
            + _swarm_handoff_instruction(self.parsers["MorphologyAgent"].HANDOFF_MENU),
        )
        sym = AssistantAgent(
            "SymptomAgent",
            model_client=model_client,
            handoffs=self.parsers["SymptomAgent"].HANDOFF_MENU,
            system_message=self.parsers["SymptomAgent"].SYSTEM_PROMPT
            + _swarm_handoff_instruction(self.parsers["SymptomAgent"].HANDOFF_MENU),
        )
        patho = AssistantAgent(
            "PathogenAgent",
            model_client=model_client,
            handoffs=self.parsers["PathogenAgent"].HANDOFF_MENU,
            system_message=self.parsers["PathogenAgent"].SYSTEM_PROMPT
            + _swarm_handoff_instruction(self.parsers["PathogenAgent"].HANDOFF_MENU),
        )
        sev = AssistantAgent(
            "SeverityAgent",
            model_client=model_client,
            handoffs=self.parsers["SeverityAgent"].HANDOFF_MENU,
            system_message=self.parsers["SeverityAgent"].SYSTEM_PROMPT
            + _swarm_handoff_instruction(self.parsers["SeverityAgent"].HANDOFF_MENU),
        )
        diag = AssistantAgent(
            "DiagnosisAgent",
            model_client=model_client,
            handoffs=[],
            system_message=self.parsers["DiagnosisAgent"].SYSTEM_PROMPT
            + _swarm_handoff_instruction(self.parsers["DiagnosisAgent"].HANDOFF_MENU),
        )

        termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(max_messages=self.Tmax * 3)
        team = Swarm(
            participants=[morph, sym, patho, sev, diag],
            termination_condition=termination,
        )
        return team, model_client

    def _to_agent_output(
        self,
        source: str,
        text: str,
        context: List[ContextEntry],
        backtrack_count: int,
        usage: Any,
        image_b64: str,
    ) -> AgentOutput:
        parser = self.parsers[source]
        predictions, confidence, handoff = parser._parse_response(text, context, backtrack_count)  # noqa: SLF001
        log_probs = parser._score_all_tasks(parser._context_to_text(context), image_b64)  # noqa: SLF001

        tokens_used = 0
        if usage is not None:
            tokens_used = getattr(usage, "completion_tokens", 0) or 0

        return AgentOutput(
            agent_name=source,
            message=text,
            confidence=confidence,
            predictions=predictions,
            log_probs=log_probs,
            handoff_target=handoff,
            tokens_used=tokens_used,
            token_entropies=None,
            mean_entropy_H=None,
            entropy_dispersion_D=None,
            targeted_disease_entropy=None,
        )

    async def _run_swarm_once(self, image_b64: str):
        team, model_client = self._build_swarm()
        task = MultiModalMessage(
            content=[
                "Run PlantSwarm on this crop disease image. The team uses AutoGen Swarm: "
                "non-terminal agents hand off with the transfer_to_* tools; "
                "DiagnosisAgent finishes with JSON then the line TERMINATE.",
                f"data:image/jpeg;base64,{image_b64}",
            ],
            source="user",
        )
        try:
            result = await team.run(task=task)
        finally:
            await model_client.close()
        return result

    def run(self, image_id: str, image_b64: str) -> RoutingTrace:
        t0 = time.time()
        result = asyncio.run(self._run_swarm_once(image_b64))

        context: List[ContextEntry] = []
        path: List[str] = []
        agent_outputs: List[AgentOutput] = []
        agent_log_probs: Dict[str, Dict[str, Dict[str, float]]] = {}
        agent_confidences: Dict[str, str] = {}
        backtrack_count = 0
        total_tokens = 0
        early_terminated = False

        for msg in getattr(result, "messages", []):
            source = getattr(msg, "source", "")
            if source not in self.parsers:
                continue
            text = getattr(msg, "content", "")
            if not isinstance(text, str):
                continue

            output = self._to_agent_output(
                source=source,
                text=text,
                context=context,
                backtrack_count=backtrack_count,
                usage=getattr(msg, "models_usage", None),
                image_b64=image_b64,
            )

            context.append(
                ContextEntry(
                    agent_name=source,
                    message=text,
                    confidence=output.confidence,
                    log_probs=output.log_probs,
                )
            )
            path.append(source)
            agent_outputs.append(output)
            total_tokens += output.tokens_used
            agent_log_probs[source] = output.log_probs
            agent_confidences[source] = output.confidence
            if output.handoff_target == "MorphologyAgent" and source != "MorphologyAgent":
                backtrack_count += 1

        if path and path[-1] == "DiagnosisAgent":
            early_terminated = True

        synth_output = next((o for o in reversed(agent_outputs) if o.agent_name == "DiagnosisAgent"), None)
        if synth_output is None:
            synth_output = self._to_agent_output(
                source="DiagnosisAgent",
                text="{}",
                context=context,
                backtrack_count=backtrack_count,
                usage=None,
                image_b64=image_b64,
            )
            path.append("DiagnosisAgent")
            agent_outputs.append(synth_output)

        ensemble_probs: Dict[str, Dict[str, float]] = {}
        for task_id, labels in self.label_space.items():
            ensemble_probs[task_id] = ensemble_probabilities(
                agent_log_probs=agent_log_probs,
                agent_confidences=agent_confidences,
                task_id=task_id,
                label_list=labels,
                confidence_weights=self.confidence_weights,
            )

        synth_preds = synth_output.predictions
        final_predictions = {
            "T1": synth_preds.get("symptom_type", argmax_label(ensemble_probs.get("T1", {}))),
            "T2": synth_preds.get("pathogen_class", argmax_label(ensemble_probs.get("T2", {}))),
            "T3": synth_preds.get("disease_name", argmax_label(ensemble_probs.get("T3", {}))),
            "T4": synth_preds.get("severity_class", argmax_label(ensemble_probs.get("T4", {}))),
            "T5": synth_preds.get("crop_species", argmax_label(ensemble_probs.get("T5", {}))),
        }

        revisits = len(path) - len(set(path))
        loop_rate = revisits / max(len(path), 1)

        return RoutingTrace(
            image_id=image_id,
            path=path,
            path_length=len(path),
            backtrack_count=backtrack_count,
            loop_rate=loop_rate,
            early_terminated=early_terminated,
            total_tokens=total_tokens,
            agent_outputs=agent_outputs,
            final_predictions=final_predictions,
            ensemble_probs=ensemble_probs,
            wall_time_s=time.time() - t0,
            routing_signal="autogen_swarm",
        )


# ---------------------------------------------------------------------------
# Local Hugging Face Qwen (text) — notebook-style Swarm (no vLLM HTTP)
# Synced with the Colab pattern: shared HF model, ChatCompletionClient, entropy log.
# ---------------------------------------------------------------------------

@dataclass
class LocalQwenUncertaintyResult:
    text: str
    mean_entropy: float
    max_entropy: float
    max_entropy_pos: int
    avg_logprob: float
    kappa: str


local_qwen_uncertainty_log: dict[str, LocalQwenUncertaintyResult] = {}

_local_qwen_tokenizer = None
_local_qwen_model = None
_local_qwen_model_name: Optional[str] = None


_LOCAL_QWEN_DEPS_MSG = (
    "Local Qwen demo needs PyTorch + Transformers in this environment.\n"
    "  pip install torch transformers accelerate\n"
    "or install full project deps:\n"
    "  pip install -r requirements.txt"
)


def _require_torch_transformers_for_local_qwen() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ModuleNotFoundError as e:
        raise ImportError(_LOCAL_QWEN_DEPS_MSG) from e


def _ensure_local_qwen_hf(model_name: str) -> tuple[Any, Any]:
    global _local_qwen_tokenizer, _local_qwen_model, _local_qwen_model_name
    _require_torch_transformers_for_local_qwen()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (
        _local_qwen_model is not None
        and _local_qwen_model_name == model_name
        and _local_qwen_tokenizer is not None
    ):
        return _local_qwen_tokenizer, _local_qwen_model

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    _local_qwen_tokenizer = tok
    _local_qwen_model = model
    _local_qwen_model_name = model_name
    return tok, model


def _flatten_llm_messages_to_chat_dicts(messages: Sequence[Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, UserMessage):
            c = m.content
            if isinstance(c, str):
                text = c
            else:
                parts: list[str] = []
                for item in c:
                    if isinstance(item, str):
                        parts.append(item)
                    elif Image is not None and isinstance(item, Image):
                        parts.append("[image omitted — use vLLM / Qwen-VL for vision]")
                text = "\n".join(parts) if parts else ""
            out.append({"role": "user", "content": text})
        elif isinstance(m, AssistantMessage):
            c = m.content
            if isinstance(c, str):
                text = c
            else:
                text = "\n".join(f"{fc.name}({fc.arguments})" for fc in c)
            out.append({"role": "assistant", "content": text})
        elif isinstance(m, FunctionExecutionResultMessage):
            blob = "\n".join(f"{x.name}: {x.content}" for x in m.content)
            out.append({"role": "user", "content": f"Tool results:\n{blob}"})
    return out


def _local_qwen_tool_prompt_section(tools: Sequence[Any]) -> str:
    if not tools:
        return ""
    tool_descs: list[str] = []
    for t in tools:
        if isinstance(t, dict):
            name = str(t.get("name", ""))
            desc = str(t.get("description", ""))
        else:
            try:
                sch = t.schema  # type: ignore[union-attr]
                name = str(sch.get("name", getattr(t, "name", "")))
                desc = str(sch.get("description", getattr(t, "description", "")))
            except Exception:
                continue
        if name:
            tool_descs.append(f"- {name}: {desc}")
    if not tool_descs:
        return ""
    lines = "\n".join(tool_descs)
    return (
        f"\nYou have access to these handoff tools:\n{lines}\n"
        "To hand off, output EXACTLY this JSON on its own line:\n"
        '{"tool_call": {"name": "<tool_name>", "arguments": {}}}\n'
    )


class LocalQwenChatCompletionClient(ChatCompletionClient):
    """Hugging Face causal LM + tokenizer implementing AutoGen's ChatCompletionClient (text-only)."""

    def __init__(self, agent_name: str, model_name: str = "Qwen/Qwen2.5-3B-Instruct", max_new_tokens: int = 200):
        if ChatCompletionClient is object or RequestUsage is None:
            raise ImportError("autogen_core is required for LocalQwenChatCompletionClient")
        self.agent_name = agent_name
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    async def close(self) -> None:
        return

    def actual_usage(self) -> RequestUsage:
        return self._total_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Union[Tool, ToolSchema]] = []) -> int:
        return 0

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Union[Tool, ToolSchema]] = []) -> int:
        return 4096

    @property
    def capabilities(self) -> ModelCapabilities:
        return {
            "vision": False,
            "function_calling": True,
            "json_output": False,
        }

    @property
    def model_info(self) -> ModelInfo:
        return {
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "structured_output": False,
            "family": ModelFamily.UNKNOWN,
        }

    def _compute_uncertainty(self, scores: tuple, generated_ids: Any, input_len: int) -> LocalQwenUncertaintyResult:
        import numpy as np
        import torch
        import torch.nn.functional as F

        log_probs: list[float] = []
        entropies: list[float] = []
        for step, s in enumerate(scores):
            p = F.softmax(s[0], dim=-1)[0]
            token_id = int(generated_ids[input_len + step])
            log_probs.append(torch.log(p[token_id] + 1e-12).item())
            H = float(-(p * torch.log(p + 1e-12)).sum().item())
            entropies.append(H)

        ent_arr = np.array(entropies)
        mean_h = float(ent_arr.mean())
        kappa = "L" if mean_h < 1.5 else ("M" if mean_h < 3.0 else "H")

        return LocalQwenUncertaintyResult(
            text="",
            mean_entropy=mean_h,
            max_entropy=float(ent_arr.max()),
            max_entropy_pos=int(int(ent_arr.argmax())),
            avg_logprob=float(np.mean(log_probs)),
            kappa=kappa,
        )

    def _parse_tool_call(self, text: str, tools: Sequence[Any]) -> tuple[Optional[str], str]:
        match = re.search(r"\{.*\"tool_call\".*\}", text, re.DOTALL)
        if not match:
            return None, text
        try:
            parsed = json.loads(match.group())
            tool_name = parsed["tool_call"]["name"]
            clean_text = text[: match.start()].strip()
            return tool_name, clean_text
        except Exception:
            return None, text

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Union[Tool, ToolSchema]] = [],
        tool_choice: Union[Tool, Literal["auto", "required", "none"]] = "auto",
        json_output: Optional[Union[bool, type[BaseModel]]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        tokenizer, model = _ensure_local_qwen_hf(self.model_name)
        import torch
        chat_messages = _flatten_llm_messages_to_chat_dicts(messages)
        tool_prompt = _local_qwen_tool_prompt_section(tools)
        if tool_prompt and chat_messages:
            chat_messages[-1]["content"] = chat_messages[-1].get("content", "") + tool_prompt

        text = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = int(inputs.input_ids.shape[1])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
            )

        generated_ids = outputs.sequences[0]
        response_text = tokenizer.decode(generated_ids[input_len:], skip_special_tokens=True)

        scores = outputs.scores or ()
        if scores:
            u = self._compute_uncertainty(scores, generated_ids, input_len)
            u.text = response_text
            local_qwen_uncertainty_log[self.agent_name] = u
        else:
            local_qwen_uncertainty_log[self.agent_name] = LocalQwenUncertaintyResult(
                text=response_text,
                mean_entropy=0.0,
                max_entropy=0.0,
                max_entropy_pos=0,
                avg_logprob=0.0,
                kappa="M",
            )

        n_scores = len(outputs.scores) if outputs.scores else 0
        usage = RequestUsage(prompt_tokens=input_len, completion_tokens=n_scores)
        self._total_usage = RequestUsage(
            prompt_tokens=self._total_usage.prompt_tokens + usage.prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens + usage.completion_tokens,
        )

        tool_name, _clean = self._parse_tool_call(response_text, tools)

        if tool_name and FunctionCall is not None:
            return CreateResult(
                content=[
                    FunctionCall(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=tool_name,
                        arguments="{}",
                    )
                ],
                usage=usage,
                finish_reason="function_calls",
                cached=False,
            )

        return CreateResult(
            content=response_text,
            usage=usage,
            finish_reason="stop",
            cached=False,
        )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Union[Tool, ToolSchema]] = [],
        tool_choice: Union[Tool, Literal["auto", "required", "none"]] = "auto",
        json_output: Optional[Union[bool, type[BaseModel]]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        result = await self.create(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )
        yield result


async def run_local_qwen_text_swarm_demo(
    query: str,
    *,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
) -> None:
    """
    Notebook-equivalent: triage → specialist → reviewer Swarm on a text query, with entropy log.

    Requires: transformers, torch, accelerate (see requirements.txt). Not used by PlantSwarm image pipeline.
    """
    _require_torch_transformers_for_local_qwen()

    if any(
        x is None
        for x in (
            AssistantAgent,
            Console,
            MaxMessageTermination,
            Swarm,
            TextMentionTermination,
        )
    ):
        raise ImportError(
            "AutoGen AgentChat is missing. Install with: pip install autogen-agentchat autogen-ext[openai]"
        )
    if ChatCompletionClient is object:
        raise ImportError("autogen_core models are required for LocalQwenChatCompletionClient")

    local_qwen_uncertainty_log.clear()

    triage_client = LocalQwenChatCompletionClient("triage", model_name=model_name)
    specialist_client = LocalQwenChatCompletionClient("specialist", model_name=model_name)
    reviewer_client = LocalQwenChatCompletionClient("reviewer", model_name=model_name)

    triage_agent = AssistantAgent(
        "triage",
        model_client=triage_client,
        handoffs=["specialist", "reviewer"],
        system_message=(
            "You are a plant disease triage agent.\n"
            "Give an initial diagnosis based on the symptoms.\n"
            "If uncertain, hand off to specialist using the tool.\n"
            "If confident, hand off to reviewer using the tool."
        ),
    )
    specialist_agent = AssistantAgent(
        "specialist",
        model_client=specialist_client,
        handoffs=["reviewer"],
        system_message=(
            "You are a plant pathology specialist.\n"
            "Provide detailed diagnosis with reasoning.\n"
            "Then hand off to reviewer using the tool."
        ),
    )
    reviewer_agent = AssistantAgent(
        "reviewer",
        model_client=reviewer_client,
        handoffs=[],
        system_message=(
            "You are a diagnostic reviewer.\n"
            "Summarize the final diagnosis in 2-3 sentences.\n"
            "End with TERMINATE."
        ),
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(10)
    team = Swarm(
        [triage_agent, specialist_agent, reviewer_agent],
        termination_condition=termination,
    )

    print(f"\n{'=' * 60}\nQuery: {query}\n{'=' * 60}\n")
    await Console(team.run_stream(task=query))

    print("\n" + "=" * 60)
    print("UNCERTAINTY REPORT (local Qwen entropy)")
    print("=" * 60)
    for agent_name, u in local_qwen_uncertainty_log.items():
        label = {"L": "confident", "M": "borderline", "H": "uncertain"}[u.kappa]
        print(f"\n[{agent_name.upper()}]")
        print(f"  Mean entropy : {u.mean_entropy:.4f}")
        print(f"  Max entropy  : {u.max_entropy:.4f} @ pos {u.max_entropy_pos}")
        print(f"  Avg log prob : {u.avg_logprob:.4f}")
        print(f"  Kappa        : {u.kappa}  ({label})")

    await triage_client.close()
    await specialist_client.close()
    await reviewer_client.close()
