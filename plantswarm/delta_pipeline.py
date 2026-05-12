"""
plantswarm/delta_pipeline.py
============================
Paper-faithful Qwen swarm for regional delta extraction (PlantSwarm §4 /
Algorithm 1, adapted for deltas).

Per (crop, disease, state, cached Bugwood image):

    N stochastic traces  →  per-trace consolidated deltas
                         →  cross-run agreement filter
                         →  final regional deltas

A single trace is a routed traversal of the swarm:

    entry agent (MorphologyAgent)
        ↓ kappa-gated handoff
    next agent (model-chosen, overridden by Algorithm 1)
        ↓ kappa-gated handoff
    ...                                      ← context buffer grows at each step
        ↓
    DiagnosisAgent (terminal consolidator) → per-trace delta list

Algorithm 1 (kappa = confidence ∈ {high, medium, low}, b = backtrack count):

    κ=low  AND b == 0           → MorphologyAgent (regrounding)
    κ=low  AND b >= 1           → default forward (loop guard)
    κ=high AND all specialists ran → DiagnosisAgent (early terminate)
    otherwise                   → model's chosen handoff

After N traces, deltas are clustered by (field, image_shows Jaccard
similarity) and clusters with support ≥ K (number of distinct runs that
emitted a matching delta) are kept. The final set is then passed
through a single consolidator pass for shape sanity (idempotent when
agreement already deduped well).

Output (per state) matches what pathome_kb.symptoms_adapter expects:
    {
      "state":         "Alabama",
      "deltas":        [{field, canonical_says, image_shows, image_quote, image_id}, ...],
      "__image_ids__": ["bugwood::1568038", ...],
      "__swarm_meta__": {n_runs, agreement_min, paths, ...},
    }

Configuration via env vars (read at client-build time):
    VLLM_BASE_URL          default http://localhost:8000/v1
    VLLM_MODEL             default Qwen/Qwen2.5-VL-7B-Instruct
    VLLM_TIMEOUT           seconds per HTTP call (default 180)
    VLLM_TEMPERATURE       per-call sampling temperature (default 0.8)
    VLLM_N_RUNS            stochastic traces per tuple (default 10)
    VLLM_AGREEMENT_MIN     min K-of-N agreement to keep a delta (default 3)
    VLLM_TMAX              max path length per trace (default 15)
    VLLM_MAX_BACKTRACKS    max backtracks per trace (default 1)
    VLLM_SIM_THRESHOLD     Jaccard threshold for delta clustering (default 0.4)
"""

from __future__ import annotations

import base64
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agents.base_agent import AgentDeltaOutput, BaseAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.morphology_agent import MorphologyAgent
from agents.pathogen_agent import PathogenAgent
from agents.severity_agent import SeverityAgent
from agents.symptom_agent import SymptomAgent
from utils.vllm_client import VLLMClient


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

SPECIALIST_NAMES = ("MorphologyAgent", "SymptomAgent", "PathogenAgent", "SeverityAgent")

_AGENT_REGISTRY: Dict[str, type] = {
    "MorphologyAgent": MorphologyAgent,
    "SymptomAgent":    SymptomAgent,
    "PathogenAgent":   PathogenAgent,
    "SeverityAgent":   SeverityAgent,
    "DiagnosisAgent":  DiagnosisAgent,
}


def _make_agent(name: str, client: VLLMClient) -> BaseAgent:
    cls = _AGENT_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"unknown agent: {name}")
    return cls(client)


# ---------------------------------------------------------------------------
# Swarm config
# ---------------------------------------------------------------------------

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


def build_client_from_env() -> VLLMClient:
    """Build a VLLMClient from environment variables."""
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    model    = os.environ.get("VLLM_MODEL",    "Qwen/Qwen2.5-VL-7B-Instruct")
    timeout  = _int_env("VLLM_TIMEOUT", 180)
    temperature = _float_env("VLLM_TEMPERATURE", 0.8)
    client = VLLMClient(
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout=timeout,
    )
    client.chat_request_logprobs = False     # not needed for delta mode
    return client


# ---------------------------------------------------------------------------
# Canonical flattener
# ---------------------------------------------------------------------------

def flatten_canonical(record: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a SAGE final_registry.json disease record to plain values."""
    def _v(field: Any) -> Any:
        if not isinstance(field, dict):
            return field
        return field.get("value")

    visual = record.get("visual_symptoms") or {}
    return {
        "summary":                  _v(visual.get("summary"))               or "",
        "diagnostic_features":      _v(visual.get("diagnostic_features"))   or [],
        "look_alikes":              _v(visual.get("look_alikes"))           or [],
        "affected_parts":           _v(record.get("affected_parts"))        or [],
        "treatments":               _v(record.get("treatments"))            or [],
        "pathogen_scientific_name": _v(record.get("pathogen_scientific_name")) or "",
        "type_of_disease":          _v(record.get("type_of_disease"))       or "",
        "notes":                    _v(record.get("notes"))                 or "",
    }


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_image_b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Algorithm 1 — routing decision
# ---------------------------------------------------------------------------

def algorithm1_handoff(
    *,
    current_agent_name: str,
    model_handoff: Optional[str],
    confidence: str,
    backtrack_count: int,
    max_backtracks: int,
    specialists_run: set,
    default_forward: str,
) -> Tuple[Optional[str], str]:
    """Apply paper Algorithm 1 to override the model's chosen handoff.

    Returns (next_agent_or_None, decision_reason). ``None`` means
    terminate. Reason string is for trace logging.
    """
    # Rule 1: low confidence + no backtrack → MorphologyAgent (regrounding).
    if confidence == "low" and backtrack_count == 0 and current_agent_name != "MorphologyAgent":
        return "MorphologyAgent", "alg1_low_kappa_backtrack"

    # Rule 2: low confidence + already backtracked → default forward (loop guard).
    if confidence == "low" and backtrack_count >= max_backtracks:
        return default_forward or "DiagnosisAgent", "alg1_loop_guard_forward"

    # Rule 3: high confidence + all specialists already contributed → terminate.
    if (
        confidence == "high"
        and len(specialists_run) >= len(SPECIALIST_NAMES)
        and current_agent_name != "DiagnosisAgent"
    ):
        return "DiagnosisAgent", "alg1_high_kappa_all_covered_terminate"

    # Rule 4: otherwise use the model's choice (or fall back to default forward).
    if model_handoff:
        return model_handoff, "model_choice"
    return default_forward or "DiagnosisAgent", "default_forward"


# ---------------------------------------------------------------------------
# One stochastic trace
# ---------------------------------------------------------------------------

def _run_single_trace(
    *,
    crop: str,
    disease: str,
    state: str,
    canonical: Dict[str, Any],
    image_b64: str,
    client: VLLMClient,
    run_idx: int,
    seed: int,
    temperature: float,
    entry_agent: str = "MorphologyAgent",
    Tmax: int = 15,
    max_backtracks: int = 1,
) -> Dict[str, Any]:
    """One routed traversal of the swarm. Returns a trace record:

        {
          "path":             [agent_name, ...],
          "context_buffer":   [AgentDeltaOutput, ...],
          "final_deltas":     [delta, ...],            ← from DiagnosisAgent
          "confidences":      [kappa, ...],
          "decisions":        [reason_str, ...],
          "backtrack_count":  int,
          "early_terminated": bool,
        }
    """
    context_buffer: List[AgentDeltaOutput] = []
    path: List[str] = []
    decisions: List[str] = []
    backtrack_count = 0
    specialists_run: set = set()

    current = entry_agent
    early_terminated = False

    while len(path) < Tmax:
        if current == "DiagnosisAgent":
            # Terminal consolidation.
            consolidator = DiagnosisAgent(client)
            out = consolidator.consolidate(
                crop=crop,
                disease=disease,
                state=state,
                canonical=canonical,
                image_b64=image_b64,
                context_buffer=context_buffer,
                seed=seed + 1000,        # different seed slot from specialists
                temperature=temperature,
            )
            path.append(current)
            context_buffer.append(out)
            decisions.append("terminate_diagnosis")
            early_terminated = True
            break

        agent = _make_agent(current, client)
        out = agent.extract_with_routing(
            crop=crop,
            disease=disease,
            state=state,
            canonical=canonical,
            image_b64=image_b64,
            prior_context=context_buffer,
            seed=seed + len(path),       # vary seed across steps
            temperature=temperature,
        )
        path.append(current)
        context_buffer.append(out)
        if current in SPECIALIST_NAMES:
            specialists_run.add(current)

        # Algorithm 1 routing decision.
        nxt, reason = algorithm1_handoff(
            current_agent_name=current,
            model_handoff=out.handoff_target,
            confidence=out.confidence,
            backtrack_count=backtrack_count,
            max_backtracks=max_backtracks,
            specialists_run=specialists_run,
            default_forward=agent.DEFAULT_FORWARD,
        )
        decisions.append(reason)

        if nxt is None:
            break

        # Bookkeeping for backtracks (jumping back to MorphologyAgent counts).
        if nxt == "MorphologyAgent" and current != "MorphologyAgent":
            backtrack_count += 1

        current = nxt

    # If we hit Tmax without DiagnosisAgent, force a terminal consolidation now.
    if not early_terminated:
        consolidator = DiagnosisAgent(client)
        out = consolidator.consolidate(
            crop=crop,
            disease=disease,
            state=state,
            canonical=canonical,
            image_b64=image_b64,
            context_buffer=context_buffer,
            seed=seed + 1000,
            temperature=temperature,
        )
        path.append("DiagnosisAgent")
        context_buffer.append(out)
        decisions.append("tmax_forced_terminate")

    return {
        "run_idx":          run_idx,
        "path":             path,
        "context_buffer":   context_buffer,
        "final_deltas":     context_buffer[-1].deltas,
        "confidences":      [c.confidence for c in context_buffer],
        "decisions":        decisions,
        "backtrack_count":  backtrack_count,
        "early_terminated": early_terminated,
    }


# ---------------------------------------------------------------------------
# Cross-run agreement filter
# ---------------------------------------------------------------------------

def _tokenize(s: str) -> set:
    """Cheap tokenizer for Jaccard similarity. Lowercase, strip punctuation."""
    if not s:
        return set()
    out = set()
    for tok in s.lower().split():
        cleaned = "".join(ch for ch in tok if ch.isalnum())
        if cleaned:
            out.add(cleaned)
    return out


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cluster_by_similarity(
    items: List[Tuple[int, Dict[str, str]]],
    threshold: float,
) -> List[List[Tuple[int, Dict[str, str]]]]:
    """Greedy Jaccard clustering on image_shows tokens.

    A delta joins a cluster if its Jaccard similarity to ANY existing
    cluster member is ≥ threshold. Using any-member matching (vs.
    single-representative) lets transitively-similar deltas cluster even
    when their phrasings don't directly overlap heavily — e.g.
    "raised pustular lesions with chlorotic halos" and "yellow halos
    around dark pustular lesions" can join via a bridge phrasing.
    """
    clusters: List[List[Tuple[int, Dict[str, str]]]] = []
    for run_idx, d in items:
        d_tokens = _tokenize(d.get("image_shows", ""))
        placed = False
        for cluster in clusters:
            for _, member in cluster:
                m_tokens = _tokenize(member.get("image_shows", ""))
                if _jaccard(d_tokens, m_tokens) >= threshold:
                    cluster.append((run_idx, d))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            clusters.append([(run_idx, d)])
    return clusters


def _agreement_filter(
    per_run_deltas: List[List[Dict[str, str]]],
    *,
    min_support: int,
    similarity_threshold: float = 0.4,
) -> List[Dict[str, str]]:
    """Keep deltas that survived K-of-N agreement.

    A delta "survives" iff, when clustered with all other deltas of the
    SAME field by image_shows token Jaccard ≥ ``similarity_threshold``,
    the cluster covers ≥ ``min_support`` DISTINCT runs.
    """
    all_with_run: List[Tuple[int, Dict[str, str]]] = []
    for run_idx, deltas in enumerate(per_run_deltas):
        for d in deltas or []:
            all_with_run.append((run_idx, d))

    by_field: Dict[str, List[Tuple[int, Dict[str, str]]]] = defaultdict(list)
    for run_idx, d in all_with_run:
        by_field[d.get("field", "other")].append((run_idx, d))

    survivors: List[Dict[str, str]] = []
    for fld, items in by_field.items():
        clusters = _cluster_by_similarity(items, threshold=similarity_threshold)
        for cluster in clusters:
            run_set = {ri for ri, _ in cluster}
            if len(run_set) >= min_support:
                # Representative = the delta in the cluster with the longest
                # image_shows (typically the most specific).
                rep = max((d for _, d in cluster),
                          key=lambda d: len(d.get("image_shows", "")))
                rep = dict(rep)
                rep["__support__"]     = len(run_set)
                rep["__cluster_size__"] = len(cluster)
                survivors.append(rep)
    return survivors


# ---------------------------------------------------------------------------
# Top-level: one (crop, disease, state) tuple → final deltas via N traces
# ---------------------------------------------------------------------------

def run_for_state(
    *,
    crop: str,
    disease: str,
    state: str,
    canonical_record: Dict[str, Any],
    image_path: Path,
    primary_image_id: str,
    client: Optional[VLLMClient] = None,
    n_runs: Optional[int] = None,
    agreement_min: Optional[int] = None,
    temperature: Optional[float] = None,
    Tmax: Optional[int] = None,
    max_backtracks: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    parallel_runs: bool = True,
    seed_base: int = 42,
) -> Dict[str, Any]:
    """N stochastic routed traces → cross-run agreement → final deltas.

    All ``None`` knobs fall back to the corresponding ``VLLM_*`` env var
    or the documented default.
    """
    if client is None:
        client = build_client_from_env()

    n_runs              = n_runs              or _int_env  ("VLLM_N_RUNS",          10)
    agreement_min       = agreement_min       or _int_env  ("VLLM_AGREEMENT_MIN",    3)
    temperature         = temperature         or _float_env("VLLM_TEMPERATURE",      0.8)
    Tmax                = Tmax                or _int_env  ("VLLM_TMAX",            15)
    max_backtracks      = max_backtracks      or _int_env  ("VLLM_MAX_BACKTRACKS",   1)
    similarity_threshold = similarity_threshold or _float_env("VLLM_SIM_THRESHOLD",  0.4)
    # Floor agreement_min at 1 (K=0 would let pure hallucinations through).
    agreement_min = max(1, min(agreement_min, n_runs))

    canonical = flatten_canonical(canonical_record)
    image_b64 = _load_image_b64(image_path)

    def _one(i: int) -> Dict[str, Any]:
        return _run_single_trace(
            crop=crop,
            disease=disease,
            state=state,
            canonical=canonical,
            image_b64=image_b64,
            client=client,
            run_idx=i,
            seed=seed_base + i * 100,
            temperature=temperature,
            Tmax=Tmax,
            max_backtracks=max_backtracks,
        )

    traces: List[Dict[str, Any]] = []
    if parallel_runs and n_runs > 1:
        with ThreadPoolExecutor(max_workers=min(n_runs, 8)) as pool:
            for t in pool.map(_one, range(n_runs)):
                traces.append(t)
    else:
        for i in range(n_runs):
            traces.append(_one(i))

    # Cross-run agreement.
    per_run_final = [t["final_deltas"] for t in traces]
    survivors = _agreement_filter(
        per_run_final,
        min_support=agreement_min,
        similarity_threshold=similarity_threshold,
    )

    for d in survivors:
        d.setdefault("image_id", primary_image_id)

    return {
        "state":          state,
        "deltas":         survivors,
        "__image_ids__":  [primary_image_id],
        "__swarm_meta__": {
            "n_runs":               n_runs,
            "agreement_min":        agreement_min,
            "temperature":          temperature,
            "Tmax":                 Tmax,
            "max_backtracks":       max_backtracks,
            "similarity_threshold": similarity_threshold,
            "paths":                [t["path"] for t in traces],
            "path_lengths":         [len(t["path"]) for t in traces],
            "backtrack_counts":     [t["backtrack_count"] for t in traces],
            "early_terminated":     [t["early_terminated"] for t in traces],
            "n_raw_per_run":        [len(t["final_deltas"]) for t in traces],
            "n_after_agreement":    len(survivors),
        },
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

# (profile_id, crop, disease, state, image_path, image_ids, canonical_record,
#  primary_image_id)
WorkItem = Tuple[str, str, str, str, Path, List[str], Dict[str, Any], str]


def run_batch(
    work_items: Iterable[WorkItem],
    *,
    client: Optional[VLLMClient] = None,
    max_parallel: int = 4,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run the swarm across many (profile, state) tuples.

    Returns ``{profile_id: {state: record}}``.
    """
    if client is None:
        client = build_client_from_env()

    items = list(work_items)
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _worker(item: WorkItem) -> Tuple[str, str, Dict[str, Any]]:
        (profile_id, crop, disease, state, image_path,
         image_ids, canonical_record, primary_image_id) = item
        record = run_for_state(
            crop=crop,
            disease=disease,
            state=state,
            canonical_record=canonical_record,
            image_path=image_path,
            primary_image_id=primary_image_id,
            client=client,
        )
        record["__image_ids__"] = list(image_ids) or [primary_image_id]
        return profile_id, state, record

    completed = 0
    total = len(items)
    with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as pool:
        futures = {pool.submit(_worker, it): it for it in items}
        for fut in as_completed(futures):
            try:
                profile_id, state, record = fut.result()
            except Exception as e:
                it = futures[fut]
                print(f"    ERROR on {it[0]} / {it[3]}: {type(e).__name__}: {e}")
                continue
            completed += 1
            meta = record.get("__swarm_meta__", {})
            n_deltas = len(record.get("deltas") or [])
            tag = "✓" if n_deltas else "·"
            print(
                f"    [{completed}/{total}] {tag} {profile_id} / {state}  "
                f"deltas={n_deltas} (N={meta.get('n_runs')}, "
                f"K≥{meta.get('agreement_min')}, "
                f"raw/run={meta.get('n_raw_per_run')})"
            )
            results.setdefault(profile_id, {})[state] = record

    return results
