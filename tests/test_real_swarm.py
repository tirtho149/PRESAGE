"""
tests/test_real_swarm.py
========================
Tests for the 2-round real-swarm protocol with shared blackboard + cross_refs.

The swarm is "real" (not just parallel-in-isolation) because:
  - Round 1 emits each specialist's independent observation.
  - A shared blackboard collects all round-1 outputs.
  - Round 2 re-runs every specialist WITH the blackboard, letting
    peers refine, support, challenge, or withdraw findings.
  - The consolidator sees BOTH rounds + cross_refs, walks CoT, emits
    the final delta list.

These tests don't exercise vLLM — they assert structural properties
and parse correctness.
"""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# AgentDeltaOutput schema
# ---------------------------------------------------------------------------

def test_agent_delta_output_has_round_and_cross_refs():
    from agents.base_agent import AgentDeltaOutput
    o = AgentDeltaOutput(agent_name="StemPithAgent")
    assert hasattr(o, "round_idx")
    assert hasattr(o, "cross_refs")
    assert o.round_idx == 1
    assert o.cross_refs == []


def test_cross_ref_actions_set():
    from agents.base_agent import CROSS_REF_ACTIONS
    assert set(CROSS_REF_ACTIONS) == {"support", "challenge", "withdraw"}


# ---------------------------------------------------------------------------
# parse_agent_output now returns cross_refs as the 4th tuple element
# ---------------------------------------------------------------------------

def test_parse_round2_output_with_support_action():
    from agents.base_agent import parse_agent_output
    raw = json.dumps({
        "deltas": [{
            "field":          "stem_pith",
            "canonical_says": "(not specified)",
            "image_shows":    "white pith with chocolate-brown vascular ring",
            "image_quote":    "split lower stem clearly shows the white center",
        }],
        "cross_refs": [{
            "action":       "support",
            "target_agent": "DefoliationAgent",
            "rationale":    "bare-petiole skeletons also point to SDS",
        }],
        "confidence": "high",
        "reasoning":  "consistent with SDS",
    })
    deltas, conf, _why, refs = parse_agent_output(raw, owned_fields=["stem_pith"])
    assert len(deltas) == 1
    assert conf == "high"
    assert len(refs) == 1
    assert refs[0]["action"] == "support"
    assert refs[0]["target_agent"] == "DefoliationAgent"
    assert "SDS" in refs[0]["rationale"]


def test_parse_round2_output_with_challenge_action():
    from agents.base_agent import parse_agent_output
    raw = json.dumps({
        "deltas": [],
        "cross_refs": [{
            "action":       "challenge",
            "target_agent": "LeafLesionColorAgent",
            "rationale":    "colors look tan, not brown as reported",
        }],
        "confidence": "medium",
        "reasoning":  "challenging peer color call",
    })
    _deltas, _conf, _why, refs = parse_agent_output(raw, owned_fields=["color_palette"])
    assert len(refs) == 1
    assert refs[0]["action"] == "challenge"


def test_parse_round2_withdraw_action():
    from agents.base_agent import parse_agent_output
    raw = json.dumps({
        "deltas": [],
        "cross_refs": [{
            "action":       "withdraw",
            "target_agent": "StemPithAgent",   # self-withdraw uses own name
            "rationale":    "after seeing peer outputs, my round-1 was wrong",
        }],
        "confidence": "low",
        "reasoning":  "self-withdraw",
    })
    _deltas, _conf, _why, refs = parse_agent_output(raw, owned_fields=["stem_pith"])
    assert len(refs) == 1
    assert refs[0]["action"] == "withdraw"


def test_parse_unknown_cross_ref_action_is_dropped():
    """Only the three canonical actions survive; anything else is silently
    dropped (not coerced to 'other' — that would be misleading for
    cross-agent provenance)."""
    from agents.base_agent import parse_agent_output
    raw = json.dumps({
        "deltas": [],
        "cross_refs": [
            {"action": "support",   "target_agent": "X", "rationale": "ok"},
            {"action": "nonsense",  "target_agent": "Y", "rationale": "bad"},
            {"action": "challenge", "target_agent": "Z", "rationale": "ok"},
        ],
        "confidence": "medium",
        "reasoning":  "mixed",
    })
    _deltas, _conf, _why, refs = parse_agent_output(raw, owned_fields=["stem_pith"])
    actions = [r["action"] for r in refs]
    assert "nonsense" not in actions
    assert actions == ["support", "challenge"]


def test_parse_cross_ref_without_rationale_dropped():
    """A cross_ref with no rationale is unfalsifiable — drop it."""
    from agents.base_agent import parse_agent_output
    raw = json.dumps({
        "deltas": [],
        "cross_refs": [{"action": "support", "target_agent": "X"}],
        "confidence": "medium",
        "reasoning":  "x",
    })
    _deltas, _conf, _why, refs = parse_agent_output(raw, owned_fields=["stem_pith"])
    assert refs == []


# ---------------------------------------------------------------------------
# Blackboard rendering
# ---------------------------------------------------------------------------

def test_blackboard_format_excludes_self():
    """An agent reading its own AGENT_NAME from the blackboard would
    be reading its round-1 output back — wasteful. Verify exclude works."""
    from agents.base_agent import AgentDeltaOutput, BaseAgent

    bb = {
        "StemPithAgent": AgentDeltaOutput(
            agent_name="StemPithAgent",
            deltas=[{"field": "stem_pith", "canonical_says": "(n/s)",
                     "image_shows": "white pith", "image_quote": "X"}],
            confidence="high",
            reasoning="white-pith fork: SDS",
        ),
        "DefoliationAgent": AgentDeltaOutput(
            agent_name="DefoliationAgent",
            deltas=[{"field": "defoliation", "canonical_says": "(n/s)",
                     "image_shows": "bare-petiole skeletons",
                     "image_quote": "Y"}],
            confidence="high",
            reasoning="petiole-attached fork: SDS",
        ),
    }
    out_pith = BaseAgent._format_blackboard(bb, exclude="StemPithAgent")
    assert "StemPithAgent" not in out_pith
    assert "DefoliationAgent" in out_pith
    assert "bare-petiole" in out_pith


def test_blackboard_empty_renders_marker():
    from agents.base_agent import BaseAgent
    out = BaseAgent._format_blackboard({}, exclude=None)
    assert "empty" in out.lower()


# ---------------------------------------------------------------------------
# Consolidator splits round-1 / round-2 + collects cross_refs
# ---------------------------------------------------------------------------

def test_consolidator_collect_cross_refs_groups_by_action():
    from agents.base_agent import AgentDeltaOutput
    from agents.diagnosis_agent import DiagnosisAgent

    round2 = [
        AgentDeltaOutput(
            agent_name="StemPithAgent",
            cross_refs=[{"action": "support", "target_agent": "DefoliationAgent",
                         "rationale": "SDS confirmed by petioles"}],
            round_idx=2,
        ),
        AgentDeltaOutput(
            agent_name="ColorPaletteAgent",
            cross_refs=[{"action": "challenge", "target_agent": "LeafLesionColorAgent",
                         "rationale": "tan not brown"}],
            round_idx=2,
        ),
    ]
    refs = DiagnosisAgent._collect_cross_refs(round2)
    assert len(refs) == 2
    actions = {r["action"] for r in refs}
    assert actions == {"support", "challenge"}
    # Each ref tags the originating agent.
    sources = {r["from"] for r in refs}
    assert sources == {"StemPithAgent", "ColorPaletteAgent"}


def test_consolidator_format_cross_refs_groups_by_action():
    from agents.diagnosis_agent import DiagnosisAgent
    refs = [
        {"from": "A", "action": "support",   "target_agent": "B", "rationale": "ok"},
        {"from": "C", "action": "challenge", "target_agent": "D", "rationale": "bad"},
        {"from": "E", "action": "withdraw",  "target_agent": "(self)", "rationale": "oops"},
    ]
    out = DiagnosisAgent._format_cross_refs(refs)
    # Each action header is uppercased.
    assert "CHALLENGE" in out
    assert "SUPPORT"   in out
    assert "WITHDRAW"  in out
    # Each rationale appears.
    assert "ok"  in out
    assert "bad" in out
    assert "oops" in out


# ---------------------------------------------------------------------------
# Round-2 prompt template exists and references peer-blackboard concepts
# ---------------------------------------------------------------------------

def test_round2_prompt_template_references_blackboard_and_actions():
    from agents.base_agent import DELTA_USER_PROMPT_R2
    # The round-2 template must mention the blackboard and the cross-ref
    # action vocabulary so the model knows what it's allowed to emit.
    assert "BLACKBOARD" in DELTA_USER_PROMPT_R2 or "blackboard" in DELTA_USER_PROMPT_R2.lower()
    for action in ("support", "challenge", "withdraw"):
        assert action in DELTA_USER_PROMPT_R2.lower(), f"missing {action!r}"
    # And it must reference ROUND 2 explicitly.
    assert "ROUND 2" in DELTA_USER_PROMPT_R2
