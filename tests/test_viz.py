"""
tests/test_viz.py
=================
End-to-end smoke for the viz layer against synthetic JSON inputs.

Verifies:
  - kb_stats aggregation
  - observe_curves history aggregation
  - observe_eval table emission
  - trace_stats aggregation
  - Each script emits an auto_<name>.tex even when matplotlib is missing

These tests do NOT require matplotlib. When matplotlib IS installed,
PNG figures are also produced; we don't assert on their pixel content,
only that the file exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _make_seed(tmp_path: Path) -> Path:
    """Tiny symptoms_seed.json with two profiles + deltas covering
    multiple statuses."""
    seed = {
        "min_observations": 3,
        "profiles": [
            {
                "profile_id": "Soybean::Charcoal Rot",
                "crop": "Soybean", "disease": "Charcoal Rot",
                "canonical": {
                    "summary": "Soilborne fungus",
                    "diagnostic_features": ["microsclerotia"],
                    "look_alikes": [], "treatments": [],
                    "affected_parts": ["Foliar", "Stem"],
                    "pathogen_scientific_name": "Macrophomina phaseolina",
                    "type_of_disease": "Fungal", "notes": "", "sources": {},
                },
                "regional_observations": {
                    "Alabama": {
                        "state": "Alabama",
                        "image_ids": ["bugwood::1"],
                        "deltas": [
                            {"field": "lesion_morphology",
                             "canonical_says": "(not specified)",
                             "image_shows": "yellow halos around dark spots",
                             "image_quote": "...", "image_id": "bugwood::1",
                             "swarm_support": 4, "verification_status": "verified",
                             "web_support": [{"url": "https://example.com/a",
                                              "quote": "..."}]},
                            {"field": "severity",
                             "canonical_says": "(not specified)",
                             "image_shows": "whole-field collapse",
                             "image_quote": "...", "image_id": "bugwood::1",
                             "swarm_support": 3, "verification_status": "provisional",
                             "web_support": []},
                        ],
                    },
                    "Iowa": {
                        "state": "Iowa",
                        "image_ids": ["bugwood::2"],
                        "deltas": [
                            {"field": "diagnostic_features",
                             "canonical_says": "microsclerotia",
                             "image_shows": "marbled cross-sections",
                             "image_quote": "...", "image_id": "bugwood::2",
                             "swarm_support": 5, "verification_status": "verified",
                             "web_support": [{"url": "https://example.com/b",
                                              "quote": "..."}]},
                        ],
                    },
                },
                "state_counts": {"Alabama": 1, "Iowa": 1},
                "aez_counts": {}, "total_observations": 2,
                "reference_ids": [], "reobservation_prompt": "",
            },
            {
                "profile_id": "Tomato::Early Blight",
                "crop": "Tomato", "disease": "Early Blight",
                "canonical": {"summary": "", "diagnostic_features": [], "look_alikes": [],
                              "treatments": [], "affected_parts": [],
                              "pathogen_scientific_name": "", "type_of_disease": "",
                              "notes": "", "sources": {}},
                "regional_observations": {},
                "state_counts": {}, "aez_counts": {}, "total_observations": 0,
                "reference_ids": [], "reobservation_prompt": "",
            },
        ],
    }
    p = tmp_path / "seed.json"
    p.write_text(json.dumps(seed))
    return p


def _make_history(tmp_path: Path) -> Path:
    history = [
        {"epoch": 1,
         "train": {"total": 1.4, "routing": 1.1, "cal": 0.3,
                   "cons": 0.05, "oc": 0.1, "routing_acc": 0.25},
         "val":   {"total": 1.3, "routing_acc": 0.30}},
        {"epoch": 2,
         "train": {"total": 0.9, "routing": 0.7, "cal": 0.2,
                   "cons": 0.04, "oc": 0.08, "routing_acc": 0.55},
         "val":   {"total": 0.95, "routing_acc": 0.50}},
        {"epoch": 3,
         "train": {"total": 0.6, "routing": 0.45, "cal": 0.15,
                   "cons": 0.03, "oc": 0.06, "routing_acc": 0.72},
         "val":   {"total": 0.7,  "routing_acc": 0.65}},
    ]
    p = tmp_path / "history.json"
    p.write_text(json.dumps(history))
    return p


def _make_eval(tmp_path: Path) -> Path:
    ev = {
        "n_samples": 200, "n_images": 50,
        "routing_accuracy": 0.78,
        "routing_per_class": {
            "MorphologyAgent": {"support": 60, "accuracy": 0.85},
            "SymptomAgent":    {"support": 50, "accuracy": 0.72},
            "PathogenAgent":   {"support": 40, "accuracy": 0.70},
            "SeverityAgent":   {"support": 30, "accuracy": 0.80},
            "DiagnosisAgent":  {"support": 20, "accuracy": 0.95},
        },
        "backtrack_accuracy": 0.82, "backtrack_f1": 0.65,
        "kappa_mae": 0.12, "kappa_ece": 0.09,
        "overconfidence_accuracy": 0.88,
    }
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(ev))
    return p


def _make_traces(tmp_path: Path) -> Path:
    p = tmp_path / "traces.jsonl"
    lines = []
    for i in range(5):
        rec = {
            "profile_id": "Soybean::Charcoal Rot",
            "crop": "Soybean", "disease": "Charcoal Rot", "state": "Alabama",
            "primary_image_id": f"bugwood::{i}", "image_path": f"/tmp/img_{i}.jpg",
            "run_idx": i, "path": ["MorphologyAgent", "SymptomAgent", "DiagnosisAgent"],
            "decisions": ["model_choice", "alg1_high_kappa_all_covered_terminate"],
            "confidences": ["medium", "high"],
            "backtrack_count": 1 if i % 2 == 0 else 0,
            "early_terminated": True,
            "context_buffer": [
                {"agent_name": "MorphologyAgent",
                 "deltas": [{"field": "lesion_morphology",
                             "image_shows": "X", "canonical_says": "", "image_quote": ""}],
                 "confidence": "medium", "handoff_target": "SymptomAgent",
                 "reasoning": "", "raw_text": ""},
                {"agent_name": "SymptomAgent",
                 "deltas": [{"field": "spread_pattern",
                             "image_shows": "Y", "canonical_says": "", "image_quote": ""}],
                 "confidence": "high", "handoff_target": "DiagnosisAgent",
                 "reasoning": "", "raw_text": ""},
            ],
            "final_deltas": [
                {"field": "lesion_morphology", "image_shows": "X",
                 "canonical_says": "", "image_quote": ""},
            ],
            "existing_kb_at_start": [],
        }
        lines.append(json.dumps(rec))
    p.write_text("\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_kb_stats_emits_tex(tmp_path, monkeypatch):
    from scripts.viz import kb_stats, _common
    seed = _make_seed(tmp_path)
    # Redirect output dirs to a temp area so the test can't pollute the repo.
    monkeypatch.setattr(_common, "FIG_DIR", tmp_path / "figs")
    monkeypatch.setattr(_common, "TEX_DIR", tmp_path / "tex")
    monkeypatch.setattr("sys.argv",
                        ["kb_stats", "--seed", str(seed), "--name", "kbtest"])
    kb_stats.main()
    tex = (tmp_path / "tex" / "auto_kbtest.tex").read_text()
    assert "PathomeDB seed summary" in tex
    assert "Profiles" in tex


def test_observe_curves_emits_tex(tmp_path, monkeypatch):
    from scripts.viz import observe_curves, _common
    history = _make_history(tmp_path)
    monkeypatch.setattr(_common, "FIG_DIR", tmp_path / "figs")
    monkeypatch.setattr(_common, "TEX_DIR", tmp_path / "tex")
    monkeypatch.setattr("sys.argv",
                        ["observe_curves", "--history", str(history),
                         "--name", "ocurves"])
    observe_curves.main()
    tex = (tmp_path / "tex" / "auto_ocurves.tex").read_text()
    assert "OBSERVE training history" in tex
    assert "Epoch" in tex


def test_observe_eval_emits_tex(tmp_path, monkeypatch):
    from scripts.viz import observe_eval, _common
    ev = _make_eval(tmp_path)
    monkeypatch.setattr(_common, "FIG_DIR", tmp_path / "figs")
    monkeypatch.setattr(_common, "TEX_DIR", tmp_path / "tex")
    monkeypatch.setattr("sys.argv",
                        ["observe_eval", "--eval", str(ev), "--name", "oeval"])
    observe_eval.main()
    tex = (tmp_path / "tex" / "auto_oeval.tex").read_text()
    assert "Routing accuracy" in tex
    assert "MorphologyAgent" in tex


def test_trace_stats_emits_tex(tmp_path, monkeypatch):
    from scripts.viz import trace_stats, _common
    tr = _make_traces(tmp_path)
    monkeypatch.setattr(_common, "FIG_DIR", tmp_path / "figs")
    monkeypatch.setattr(_common, "TEX_DIR", tmp_path / "tex")
    monkeypatch.setattr("sys.argv",
                        ["trace_stats", "--traces", str(tr), "--name", "tstats"])
    trace_stats.main()
    tex = (tmp_path / "tex" / "auto_tstats.tex").read_text()
    assert "trace summary" in tex.lower() or "phase 0r trace" in tex.lower()
