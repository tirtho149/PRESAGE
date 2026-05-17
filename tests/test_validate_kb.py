"""
tests/test_validate_kb.py
=========================
The step-3 driver must (a) never silently drop candidates when the
verifier fails, and (b) refuse to report success on a degraded run
unless ALLOW_UNVERIFIED=1. No network, no GPU.
"""

from __future__ import annotations

import json

import pytest

from scripts import validate_kb as V


# ---- _apply_verifier_result --------------------------------------------

def _state_block():
    return {"deltas": [
        {"field": "severity", "image_shows": "collapse",
         "verification_status": "unverified"},
        {"field": "look_alikes", "image_shows": "old",
         "verification_status": "verified"},   # pre-existing, must survive
    ]}


def test_apply_failed_preserves_candidates_not_dropped():
    sb = _state_block()
    out = V._apply_verifier_result(sb, {
        "_verifier_failed": True,
        "_failure_reason": "claude_query returned None",
        "_preserved_unverified": [
            {"field": "severity", "image_shows": "collapse",
             "verification_status": "unverified"}],
    })
    assert out == "failed"
    statuses = sorted(d["verification_status"] for d in sb["deltas"])
    # existing 'verified' kept + candidate preserved as 'unverified'
    assert statuses == ["unverified", "verified"]
    assert sb["__verifier_meta__"]["failed"] is True


def test_apply_ok_persists_dropped_contradictory_bodies():
    sb = _state_block()
    out = V._apply_verifier_result(sb, {
        "accepted": [{"field": "severity", "image_shows": "collapse",
                      "verification_status": "verified"}],
        "verified": [{"x": 1}],
        "provisional": [],
        "contradictory": [{"field": "look_alikes",
                           "image_shows": "actually blight",
                           "verification_status": "contradictory"}],
        "duplicates_of_existing": [],
    })
    assert out == "ok"
    meta = sb["__verifier_meta__"]
    assert meta["failed"] is False
    assert meta["contradictory_count"] == 1
    # bodies (not just count) persisted -> true rejection rate recoverable
    assert meta["dropped_contradictory"][0]["image_shows"] == "actually blight"


# ---- main(): degraded run fails loud -----------------------------------

def _kb(tmp_path):
    d = tmp_path / "Soybean"
    d.mkdir(parents=True)
    (d / "final_registry.json").write_text(json.dumps({
        "crop": "Soybean",
        "diseases": [{
            "disease_name": "Charcoal Rot",
            "visual_symptoms": {"summary": "s"},
            "regional_observations": {"Alabama": {
                "image_ids": ["bugwood::1"],
                "deltas": [{"field": "severity", "image_shows": "collapse",
                            "verification_status": "unverified"}],
            }},
        }],
    }))
    return d / "final_registry.json"


def _failing_verifier(**kw):
    return {
        "verified": [], "provisional": [], "contradictory": [],
        "duplicates_of_existing": [], "accepted": [],
        "_verifier_failed": True, "_failure_reason": "claude down",
        "_preserved_unverified": [
            {"field": "severity", "image_shows": "collapse",
             "verification_status": "unverified"}],
    }


def test_main_exits_3_on_degraded_run(tmp_path, monkeypatch):
    reg = _kb(tmp_path)
    monkeypatch.setattr(V, "verify_candidates", _failing_verifier)
    monkeypatch.setattr(V.sys, "argv",
                        ["validate_kb.py", "--kb-root", str(tmp_path.resolve())])
    monkeypatch.delenv("ALLOW_UNVERIFIED", raising=False)
    monkeypatch.delenv("CROPS", raising=False)
    monkeypatch.delenv("MAX_TUPLES", raising=False)
    monkeypatch.delenv("DRY_RUN", raising=False)

    with pytest.raises(SystemExit) as ei:
        V.main()
    assert ei.value.code == 3

    # Deltas were preserved as 'unverified' (NOT dropped) in the file.
    data = json.loads(reg.read_text())
    deltas = data["diseases"][0]["regional_observations"]["Alabama"]["deltas"]
    assert len(deltas) == 1
    assert deltas[0]["verification_status"] == "unverified"


def test_main_allow_unverified_does_not_exit(tmp_path, monkeypatch):
    _kb(tmp_path)
    monkeypatch.setattr(V, "verify_candidates", _failing_verifier)
    monkeypatch.setattr(V.sys, "argv",
                        ["validate_kb.py", "--kb-root", str(tmp_path.resolve())])
    monkeypatch.setenv("ALLOW_UNVERIFIED", "1")
    monkeypatch.delenv("CROPS", raising=False)
    monkeypatch.delenv("MAX_TUPLES", raising=False)
    monkeypatch.delenv("DRY_RUN", raising=False)

    V.main()  # must NOT raise SystemExit


def test_main_clean_run_succeeds(tmp_path, monkeypatch):
    _kb(tmp_path)

    def _ok_verifier(**kw):
        return {
            "verified": [{"field": "severity", "image_shows": "collapse",
                          "verification_status": "verified"}],
            "provisional": [], "contradictory": [],
            "duplicates_of_existing": [],
            "accepted": [{"field": "severity", "image_shows": "collapse",
                          "verification_status": "verified"}],
        }

    monkeypatch.setattr(V, "verify_candidates", _ok_verifier)
    monkeypatch.setattr(V.sys, "argv",
                        ["validate_kb.py", "--kb-root", str(tmp_path.resolve())])
    monkeypatch.delenv("ALLOW_UNVERIFIED", raising=False)
    monkeypatch.delenv("CROPS", raising=False)
    monkeypatch.delenv("MAX_TUPLES", raising=False)
    monkeypatch.delenv("DRY_RUN", raising=False)

    V.main()  # no SystemExit; clean verified run
