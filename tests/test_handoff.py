"""
tests/test_handoff.py
=====================
Unit tests for scripts/check_handoff.py — the step-handoff guard.
Builds tiny synthetic registries in tmp_path; no network, no GPU.
"""

from __future__ import annotations

import json

from scripts import check_handoff as H


def _write_registry(kb_root, crop, diseases):
    d = kb_root / crop
    d.mkdir(parents=True, exist_ok=True)
    (d / "final_registry.json").write_text(json.dumps({"diseases": diseases}))


def _disease(name="Charcoal Rot", summary="dark microsclerotia", deltas=None):
    return {
        "disease_name": name,
        "visual_symptoms": {"summary": summary},
        "regional_observations": {
            "Alabama": {"deltas": deltas or []}
        },
    }


def _delta(status):
    return {"field": "severity", "image_shows": "x",
            "verification_status": status}


# ---- canonical-kb -------------------------------------------------------

def test_canonical_ok(tmp_path):
    _write_registry(tmp_path, "Soybean", [_disease()])
    ok, msg = H.check_canonical_kb(tmp_path, ["Soybean"])
    assert ok, msg


def test_canonical_missing_crop(tmp_path):
    _write_registry(tmp_path, "Soybean", [_disease()])
    ok, msg = H.check_canonical_kb(tmp_path, ["Soybean", "Tomato"])
    assert not ok
    assert "Tomato" in msg and "Step 1" in msg


def test_canonical_stub_summary_fails(tmp_path):
    _write_registry(tmp_path, "Soybean", [_disease(summary="")])
    ok, msg = H.check_canonical_kb(tmp_path, ["Soybean"])
    assert not ok
    assert "visual_symptoms.summary" in msg


def test_canonical_summary_as_cited_value(tmp_path):
    """visual_symptoms.summary may be a {'value': ...} cited field."""
    dis = _disease()
    dis["visual_symptoms"]["summary"] = {"value": "real text", "url": "u"}
    _write_registry(tmp_path, "Soybean", [dis])
    ok, _ = H.check_canonical_kb(tmp_path, ["Soybean"])
    assert ok


# ---- unverified-deltas --------------------------------------------------

def test_unverified_present_ok(tmp_path):
    _write_registry(tmp_path, "Soybean",
                    [_disease(deltas=[_delta("unverified"), _delta("verified")])])
    ok, msg = H.check_unverified_deltas(tmp_path, ["Soybean"])
    assert ok and "1 unverified" in msg


def test_unverified_none_fails(tmp_path):
    _write_registry(tmp_path, "Soybean",
                    [_disease(deltas=[_delta("verified")])])
    ok, msg = H.check_unverified_deltas(tmp_path, ["Soybean"])
    assert not ok and "Step 2" in msg


def test_unverified_empty_status_counts_as_unverified(tmp_path):
    _write_registry(tmp_path, "Soybean",
                    [_disease(deltas=[_delta("")])])
    ok, _ = H.check_unverified_deltas(tmp_path, ["Soybean"])
    assert ok


# ---- verified-kb --------------------------------------------------------

def test_verified_kb_ok(tmp_path):
    _write_registry(tmp_path, "Soybean",
                    [_disease(deltas=[_delta("verified"),
                                      _delta("provisional")])])
    ok, msg = H.check_verified_kb(tmp_path, ["Soybean"])
    assert ok and "0 unverified" in msg


def test_verified_kb_leftover_fails(tmp_path):
    _write_registry(tmp_path, "Soybean",
                    [_disease(deltas=[_delta("verified"),
                                      _delta("unverified")])])
    ok, msg = H.check_verified_kb(tmp_path, ["Soybean"])
    assert not ok
    assert "1 delta" in msg and "Step 3" in msg


def test_verified_kb_no_registry_fails(tmp_path):
    ok, msg = H.check_verified_kb(tmp_path, ["Soybean"])
    assert not ok


# ---- checkpoint ---------------------------------------------------------

def test_checkpoint_missing(tmp_path):
    ok, msg = H.check_checkpoint(tmp_path / "nope.pt")
    assert not ok and "Step 4" in msg


def test_checkpoint_truncated(tmp_path):
    p = tmp_path / "epoch_50.pt"
    p.write_bytes(b"\x00" * 1024)
    ok, msg = H.check_checkpoint(p)
    assert not ok and "truncated" in msg


def test_checkpoint_ok(tmp_path):
    p = tmp_path / "epoch_50.pt"
    p.write_bytes(b"\x00" * (H._CKPT_MIN_BYTES + 1))
    ok, _ = H.check_checkpoint(p)
    assert ok


# ---- CLI main() return codes -------------------------------------------

def test_main_returns_fail_code(tmp_path):
    _write_registry(tmp_path, "Soybean", [_disease(summary="")])
    rc = H.main(["canonical-kb", "--kb-root", str(tmp_path),
                 "--crops", "Soybean"])
    assert rc == H.FAIL


def test_main_returns_zero_on_ok(tmp_path):
    _write_registry(tmp_path, "Soybean", [_disease()])
    rc = H.main(["canonical-kb", "--kb-root", str(tmp_path),
                 "--crops", "Soybean"])
    assert rc == 0


def test_main_unreadable_registry_is_fail(tmp_path):
    d = tmp_path / "Soybean"
    d.mkdir()
    (d / "final_registry.json").write_text("{ this is not json")
    rc = H.main(["verified-kb", "--kb-root", str(tmp_path),
                 "--crops", "Soybean"])
    assert rc == H.FAIL
