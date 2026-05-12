"""
tests/test_verifier.py
======================
Unit tests for the Claude-headless verifier. claude_query is monkey-
patched to return canned JSON so the tests run offline (no API spend,
no claude CLI required).
"""

from __future__ import annotations

import json

import pytest


CANON = {
    "pathogen_scientific_name": "Macrophomina phaseolina",
    "type_of_disease":          "Fungal",
    "affected_parts":           ["Foliar", "Stem"],
    "summary":                  "Soilborne fungus; charcoal-sprinkled stem",
    "diagnostic_features":      ["microsclerotia in pith"],
    "look_alikes":              [],
    "treatments":               ["drought mitigation"],
}


def _make_candidate(field: str, image_shows: str, support: int = 3) -> dict:
    return {
        "field":          field,
        "canonical_says": "(not specified)",
        "image_shows":    image_shows,
        "image_quote":    "visible field evidence",
        "image_id":       "bugwood::1568038",
        "__support__":    support,
        "__cluster_size__": support + 1,
    }


def test_verifier_offline_passthrough(monkeypatch):
    """No Claude available -> candidates pass through as 'unverified'."""
    from pathome_kb import verifier

    monkeypatch.setattr(verifier, "_claude_available", lambda: False)
    candidates = [
        _make_candidate("lesion_morphology", "yellow halos around dark spots"),
        _make_candidate("severity", "whole-field collapse"),
    ]
    verdict = verifier.verify_candidates(
        crop="Soybean", disease="Charcoal Rot", state="Alabama",
        canonical=CANON,
        existing_kb_deltas=[],
        candidates=candidates,
        primary_image_id="bugwood::1568038",
    )
    assert verdict["verified"] == []
    assert verdict["contradictory"] == []
    # All passthrough land in 'provisional' + 'accepted'.
    assert len(verdict["provisional"]) == 2
    assert len(verdict["accepted"]) == 2
    for d in verdict["accepted"]:
        assert d["verification_status"] == "unverified"
        assert d["web_support"] == []
        assert d["swarm_support"] >= 1


def test_verifier_parses_claude_verdict(monkeypatch):
    """Claude returns a well-formed JSON verdict -> deltas bucketed correctly."""
    from pathome_kb import verifier

    monkeypatch.setattr(verifier, "_claude_available", lambda: True)

    # Canned Claude response: one verified, one provisional, one contradicted,
    # one duplicate.
    canned = {
        "verified": [{
            "field":          "lesion_morphology",
            "canonical_says": "(not specified)",
            "image_shows":    "yellow chlorotic halos around dark necrotic centers",
            "image_quote":    "visible halo pattern",
            "image_id":       "bugwood::1568038",
            "swarm_support":  4,
            "verification_status": "verified",
            "web_support": [
                {"url": "https://extension.umn.edu/.../charcoal-rot",
                 "quote": "Lesions develop yellow halos."},
            ],
            "reasoning": "Umn extension confirms halo pattern.",
        }],
        "provisional": [{
            "field":          "severity",
            "canonical_says": "(not specified)",
            "image_shows":    "whole-field collapse in Alabama planting",
            "image_quote":    "rows defoliated",
            "image_id":       "bugwood::1568038",
            "swarm_support":  3,
            "verification_status": "novel_plausible",
            "web_support":    [],
            "reasoning":      "No source describes whole-field collapse.",
        }],
        "contradictory": [{
            "field":          "look_alikes",
            "canonical_says": "(not specified)",
            "image_shows":    "almost certainly bacterial blight",
            "image_quote":    "...",
            "image_id":       "bugwood::1568038",
            "swarm_support":  1,
            "verification_status": "contradictory",
            "web_support": [
                {"url": "https://aps.org/.../blight",
                 "quote": "Charcoal rot lacks water-soaked halos."},
            ],
            "reasoning": "APS notes charcoal rot lacks water-soaking.",
        }],
        "duplicates_of_existing": [{
            "field":          "diagnostic_features",
            "canonical_says": "microsclerotia in pith",
            "image_shows":    "marbled cross-sections with microsclerotia",
            "image_quote":    "...",
            "image_id":       "bugwood::1568038",
            "swarm_support":  2,
            "verification_status": "duplicate_existing",
            "web_support":    [],
            "reasoning":      "restates canonical diagnostic_features.",
        }],
    }
    monkeypatch.setattr(verifier, "claude_query", lambda **kw: json.dumps(canned))

    candidates = [_make_candidate("lesion_morphology", "yellow halos", 4)]
    verdict = verifier.verify_candidates(
        crop="Soybean", disease="Charcoal Rot", state="Alabama",
        canonical=CANON,
        existing_kb_deltas=[],
        candidates=candidates,
        primary_image_id="bugwood::1568038",
    )
    assert len(verdict["verified"])              == 1
    assert len(verdict["provisional"])           == 1
    assert len(verdict["contradictory"])         == 1
    assert len(verdict["duplicates_of_existing"]) == 1
    # accepted = verified + provisional (contradictory dropped, duplicates dropped)
    assert len(verdict["accepted"]) == 2

    v = verdict["verified"][0]
    assert v["verification_status"] == "verified"
    assert v["swarm_support"] == 4
    assert v["web_support"][0]["url"].startswith("https://")
    assert v["reasoning"].startswith("Umn")

    # Contradictory delta still has the citation that contradicted it (audit trail).
    c = verdict["contradictory"][0]
    assert c["verification_status"] == "contradictory"
    assert c["web_support"][0]["url"].startswith("https://")


def test_verifier_drops_empty_image_shows(monkeypatch):
    """Verifier output with empty image_shows is dropped at normalization."""
    from pathome_kb import verifier

    monkeypatch.setattr(verifier, "_claude_available", lambda: True)
    canned = {
        "verified":               [{"field": "severity", "image_shows": "",
                                     "verification_status": "verified"}],
        "provisional":            [],
        "contradictory":          [],
        "duplicates_of_existing": [],
    }
    monkeypatch.setattr(verifier, "claude_query", lambda **kw: json.dumps(canned))

    verdict = verifier.verify_candidates(
        crop="X", disease="Y", state="Z",
        canonical=CANON, existing_kb_deltas=[],
        candidates=[_make_candidate("severity", "X")],
    )
    assert verdict["verified"] == []


def test_verifier_empty_candidates_short_circuits(monkeypatch):
    """No candidates -> all buckets empty, no Claude call."""
    from pathome_kb import verifier

    called = []
    def _no_call(**kw):
        called.append(True)
        return None
    monkeypatch.setattr(verifier, "claude_query", _no_call)

    verdict = verifier.verify_candidates(
        crop="X", disease="Y", state="Z",
        canonical=CANON, existing_kb_deltas=[], candidates=[],
    )
    assert called == []
    assert all(verdict[k] == [] for k in
               ("verified", "provisional", "contradictory",
                "duplicates_of_existing", "accepted"))


def test_merge_with_existing_upgrades_verification_status():
    """When a verified delta overlaps an unverified existing one, the
    existing's status is upgraded and the citations merged."""
    from plantswarm.delta_pipeline import _merge_with_existing

    existing = [{
        "field": "lesion_morphology",
        "image_shows": "raised pustular lesions with halos",
        "canonical_says": "", "image_quote": "",
        "swarm_support": 3,
        "verification_status": "unverified",
        "web_support": [],
    }]
    new = [{
        "field": "lesion_morphology",
        "image_shows": "yellow halos surround raised pustular lesions",
        "canonical_says": "", "image_quote": "",
        "swarm_support": 4,
        "verification_status": "verified",
        "web_support": [{"url": "https://example.com/a", "quote": "..."}],
    }]
    merged, counts = _merge_with_existing(
        existing=existing, new=new, similarity_threshold=0.3,
    )
    assert counts["n_overlaps_bumped"] == 1
    assert counts["n_upgraded"] == 1
    # Existing entry got upgraded to verified and absorbed the citation.
    only = merged[0]
    assert only["verification_status"] == "verified"
    assert only["swarm_support"] == 7
    assert any(s["url"] == "https://example.com/a" for s in only["web_support"])
