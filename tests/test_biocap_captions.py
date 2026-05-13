"""
tests/test_biocap_captions.py
=============================
Unit tests for ``plantswarm.captioning``.

Covers:
  - Strategy registry shape
  - taxon_text + healthy template
  - Per-strategy caption shape on a synthetic disease profile
  - State-aware delta selection
  - Hard-fail when a delta strategy meets a profile with no
    regional_observations
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

def test_strategies_registry_is_stable():
    from plantswarm.captioning import STRATEGIES, DELTA_STRATEGIES
    assert STRATEGIES == (
        "label_only",
        "summary_only",
        "canonical_full",
        "canonical_deltas_1",
        "canonical_deltas_3",
        "canonical_deltas_5",
        "canonical_deltas_7",
    )
    assert DELTA_STRATEGIES == (
        "canonical_deltas_1",
        "canonical_deltas_3",
        "canonical_deltas_5",
        "canonical_deltas_7",
    )


def test_taxon_text_format():
    from plantswarm.captioning import taxon_text
    assert taxon_text("Tomato", "Early Blight") == "Tomato Early Blight"
    assert taxon_text(" Tomato ", " Late Blight ") == "Tomato   Late Blight"


def test_healthy_template_mentions_crop():
    from plantswarm.captioning import build_healthy_caption
    out = build_healthy_caption("Tomato")
    assert "Tomato" in out
    assert "no visible disease" in out.lower()


# ---------------------------------------------------------------------------
# Synthetic disease record (no need for real KB on disk)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_profile_no_deltas():
    return {
        "disease_name": "Early Blight",
        "pathogen_scientific_name": {"value": "Alternaria linariae"},
        "type_of_disease":          {"value": "Fungal"},
        "affected_parts":           {"value": ["leaves", "stems"]},
        "visual_symptoms": {
            "summary":             {"value": "Dark brown concentric lesions on older foliage."},
            "diagnostic_features": {"value": "Target-spot pattern with yellow halo."},
            "look_alikes":         {"value": ["Septoria Leaf Spot"]},
        },
        "regional_observations": {},   # empty -> delta strategies must fail
    }


@pytest.fixture
def synthetic_profile_with_deltas(synthetic_profile_no_deltas):
    p = dict(synthetic_profile_no_deltas)
    p["regional_observations"] = {
        "TX": {
            "image_ids": ["bugwood::1"],
            "deltas": [{
                "field":               "lesion_morphology",
                "canonical_says":      "concentric rings",
                "image_shows":         "fewer rings, larger lesions",
                "image_quote":         "irregular margins",
                "verification_status": "verified",
                "swarm_support":       3,
            }],
        },
        "CA": {
            "image_ids": ["bugwood::2"],
            "deltas": [{
                "field":               "color",
                "canonical_says":      "dark brown",
                "image_shows":         "purple-tinged margins",
                "verification_status": "weakly_supported",
                "swarm_support":       2,
            }],
        },
    }
    return p


def test_label_only_returns_just_taxon(synthetic_profile_no_deltas):
    from plantswarm.captioning import build_disease_caption
    out = build_disease_caption(
        crop="Tomato", disease="Early Blight",
        disease_record=synthetic_profile_no_deltas,
        strategy="label_only",
    )
    assert out == "Tomato Early Blight"


def test_summary_only_includes_summary_but_not_diagnostic(synthetic_profile_no_deltas):
    from plantswarm.captioning import build_disease_caption
    out = build_disease_caption(
        crop="Tomato", disease="Early Blight",
        disease_record=synthetic_profile_no_deltas,
        strategy="summary_only",
    )
    assert "concentric lesions" in out
    assert "Diagnostic features" not in out
    assert "May be confused" not in out


def test_canonical_full_includes_diagnostic_and_lookalikes(synthetic_profile_no_deltas):
    from plantswarm.captioning import build_disease_caption
    out = build_disease_caption(
        crop="Tomato", disease="Early Blight",
        disease_record=synthetic_profile_no_deltas,
        strategy="canonical_full",
    )
    assert "Diagnostic features" in out
    assert "May be confused" in out
    assert "Septoria Leaf Spot" in out
    assert "leaves" in out


def test_delta_strategy_fails_on_empty_regional(synthetic_profile_no_deltas):
    from plantswarm.captioning import build_disease_caption
    with pytest.raises(ValueError, match="requires deltas"):
        build_disease_caption(
            crop="Tomato", disease="Early Blight",
            disease_record=synthetic_profile_no_deltas,
            strategy="canonical_deltas_3",
        )


def test_delta_strategy_emits_state_phrase(synthetic_profile_with_deltas):
    from plantswarm.captioning import build_disease_caption
    out = build_disease_caption(
        crop="Tomato", disease="Early Blight",
        disease_record=synthetic_profile_with_deltas,
        strategy="canonical_deltas_3",
    )
    assert "Regional variations:" in out
    # Verified delta should rank above weakly-supported.
    assert "in TX, fewer rings" in out


def test_delta_strategy_state_filter(synthetic_profile_with_deltas):
    """When a state is given, only that state's deltas should appear,
    falling back to cross-state if the chosen state has none."""
    from plantswarm.captioning import build_disease_caption
    out_ca = build_disease_caption(
        crop="Tomato", disease="Early Blight",
        disease_record=synthetic_profile_with_deltas,
        strategy="canonical_deltas_3",
        state="CA",
    )
    assert "in CA," in out_ca
    assert "in TX," not in out_ca


def test_delta_strategy_state_filter_falls_back(synthetic_profile_with_deltas):
    """If the chosen state has no deltas, fall back to cross-state."""
    from plantswarm.captioning import build_disease_caption
    out = build_disease_caption(
        crop="Tomato", disease="Early Blight",
        disease_record=synthetic_profile_with_deltas,
        strategy="canonical_deltas_3",
        state="NY",   # no deltas for NY
    )
    # Should still produce a Regional variations sentence pulling cross-state.
    assert "Regional variations:" in out


def test_caption_for_row_healthy_path(synthetic_profile_no_deltas):
    """healthy diseases bypass the KB and use the static template."""
    from plantswarm.captioning import caption_for_row
    profiles = {("Tomato", "Early Blight"): synthetic_profile_no_deltas}
    out, used_kb = caption_for_row(
        crop="Tomato", disease="healthy", state=None,
        profiles=profiles, strategy="canonical_deltas_3",
    )
    assert "healthy Tomato" in out
    assert "no visible disease" in out.lower()
    assert used_kb is False


def test_caption_for_row_fallback_when_no_kb_profile(synthetic_profile_no_deltas):
    """Unknown (crop, disease) pairs get the fallback template, NOT a KeyError.
    This is what lets full Bugwood (459 non-KB pairs) join training."""
    from plantswarm.captioning import caption_for_row
    profiles = {("Tomato", "Early Blight"): synthetic_profile_no_deltas}
    out, used_kb = caption_for_row(
        crop="Apple", disease="Cedar Apple Rust", state=None,
        profiles=profiles, strategy="canonical_full",
    )
    assert out == "A field photograph of Apple affected by Cedar Apple Rust."
    assert used_kb is False


def test_fallback_label_only_returns_taxon_text(synthetic_profile_no_deltas):
    from plantswarm.captioning import caption_for_row
    profiles = {("Tomato", "Early Blight"): synthetic_profile_no_deltas}
    out, used_kb = caption_for_row(
        crop="Apple", disease="Scab", state=None,
        profiles=profiles, strategy="label_only",
    )
    assert out == "Apple Scab"
    assert used_kb is False


def test_fallback_used_kb_flag_true_when_profile_matched(synthetic_profile_no_deltas):
    from plantswarm.captioning import caption_for_row
    profiles = {("Tomato", "Early Blight"): synthetic_profile_no_deltas}
    out, used_kb = caption_for_row(
        crop="Tomato", disease="Early Blight", state=None,
        profiles=profiles, strategy="canonical_full",
    )
    assert "Diagnostic features" in out
    assert used_kb is True


def test_fallback_does_not_require_deltas(synthetic_profile_no_deltas):
    """A delta strategy on a non-KB pair must NOT raise — fallback caption
    has no delta dependency."""
    from plantswarm.captioning import caption_for_row
    profiles = {("Tomato", "Early Blight"): synthetic_profile_no_deltas}
    out, used_kb = caption_for_row(
        crop="Apple", disease="Scab", state="CA",
        profiles=profiles, strategy="canonical_deltas_3",
    )
    assert out == "A field photograph of Apple affected by Scab."
    assert used_kb is False


# ---------------------------------------------------------------------------
# assert_deltas_populated guard
# ---------------------------------------------------------------------------

def test_guard_no_op_for_non_delta_strategy(synthetic_profile_no_deltas):
    """Strategies that don't need deltas should never trigger the guard."""
    from plantswarm.captioning import assert_deltas_populated
    profs = {("Tomato", "Early Blight"): synthetic_profile_no_deltas}
    # Should not raise.
    assert_deltas_populated(profs, strategies=["label_only", "canonical_full"])


def test_guard_raises_for_delta_strategy(synthetic_profile_no_deltas):
    from plantswarm.captioning import assert_deltas_populated
    profs = {("Tomato", "Early Blight"): synthetic_profile_no_deltas}
    with pytest.raises(RuntimeError, match="Phase 0R has not populated"):
        assert_deltas_populated(profs, strategies=["canonical_deltas_3"])


def test_guard_passes_when_at_least_one_delta(synthetic_profile_with_deltas):
    from plantswarm.captioning import assert_deltas_populated
    profs = {("Tomato", "Early Blight"): synthetic_profile_with_deltas}
    # Should not raise.
    assert_deltas_populated(profs, strategies=["canonical_deltas_5"])
