"""Tests for ProfileRegistry domain feedback (record_domain_feedback / get_domain_corrections)."""

import json
import time

import pytest
from pathlib import Path

from agent_framework.core.profile_registry import ProfileRegistry


@pytest.fixture
def registry(tmp_path):
    """Create a registry with a writable workspace."""
    return ProfileRegistry(tmp_path)


class TestRecordDomainFeedback:
    """Tests for recording domain mismatch signals."""

    def test_stores_mismatch_signal(self, registry):
        registry.record_domain_feedback(
            profile_id="backend-eng",
            domain_tags=["database", "infrastructure"],
            mismatch_signal=True,
        )

        corrections = registry.get_domain_corrections("backend-eng")
        assert corrections["total_signals"] == 1
        assert len(corrections["mismatches"]) == 1
        assert corrections["mismatches"][0]["domain_tags"] == ["database", "infrastructure"]
        assert "timestamp" in corrections["mismatches"][0]

    def test_stores_positive_signal_without_mismatch(self, registry):
        """A non-mismatch signal increments total_signals but adds no mismatch entry."""
        registry.record_domain_feedback(
            profile_id="backend-eng",
            domain_tags=["backend"],
            mismatch_signal=False,
        )

        corrections = registry.get_domain_corrections("backend-eng")
        assert corrections["total_signals"] == 1
        assert len(corrections.get("mismatches", [])) == 0

    def test_accumulates_multiple_signals(self, registry):
        registry.record_domain_feedback("prof-a", ["database"], True)
        registry.record_domain_feedback("prof-a", ["frontend"], True)
        registry.record_domain_feedback("prof-a", ["backend"], False)

        corrections = registry.get_domain_corrections("prof-a")
        assert corrections["total_signals"] == 3
        assert len(corrections["mismatches"]) == 2

    def test_independent_profiles(self, registry):
        """Feedback for different profiles doesn't cross-contaminate."""
        registry.record_domain_feedback("prof-a", ["database"], True)
        registry.record_domain_feedback("prof-b", ["frontend"], True)

        a = registry.get_domain_corrections("prof-a")
        b = registry.get_domain_corrections("prof-b")
        assert a["mismatches"][0]["domain_tags"] == ["database"]
        assert b["mismatches"][0]["domain_tags"] == ["frontend"]

    def test_caps_mismatches_at_50(self, registry):
        """Prevents unbounded growth of mismatch entries per profile."""
        for i in range(55):
            registry.record_domain_feedback("prof-a", [f"domain-{i}"], True)

        corrections = registry.get_domain_corrections("prof-a")
        assert len(corrections["mismatches"]) <= 50
        assert corrections["total_signals"] == 55

    def test_persists_to_disk(self, registry):
        registry.record_domain_feedback("prof-a", ["security"], True)

        # Create a new registry pointing at the same workspace
        registry2 = ProfileRegistry(registry._store_path.parent.parent.parent)
        corrections = registry2.get_domain_corrections("prof-a")
        assert corrections["total_signals"] == 1
        assert corrections["mismatches"][0]["domain_tags"] == ["security"]


class TestGetDomainCorrections:
    """Tests for retrieving domain feedback."""

    def test_returns_empty_dict_for_unknown_profile(self, registry):
        assert registry.get_domain_corrections("nonexistent") == {}

    def test_returns_accumulated_data(self, registry):
        registry.record_domain_feedback("prof-x", ["database"], True)
        registry.record_domain_feedback("prof-x", ["backend"], False)

        result = registry.get_domain_corrections("prof-x")
        assert result["total_signals"] == 2
        assert len(result["mismatches"]) == 1

    def test_handles_corrupt_feedback_file(self, registry):
        """Gracefully returns empty on corrupt JSON."""
        feedback_path = registry._store_path.parent / "domain_feedback.json"
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        feedback_path.write_text("{{{NOT JSON")

        assert registry.get_domain_corrections("anything") == {}

    def test_handles_missing_feedback_file(self, registry):
        """Returns empty when no feedback file exists yet."""
        assert registry.get_domain_corrections("anything") == {}
