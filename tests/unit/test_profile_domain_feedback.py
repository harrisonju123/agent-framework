"""Tests for ProfileRegistry domain feedback (cross-feature learning)."""

import json
from pathlib import Path

import pytest

from agent_framework.core.profile_registry import ProfileRegistry


@pytest.fixture
def registry(tmp_path):
    return ProfileRegistry(workspace=tmp_path)


class TestRecordDomainFeedback:
    def test_records_single_mismatch(self, registry):
        registry.record_domain_feedback(
            task_id="task-1",
            detected_domain="frontend",
            original_profile_id="backend",
        )

        path = registry._domain_feedback_path()
        assert path.exists()

        data = json.loads(path.read_text())
        key = "backend->frontend"
        assert key in data
        assert data[key]["count"] == 1
        assert "task-1" in data[key]["task_ids"]

    def test_increments_count_on_repeated_mismatch(self, registry):
        for i in range(5):
            registry.record_domain_feedback(
                task_id=f"task-{i}",
                detected_domain="frontend",
                original_profile_id="backend",
            )

        data = json.loads(registry._domain_feedback_path().read_text())
        assert data["backend->frontend"]["count"] == 5
        assert len(data["backend->frontend"]["task_ids"]) == 5

    def test_caps_task_ids_at_10(self, registry):
        for i in range(15):
            registry.record_domain_feedback(
                task_id=f"task-{i}",
                detected_domain="frontend",
                original_profile_id="backend",
            )

        data = json.loads(registry._domain_feedback_path().read_text())
        assert len(data["backend->frontend"]["task_ids"]) == 10
        # Should keep the most recent
        assert "task-14" in data["backend->frontend"]["task_ids"]

    def test_tracks_multiple_mismatch_directions(self, registry):
        registry.record_domain_feedback("t1", "frontend", "backend")
        registry.record_domain_feedback("t2", "infrastructure", "backend")

        data = json.loads(registry._domain_feedback_path().read_text())
        assert "backend->frontend" in data
        assert "backend->infrastructure" in data


class TestGetDomainCorrections:
    def test_no_corrections_below_threshold(self, registry):
        # Only 2 signals, threshold is 3
        registry.record_domain_feedback("t1", "frontend", "backend")
        registry.record_domain_feedback("t2", "frontend", "backend")

        corrections = registry.get_domain_corrections()
        assert corrections == {}

    def test_corrections_at_threshold(self, registry):
        for i in range(3):
            registry.record_domain_feedback(f"t{i}", "frontend", "backend")

        corrections = registry.get_domain_corrections()
        assert "backend" in corrections
        assert corrections["backend"] < 0  # penalized
        assert "frontend" in corrections
        assert corrections["frontend"] > 0  # boosted

    def test_corrections_capped_at_max(self, registry):
        # 100 signals should still cap at 0.1
        for i in range(100):
            registry.record_domain_feedback(f"t{i}", "frontend", "backend")

        corrections = registry.get_domain_corrections()
        assert corrections["backend"] >= -0.1
        assert corrections["frontend"] <= 0.1

    def test_empty_corrections_with_no_feedback(self, registry):
        corrections = registry.get_domain_corrections()
        assert corrections == {}

    def test_multiple_directions_combined(self, registry):
        # Backend mismatched to frontend 3 times
        for i in range(3):
            registry.record_domain_feedback(f"f{i}", "frontend", "backend")

        # Backend mismatched to infrastructure 3 times
        for i in range(3):
            registry.record_domain_feedback(f"i{i}", "infrastructure", "backend")

        corrections = registry.get_domain_corrections()
        # Backend should have double penalty (from both mismatch directions)
        assert corrections["backend"] == -0.1  # clamped to max
        assert corrections.get("frontend", 0) > 0
        assert corrections.get("infrastructure", 0) > 0
