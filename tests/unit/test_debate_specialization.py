"""Tests for debate-driven specialization adjustment."""

from unittest.mock import MagicMock

import pytest

from agent_framework.core.engineer_specialization import (
    SpecializationProfile,
    adjust_specialization_from_debate,
)
from agent_framework.memory.memory_store import MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(workspace=tmp_path, enabled=True)


REPO = "org/repo"


def _make_profile(profile_id: str) -> SpecializationProfile:
    return SpecializationProfile(
        id=profile_id,
        name=f"{profile_id.title()} Engineer",
        description=f"Specializes in {profile_id}",
        file_patterns=[],
        priority_tools=[],
        prompt_additions="",
    )


class TestAdjustFromDebate:
    def test_no_adjustment_when_no_debate_memories(self, store):
        profile = _make_profile("backend")
        result = adjust_specialization_from_debate(profile, store, REPO)
        assert result is profile

    def test_no_adjustment_when_memory_disabled(self, tmp_path):
        disabled = MemoryStore(workspace=tmp_path, enabled=False)
        profile = _make_profile("backend")
        result = adjust_specialization_from_debate(profile, disabled, REPO)
        assert result is profile

    def test_no_adjustment_when_same_domain(self, store):
        """Debate recommending backend shouldn't change an existing backend profile."""
        store.remember(
            repo_slug=REPO,
            agent_type="shared",
            category="architectural_decisions",
            content="Debate concluded: use backend approach for API endpoint design",
            tags=["debate"],
        )
        profile = _make_profile("backend")
        result = adjust_specialization_from_debate(profile, store, REPO)
        assert result.id == "backend"

    def test_adjusts_to_frontend_when_debate_recommends(self, store):
        """Debate recommending frontend should override backend profile."""
        store.remember(
            repo_slug=REPO,
            agent_type="shared",
            category="architectural_decisions",
            content="Debate concluded: use frontend approach — this is a React component task",
            tags=["debate"],
        )
        profile = _make_profile("backend")
        result = adjust_specialization_from_debate(profile, store, REPO)
        assert result.id == "frontend"

    def test_adjusts_from_none_profile(self, store):
        """Should handle None initial profile (auto-detect failed)."""
        store.remember(
            repo_slug=REPO,
            agent_type="shared",
            category="architectural_decisions",
            content="Debate concluded: recommend frontend approach for this React UI task",
            tags=["debate"],
        )
        result = adjust_specialization_from_debate(None, store, REPO)
        assert result is not None
        assert result.id == "frontend"

    def test_logs_adjustment_to_session_logger(self, store):
        store.remember(
            repo_slug=REPO,
            agent_type="shared",
            category="architectural_decisions",
            content="Debate concluded: use frontend approach for UI component",
            tags=["debate"],
        )
        session_logger = MagicMock()
        profile = _make_profile("backend")

        adjust_specialization_from_debate(profile, store, REPO, session_logger=session_logger)
        session_logger.log.assert_called_once()
        call_args = session_logger.log.call_args
        assert call_args[0][0] == "specialization_adjusted"

    def test_ignores_non_debate_tagged_memories(self, store):
        """Memories without 'debate' tag shouldn't trigger adjustment."""
        store.remember(
            repo_slug=REPO,
            agent_type="shared",
            category="architectural_decisions",
            content="Use frontend approach for everything",
            tags=["convention"],  # Not a debate tag
        )
        profile = _make_profile("backend")
        result = adjust_specialization_from_debate(profile, store, REPO)
        assert result.id == "backend"
