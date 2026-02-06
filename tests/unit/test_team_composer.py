"""Tests for dynamic team composition logic."""

import logging

import pytest

from agent_framework.core.config import AgentDefinition
from agent_framework.core.team_composer import compose_team, WORKFLOW_TEAMMATES, WORKFLOW_RANK


# -- Fixtures --

def _make_agent(agent_id: str, name: str = "", prompt: str = "You are {id}.") -> AgentDefinition:
    return AgentDefinition(
        id=agent_id,
        name=name or agent_id.title(),
        queue=agent_id,
        prompt=prompt.format(id=agent_id),
    )


@pytest.fixture
def agents():
    return [
        _make_agent("engineer", "Software Engineer"),
        _make_agent("qa", "QA Engineer"),
        _make_agent("architect", "Technical Architect"),
        _make_agent("static-analysis", "Static Analysis"),
    ]


# -- Workflow → teammate mapping --

class TestWorkflowMapping:
    def test_simple_returns_none(self, agents):
        result = compose_team({}, "simple", agents)
        assert result is None

    def test_standard_returns_qa(self, agents):
        result = compose_team({}, "standard", agents)
        assert result is not None
        assert list(result.keys()) == ["qa"]

    def test_full_returns_engineer_and_qa(self, agents):
        result = compose_team({}, "full", agents)
        assert result is not None
        assert set(result.keys()) == {"engineer", "qa"}

    def test_quality_focused_returns_three_teammates(self, agents):
        result = compose_team({}, "quality-focused", agents)
        assert result is not None
        assert set(result.keys()) == {"engineer", "qa", "static-analysis"}

    def test_unknown_workflow_returns_none(self, agents):
        result = compose_team({}, "nonexistent", agents)
        assert result is None


# -- min_workflow filtering --

class TestMinWorkflow:
    def test_standard_below_full_min(self, agents):
        result = compose_team({}, "standard", agents, min_workflow="full")
        assert result is None

    def test_full_meets_full_min(self, agents):
        result = compose_team({}, "full", agents, min_workflow="full")
        assert result is not None

    def test_quality_focused_above_full_min(self, agents):
        result = compose_team({}, "quality-focused", agents, min_workflow="full")
        assert result is not None

    def test_simple_min_allows_standard(self, agents):
        result = compose_team({}, "standard", agents, min_workflow="simple")
        assert result is not None

    def test_default_min_is_standard(self, agents):
        """Standard workflow should pass with default min_workflow."""
        result = compose_team({}, "standard", agents)
        assert result is not None


# -- Task-level overrides --

class TestTaskOverride:
    def test_override_false_skips_team(self, agents):
        result = compose_team({"team_override": False}, "full", agents)
        assert result is None

    def test_override_true_bypasses_min_workflow(self, agents):
        result = compose_team(
            {"team_override": True}, "standard", agents, min_workflow="full",
        )
        assert result is not None
        assert "qa" in result

    def test_override_true_still_needs_teammates(self, agents):
        """Override can't conjure teammates for simple workflow (empty list)."""
        result = compose_team({"team_override": True}, "simple", agents)
        assert result is None


# -- Caller exclusion --

class TestCallerExclusion:
    def test_caller_excluded_from_teammates(self, agents):
        """If the lead is also in the teammate list, it should be filtered out."""
        result = compose_team({}, "full", agents, caller_agent_id="engineer")
        assert result is not None
        assert "engineer" not in result
        assert "qa" in result

    def test_caller_not_in_teammates_is_noop(self, agents):
        result = compose_team({}, "full", agents, caller_agent_id="architect")
        assert result is not None
        assert set(result.keys()) == {"engineer", "qa"}

    def test_caller_exclusion_can_empty_team(self, agents):
        """Standard has only qa; if qa is the caller, team becomes empty → None."""
        result = compose_team({}, "standard", agents, caller_agent_id="qa")
        assert result is None


# -- Missing agent config --

class TestMissingConfig:
    def test_missing_teammate_logs_warning_and_skips(self, caplog):
        incomplete_agents = [_make_agent("qa")]
        with caplog.at_level(logging.WARNING):
            result = compose_team({}, "full", incomplete_agents)
        # engineer missing → warning logged, only qa returned
        assert result is not None
        assert "qa" in result
        assert "engineer" not in result
        assert "not found in agents config" in caplog.text

    def test_all_teammates_missing_returns_none(self, caplog):
        empty_agents = [_make_agent("unrelated")]
        with caplog.at_level(logging.WARNING):
            result = compose_team({}, "full", empty_agents)
        assert result is None


# -- Model passthrough --

class TestModelPassthrough:
    def test_default_model_is_sonnet(self, agents):
        result = compose_team({}, "standard", agents)
        assert result["qa"]["model"] == "sonnet"

    def test_custom_model_passed_through(self, agents):
        result = compose_team(
            {}, "full", agents, default_model="claude-sonnet-4-5-20250929",
        )
        for teammate in result.values():
            assert teammate["model"] == "claude-sonnet-4-5-20250929"


# -- Output structure --

class TestOutputStructure:
    def test_teammate_has_required_keys(self, agents):
        result = compose_team({}, "standard", agents)
        qa = result["qa"]
        assert "model" in qa
        assert "description" in qa
        assert "prompt" in qa

    def test_prompt_comes_from_agent_config(self, agents):
        result = compose_team({}, "standard", agents)
        assert result["qa"]["prompt"] == "You are qa."


# -- Constant consistency --

class TestConstants:
    def test_workflow_teammates_and_rank_have_same_keys(self):
        assert set(WORKFLOW_TEAMMATES.keys()) == set(WORKFLOW_RANK.keys())

    def test_ranks_are_monotonically_increasing(self):
        ordered = ["simple", "standard", "full", "quality-focused"]
        for i in range(len(ordered) - 1):
            assert WORKFLOW_RANK[ordered[i]] < WORKFLOW_RANK[ordered[i + 1]]
