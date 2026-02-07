"""Tests for dynamic team composition logic."""

import logging

import pytest

from typing import Optional

from agent_framework.core.config import AgentDefinition, TeammateDefinition
from agent_framework.core.team_composer import (
    compose_default_team,
    compose_team,
    WORKFLOW_TEAMMATES,
    WORKFLOW_RANK,
)


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
        ordered = ["simple", "standard", "full"]
        for i in range(len(ordered) - 1):
            assert WORKFLOW_RANK[ordered[i]] < WORKFLOW_RANK[ordered[i + 1]]


# -- compose_default_team --

def _make_teammate(description: str, prompt: str, model: Optional[str] = None) -> TeammateDefinition:
    return TeammateDefinition(description=description, prompt=prompt, model=model)


class TestComposeDefaultTeam:
    def test_returns_none_when_no_teammates(self):
        agent_def = _make_agent("engineer")
        result = compose_default_team(agent_def)
        assert result is None

    def test_returns_dict_with_correct_format(self):
        agent_def = _make_agent("engineer")
        agent_def.teammates = {
            "peer-engineer": _make_teammate("Peer reviewer", "Review code."),
        }
        result = compose_default_team(agent_def)
        assert result is not None
        assert "peer-engineer" in result
        assert result["peer-engineer"]["description"] == "Peer reviewer"
        assert result["peer-engineer"]["prompt"] == "Review code."

    def test_respects_teammate_model_override(self):
        agent_def = _make_agent("engineer")
        agent_def.teammates = {
            "fast-helper": _make_teammate("Fast helper", "Help.", model="haiku"),
        }
        result = compose_default_team(agent_def)
        assert result["fast-helper"]["model"] == "haiku"

    def test_uses_default_model_when_teammate_model_is_none(self):
        agent_def = _make_agent("engineer")
        agent_def.teammates = {
            "helper": _make_teammate("Helper", "Help."),
        }
        result = compose_default_team(agent_def, default_model="opus")
        assert result["helper"]["model"] == "opus"

    def test_multiple_teammates(self):
        agent_def = _make_agent("engineer")
        agent_def.teammates = {
            "peer-engineer": _make_teammate("Peer", "Review."),
            "test-runner": _make_teammate("Tests", "Run tests.", model="haiku"),
        }
        result = compose_default_team(agent_def)
        assert set(result.keys()) == {"peer-engineer", "test-runner"}


# -- Merge behavior (configured teammates + workflow agents) --

class TestTeamMerge:
    """Tests for the merge logic that lives in agent.py.

    We replicate the merge algorithm here to test it in isolation
    without needing a full Agent instance.
    """

    @staticmethod
    def _merge_teams(
        agent_def: AgentDefinition,
        workflow: str,
        agents_config: list,
        default_model: str = "sonnet",
        min_workflow: str = "standard",
        caller_agent_id: Optional[str] = None,
        task_context: Optional[dict] = None,
        logger_instance: Optional[logging.Logger] = None,
    ) -> Optional[dict]:
        """Mirrors the merge logic from agent.py.

        Note: team_override is handled by the outer guard in agent.py
        before the merge runs. Pass task_context to exercise compose_team's
        own override handling if needed.
        """
        if task_context is None:
            task_context = {}

        team_agents = {}

        if agent_def.teammates:
            configured = compose_default_team(agent_def, default_model=default_model)
            if configured:
                team_agents.update(configured)

        workflow_teammates = compose_team(
            task_context, workflow, agents_config,
            min_workflow=min_workflow,
            default_model=default_model,
            caller_agent_id=caller_agent_id,
        )
        if workflow_teammates:
            collisions = sorted(set(team_agents) & set(workflow_teammates))
            if collisions and logger_instance:
                logger_instance.warning(
                    f"Teammate ID collision: {collisions} - workflow agents take precedence"
                )
            team_agents.update(workflow_teammates)

        return team_agents or None

    def test_engineer_standard_gets_configured_and_qa(self, agents):
        """Engineer with configured teammates + standard workflow → all merged."""
        engineer_def = _make_agent("engineer")
        engineer_def.teammates = {
            "peer-engineer": _make_teammate("Peer", "Review."),
            "test-runner": _make_teammate("Tests", "Run tests."),
        }
        result = self._merge_teams(
            engineer_def, "standard", agents, caller_agent_id="engineer",
        )
        assert result is not None
        assert set(result.keys()) == {"peer-engineer", "test-runner", "qa"}

    def test_engineer_simple_gets_only_configured(self, agents):
        """Simple workflow adds no workflow agents, only configured teammates remain."""
        engineer_def = _make_agent("engineer")
        engineer_def.teammates = {
            "peer-engineer": _make_teammate("Peer", "Review."),
        }
        result = self._merge_teams(
            engineer_def, "simple", agents, caller_agent_id="engineer",
        )
        assert result is not None
        assert set(result.keys()) == {"peer-engineer"}

    def test_architect_full_gets_configured_and_workflow(self, agents):
        architect_def = _make_agent("architect")
        architect_def.teammates = {
            "principal-engineer": _make_teammate("Principal", "Advise."),
        }
        result = self._merge_teams(
            architect_def, "full", agents, caller_agent_id="architect",
        )
        assert result is not None
        assert "principal-engineer" in result
        assert "engineer" in result
        assert "qa" in result

    def test_qa_standard_excludes_self_from_workflow(self, agents):
        """QA is the caller — standard workflow only adds QA, which gets excluded."""
        qa_def = _make_agent("qa")
        qa_def.teammates = {
            "security-reviewer": _make_teammate("Security", "Check security."),
        }
        result = self._merge_teams(
            qa_def, "standard", agents, caller_agent_id="qa",
        )
        assert result is not None
        # Only configured teammate remains; workflow QA excluded as caller
        assert set(result.keys()) == {"security-reviewer"}

    def test_collision_workflow_takes_precedence(self, agents):
        """When configured teammate shares an ID with workflow agent, workflow wins."""
        engineer_def = _make_agent("engineer")
        engineer_def.teammates = {
            "qa": _make_teammate("Custom QA", "Custom prompt."),
        }
        result = self._merge_teams(
            engineer_def, "standard", agents, caller_agent_id="engineer",
        )
        assert result is not None
        assert "qa" in result
        # Workflow QA prompt comes from the agent config, not the configured teammate
        assert result["qa"]["prompt"] == "You are qa."

    def test_collision_is_logged(self, agents, caplog):
        engineer_def = _make_agent("engineer")
        engineer_def.teammates = {
            "qa": _make_teammate("Custom QA", "Custom prompt."),
        }
        with caplog.at_level(logging.WARNING):
            self._merge_teams(
                engineer_def, "standard", agents,
                caller_agent_id="engineer",
                logger_instance=logging.getLogger(__name__),
            )
        assert "collision" in caplog.text.lower()

    def test_no_configured_no_workflow_returns_none(self, agents):
        """Agent with no teammates + simple workflow → None."""
        agent_def = _make_agent("engineer")
        result = self._merge_teams(
            agent_def, "simple", agents, caller_agent_id="engineer",
        )
        assert result is None
