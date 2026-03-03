"""Tests for _extract_plan_from_response() and its integration in _handle_successful_response()."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.post_completion import PostCompletionManager
from agent_framework.core.task import PlanDocument, Task, TaskStatus, TaskType


# -- Helpers --

VALID_PLAN = {
    "objectives": ["Implement caching layer"],
    "approach": ["Add Redis client", "Create cache decorator", "Wire into API"],
    "risks": ["Redis downtime"],
    "success_criteria": ["Response time < 100ms", "Cache hit rate > 80%"],
    "files_to_modify": ["src/cache.py", "src/api/handlers.py"],
    "dependencies": ["redis>=4.0"],
}

MINIMAL_PLAN = {
    "objectives": ["Build feature"],
    "approach": ["Step 1"],
    "success_criteria": ["Tests pass"],
}


def _wrap_json(obj: dict) -> str:
    """Wrap a dict in a ```json fence block."""
    return f"```json\n{json.dumps(obj, indent=2)}\n```"


def _make_response(content: str):
    @dataclass
    class FakeResponse:
        content: str
    return FakeResponse(content=content)


# -- Unit tests for _extract_plan_from_response --

class TestExtractPlanFromResponse:
    """Unit tests for the static extraction method."""

    def test_valid_plan_extracted(self):
        content = f"Here is my plan:\n{_wrap_json(VALID_PLAN)}\nDone."
        result = Agent._extract_plan_from_response(content)

        assert result is not None
        assert isinstance(result, PlanDocument)
        assert result.objectives == VALID_PLAN["objectives"]
        assert result.approach == VALID_PLAN["approach"]
        assert result.success_criteria == VALID_PLAN["success_criteria"]
        assert result.files_to_modify == VALID_PLAN["files_to_modify"]
        assert result.risks == VALID_PLAN["risks"]
        assert result.dependencies == VALID_PLAN["dependencies"]

    def test_minimal_plan_extracted(self):
        content = _wrap_json(MINIMAL_PLAN)
        result = Agent._extract_plan_from_response(content)

        assert result is not None
        assert result.objectives == ["Build feature"]
        assert result.files_to_modify == []  # default
        assert result.risks == []  # default

    def test_wrapped_in_plan_key(self):
        """{"plan": {...}} wrapper is unwrapped automatically."""
        wrapped = {"plan": VALID_PLAN}
        content = _wrap_json(wrapped)
        result = Agent._extract_plan_from_response(content)

        assert result is not None
        assert result.objectives == VALID_PLAN["objectives"]

    def test_empty_content_returns_none(self):
        assert Agent._extract_plan_from_response("") is None
        assert Agent._extract_plan_from_response(None) is None

    def test_no_json_blocks_returns_none(self):
        content = "This is just prose with no code fences."
        assert Agent._extract_plan_from_response(content) is None

    def test_invalid_json_returns_none(self):
        content = "```json\n{not valid json\n```"
        assert Agent._extract_plan_from_response(content) is None

    def test_missing_required_fields_returns_none(self):
        """JSON block with objectives but missing approach/success_criteria is skipped."""
        incomplete = {"objectives": ["Something"], "risks": ["A risk"]}
        content = _wrap_json(incomplete)
        assert Agent._extract_plan_from_response(content) is None

    def test_non_dict_json_skipped(self):
        """A JSON array at the top level is not a plan."""
        content = '```json\n["item1", "item2"]\n```'
        assert Agent._extract_plan_from_response(content) is None

    def test_multiple_json_blocks_picks_plan(self):
        """When multiple JSON blocks exist, the first valid plan wins."""
        other_json = {"findings": [{"severity": "HIGH"}]}
        content = (
            f"Pre-scan results:\n{_wrap_json(other_json)}\n"
            f"Architecture plan:\n{_wrap_json(VALID_PLAN)}\n"
        )
        result = Agent._extract_plan_from_response(content)

        assert result is not None
        assert result.objectives == VALID_PLAN["objectives"]

    def test_dict_coercion_through_extraction(self):
        """approach as dict → coerced to list of values by PlanDocument validator."""
        plan_with_dict_approach = {
            "objectives": ["Build feature"],
            "approach": {"s1": "Clone repo", "s2": "Implement logic"},
            "success_criteria": ["Tests pass"],
        }
        content = _wrap_json(plan_with_dict_approach)
        result = Agent._extract_plan_from_response(content)

        assert result is not None
        assert result.approach == ["Clone repo", "Implement logic"]

    def test_plan_with_extra_fields_still_parses(self):
        """Extra fields in the JSON are ignored by PlanDocument."""
        extended = {**VALID_PLAN, "estimated_hours": 8, "complexity": "medium"}
        content = _wrap_json(extended)
        result = Agent._extract_plan_from_response(content)

        assert result is not None
        assert result.objectives == VALID_PLAN["objectives"]

    def test_first_block_invalid_second_valid(self):
        """If the first JSON block is malformed, extraction falls through to the second."""
        content = (
            "```json\n{broken json here\n```\n"
            f"Real plan:\n{_wrap_json(VALID_PLAN)}\n"
        )
        result = Agent._extract_plan_from_response(content)
        assert result is not None
        assert result.objectives == VALID_PLAN["objectives"]


# -- Integration test: plan populated in _handle_successful_response --

class TestPlanExtractionIntegration:
    """Verify task.plan is set when architect completes a plan step."""

    def _make_task(self, **overrides):
        defaults = dict(
            id="task-plan-001",
            type=TaskType.ARCHITECTURE,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            created_by="system",
            assigned_to="architect",
            title="Plan the feature",
            description="Create architecture plan",
            context={"workflow_step": "plan", "workflow": True, "chain_step": True},
            created_at=datetime.now(timezone.utc),
        )
        defaults.update(overrides)
        return Task(**defaults)

    def _make_agent(self):
        workspace = Path("/tmp/agent-test-plan")
        agent = MagicMock()
        agent.config = MagicMock()
        agent.config.base_id = "architect"
        agent.config.id = "architect-1"
        agent._optimization_config = {}
        agent._self_eval_enabled = False
        agent._agent_definition = None
        agent.logger = MagicMock()
        agent.queue = MagicMock()
        agent.workspace = workspace
        agent._run_sandbox_tests = AsyncMock(return_value=None)
        agent.activity_manager = MagicMock()
        agent._session_logger = MagicMock()

        # PostCompletionManager handles plan extraction, verdicts, and routing
        budget_mock = MagicMock()
        budget_mock.estimate_cost.return_value = 0.0
        review_cycle_mock = MagicMock()
        # _parse_review_outcome_audited must return a 2-tuple (outcome, audit)
        outcome_mock = MagicMock(approved=True, needs_fix=False)
        audit_mock = MagicMock(method="")
        review_cycle_mock._parse_review_outcome_audited.return_value = (outcome_mock, audit_mock)
        agent._post_completion = PostCompletionManager(
            config=agent.config, queue=agent.queue, workspace=workspace,
            logger=agent.logger, session_logger=agent._session_logger,
            activity_manager=MagicMock(), review_cycle=review_cycle_mock,
            workflow_router=MagicMock(), git_ops=MagicMock(),
            budget=budget_mock, error_recovery=MagicMock(),
            optimization_config={}, session_logging_enabled=False,
            session_logs_dir=workspace / "logs",
        )
        agent._analytics = MagicMock()
        agent._analytics.extract_and_store_memories = MagicMock()
        agent._analytics.analyze_tool_patterns = MagicMock(return_value=None)
        agent._context_window_manager = None

        # Bind the real method + static helper
        agent._handle_successful_response = Agent._handle_successful_response.__get__(agent)
        agent._extract_plan_from_response = Agent._extract_plan_from_response
        return agent

    @pytest.mark.asyncio
    @patch("agent_framework.core.routing.read_routing_signal", return_value=None)
    async def test_plan_extracted_on_architect_plan_step(self, _mock_routing):
        agent = self._make_agent()
        task = self._make_task()
        assert task.plan is None

        response = _make_response(f"Here's the plan:\n{_wrap_json(VALID_PLAN)}")
        await agent._handle_successful_response(task, response, "2026-01-01T00:00:00Z")

        assert task.plan is not None
        assert task.plan.objectives == VALID_PLAN["objectives"]
        assert task.plan.approach == VALID_PLAN["approach"]
        agent.logger.info.assert_any_call(
            f"Extracted plan from response: {len(VALID_PLAN['files_to_modify'])} files, "
            f"{len(VALID_PLAN['approach'])} steps"
        )

    @pytest.mark.asyncio
    @patch("agent_framework.core.routing.read_routing_signal", return_value=None)
    async def test_no_plan_warns_on_architect_plan_step(self, _mock_routing):
        agent = self._make_agent()
        task = self._make_task()

        response = _make_response("I analyzed the codebase but produced no JSON plan.")
        await agent._handle_successful_response(task, response, "2026-01-01T00:00:00Z")

        assert task.plan is None
        agent.logger.warning.assert_any_call(
            "Architect plan step completed but no PlanDocument found in response"
        )

    @pytest.mark.asyncio
    @patch("agent_framework.core.routing.read_routing_signal", return_value=None)
    async def test_skipped_for_non_architect(self, _mock_routing):
        """Engineer agents don't trigger plan extraction."""
        agent = self._make_agent()
        agent.config.base_id = "engineer"
        task = self._make_task(context={"workflow_step": "implement", "workflow": True, "chain_step": True})

        response = _make_response(f"Done:\n{_wrap_json(VALID_PLAN)}")
        await agent._handle_successful_response(task, response, "2026-01-01T00:00:00Z")

        assert task.plan is None

    @pytest.mark.asyncio
    @patch("agent_framework.core.routing.read_routing_signal", return_value=None)
    async def test_skipped_for_non_plan_step(self, _mock_routing):
        """Architect on a code_review step doesn't trigger plan extraction."""
        agent = self._make_agent()
        task = self._make_task(context={"workflow_step": "code_review", "workflow": True, "chain_step": True})

        response = _make_response(f"Review:\n{_wrap_json(VALID_PLAN)}")
        await agent._handle_successful_response(task, response, "2026-01-01T00:00:00Z")

        assert task.plan is None

    @pytest.mark.asyncio
    @patch("agent_framework.core.routing.read_routing_signal", return_value=None)
    async def test_skipped_when_plan_already_set(self, _mock_routing):
        """Don't overwrite an existing plan."""
        agent = self._make_agent()
        existing_plan = PlanDocument(
            objectives=["Original"],
            approach=["Keep this"],
            success_criteria=["Unchanged"],
        )
        task = self._make_task(plan=existing_plan)

        response = _make_response(f"New plan:\n{_wrap_json(VALID_PLAN)}")
        await agent._handle_successful_response(task, response, "2026-01-01T00:00:00Z")

        assert task.plan.objectives == ["Original"]
