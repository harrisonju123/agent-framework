"""Integration tests for the workflow chain: plan → implement → review → PR.

Tests the full post-completion flow with real Task objects, FileQueue,
WorkflowRouter, and mock LLM. Verifies chain state accumulation, verdict
routing, and design rationale propagation across steps.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.task import Task, TaskStatus, TaskType, PlanDocument
from agent_framework.queue.file_queue import FileQueue
from agent_framework.llm.base import LLMResponse


def _make_task(tmp_path, **overrides):
    defaults = dict(
        id="test-chain-root",
        type=TaskType.PLANNING,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="cli",
        assigned_to="architect",
        created_at=datetime.now(timezone.utc),
        title="Add user authentication",
        description="Implement JWT-based auth",
        context={
            "github_repo": "org/repo",
            "workflow": "default",
            "workflow_step": "plan",
            "user_goal": "Add user authentication",
            "chain_step": True,
        },
    )
    defaults.update(overrides)
    return Task(**defaults)


class TestChainStateAccumulation:
    """Verify chain state persists across workflow steps."""

    def test_plan_step_creates_chain_state(self, tmp_path):
        """append_step creates a chain state file with plan data."""
        from agent_framework.core.chain_state import append_step, load_chain_state

        task = _make_task(tmp_path)
        task.plan = PlanDocument(
            objectives=["Add JWT auth"],
            approach=["Create auth service", "Add middleware"],
            risks=["Token expiry"],
            success_criteria=["All endpoints protected"],
            files_to_modify=["src/auth.py"],
        )
        task.context["verdict"] = "approved"
        task.context["_design_rationale"] = "JWT chosen because stateless scaling."

        with patch("agent_framework.core.chain_state._collect_files_modified", return_value=[]):
            with patch("agent_framework.core.chain_state._collect_line_counts", return_value=(0, 0)):
                with patch("agent_framework.core.chain_state._collect_commit_shas", return_value=[]):
                    state = append_step(tmp_path, task, "architect", "Plan completed")

        assert len(state.steps) == 1
        assert state.steps[0].step_id == "plan"
        assert state.steps[0].design_rationale == "JWT chosen because stateless scaling."
        assert state.steps[0].verdict == "approved"

        # Verify persisted to disk
        loaded = load_chain_state(tmp_path, task.root_id)
        assert loaded is not None
        assert len(loaded.steps) == 1

    def test_chain_state_renders_for_implement(self, tmp_path):
        """Implement step receives plan + design rationale from chain state."""
        from agent_framework.core.chain_state import (
            ChainState, StepRecord, render_for_step, save_chain_state,
        )

        state = ChainState(
            root_task_id="root-1",
            user_goal="Add auth",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="plan",
                    agent_id="architect",
                    task_id="chain-root-1-plan-d1",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    summary="Plan completed",
                    plan={
                        "objectives": ["Add JWT auth"],
                        "approach": ["Create auth service"],
                        "files_to_modify": ["src/auth.py"],
                    },
                    design_rationale="JWT because stateless.",
                ),
            ],
        )
        save_chain_state(tmp_path, state)

        rendered = render_for_step(state, "implement")
        assert "PLAN" in rendered
        assert "DESIGN RATIONALE" in rendered
        assert "JWT because stateless" in rendered


class TestVerdictRouting:
    """Verify verdict-based routing decisions."""

    def test_no_changes_verdict_skips_chain(self, tmp_path):
        """no_changes verdict at plan step terminates the workflow."""
        task = _make_task(tmp_path, context={
            "workflow": "default",
            "workflow_step": "plan",
            "verdict": "no_changes",
            "chain_step": True,
            "github_repo": "org/repo",
        })

        # _run_post_completion_flow should skip chain enforcement
        agent = MagicMock()
        agent._run_post_completion_flow = Agent._run_post_completion_flow.__get__(agent)
        agent._budget = MagicMock()
        agent._budget.estimate_cost.return_value = 0.01
        agent._workflow_router = MagicMock()
        agent._workflow_router.queue = MagicMock()
        agent._workflow_router.queue.find_task.return_value = None
        agent._workflow_router.check_and_create_fan_in_task = MagicMock()
        agent._review_cycle = MagicMock()
        agent._git_ops = MagicMock()
        agent._enforce_workflow_chain = MagicMock(return_value=False)
        agent._is_at_terminal_workflow_step = MagicMock(return_value=True)
        agent._emit_workflow_summary = MagicMock()
        agent._extract_and_store_memories = MagicMock()
        agent._analyze_tool_patterns = MagicMock(return_value=0)
        agent._log_task_completion_metrics = MagicMock()
        agent._save_pre_scan_findings = MagicMock()
        agent.logger = MagicMock()
        agent.config = MagicMock()
        agent.config.base_id = "architect"

        response = MagicMock(content="No changes needed")
        agent._run_post_completion_flow(task, response, None, datetime.now(timezone.utc))

        # Chain enforcement should NOT have been called
        agent._enforce_workflow_chain.assert_not_called()

    def test_needs_fix_verdict_blocks_pr(self, tmp_path):
        """needs_fix verdict at terminal step blocks PR creation."""
        task = _make_task(tmp_path, context={
            "workflow": "default",
            "workflow_step": "qa_review",
            "verdict": "needs_fix",
            "chain_step": True,
            "github_repo": "org/repo",
        })

        agent = MagicMock()
        agent._run_post_completion_flow = Agent._run_post_completion_flow.__get__(agent)
        agent._budget = MagicMock()
        agent._budget.estimate_cost.return_value = 0.01
        agent._workflow_router = MagicMock()
        agent._workflow_router.queue = MagicMock()
        agent._workflow_router.queue.find_task.return_value = None
        agent._workflow_router.check_and_create_fan_in_task = MagicMock()
        agent._review_cycle = MagicMock()
        agent._git_ops = MagicMock()
        agent._enforce_workflow_chain = MagicMock(return_value=False)
        agent._is_at_terminal_workflow_step = MagicMock(return_value=True)
        agent._emit_workflow_summary = MagicMock()
        agent._extract_and_store_memories = MagicMock()
        agent._analyze_tool_patterns = MagicMock(return_value=0)
        agent._log_task_completion_metrics = MagicMock()
        agent.logger = MagicMock()
        agent.config = MagicMock()
        agent.config.base_id = "qa"

        response = MagicMock(content="Needs fixes")
        agent._run_post_completion_flow(task, response, None, datetime.now(timezone.utc))

        # PR should NOT have been created
        agent._git_ops.push_and_create_pr_if_needed.assert_not_called()


class TestDesignRationaleExtraction:
    """Verify design rationale flows from planning to downstream steps."""

    def test_extract_rationale_from_planning_response(self):
        """_extract_design_rationale pulls reasoning sentences."""
        rationale = Agent._extract_design_rationale(
            "We use JWT tokens because they enable stateless auth. "
            "The main constraint is backward compatibility. "
            "We chose Redis instead of Memcached for persistence."
        )
        assert rationale is not None
        assert "because" in rationale.lower() or "constraint" in rationale.lower()

    def test_no_rationale_returns_none(self):
        assert Agent._extract_design_rationale("Add a file. Run tests.") is None
        assert Agent._extract_design_rationale("") is None
        assert Agent._extract_design_rationale(None) is None


class TestQueueFIFO:
    """Verify queue pop order uses insertion time, not filename."""

    def test_pop_returns_oldest_by_mtime(self, tmp_path):
        """Tasks are popped in insertion (mtime) order."""
        import time

        queue = FileQueue(tmp_path)

        # Create tasks with IDs that sort differently than insertion order
        task_z = Task(
            id="zzz-task",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="test",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="Task Z",
            description="Should be first (inserted first)",
        )
        queue.push(task_z, "engineer")
        time.sleep(0.05)  # ensure different mtime

        task_a = Task(
            id="aaa-task",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="test",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="Task A",
            description="Should be second (inserted second)",
        )
        queue.push(task_a, "engineer")

        # Pop should return Z first (older mtime) even though A sorts first alphabetically
        first = queue.pop("engineer")
        assert first is not None
        assert first.id == "zzz-task"
