"""Tests for chain state module."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agent_framework.core.chain_state import (
    ChainState,
    StepRecord,
    append_step,
    load_chain_state,
    save_chain_state,
    render_for_step,
    _build_step_summary,
    _find_step,
    _chain_state_path,
    CHAIN_STATE_MAX_PROMPT_CHARS,
)
from agent_framework.core.task import Task, TaskStatus, TaskType, PlanDocument


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


@pytest.fixture
def sample_plan():
    return PlanDocument(
        objectives=["Add user authentication", "Implement JWT tokens"],
        approach=["Create auth service", "Add middleware", "Write tests"],
        risks=["Token expiry edge cases"],
        success_criteria=["All endpoints protected", "Tests pass"],
        files_to_modify=["src/auth.py", "src/middleware.py", "tests/test_auth.py"],
    )


@pytest.fixture
def sample_task(sample_plan):
    return Task(
        id="chain-root-123-implement-d2",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="[chain] Implement user authentication",
        description="Add JWT-based auth",
        plan=sample_plan,
        context={
            "_root_task_id": "root-123",
            "workflow": "default",
            "workflow_step": "implement",
            "user_goal": "Add JWT-based authentication to the API",
            "verdict": "approved",
        },
    )


@pytest.fixture
def chain_state_with_steps():
    """A chain state with plan + implement + code_review steps."""
    return ChainState(
        root_task_id="root-123",
        user_goal="Add JWT-based authentication to the API",
        workflow="default",
        implementation_branch="feature/PROJ-123-auth",
        steps=[
            StepRecord(
                step_id="plan",
                agent_id="architect",
                task_id="chain-root-123-plan-d1",
                completed_at="2026-02-19T10:00:00+00:00",
                summary="Plan objectives: Add user authentication; Implement JWT tokens",
                verdict="approved",
                plan={
                    "objectives": ["Add user authentication", "Implement JWT tokens"],
                    "approach": ["Create auth service", "Add middleware", "Write tests"],
                    "risks": ["Token expiry edge cases"],
                    "success_criteria": ["All endpoints protected", "Tests pass"],
                    "files_to_modify": ["src/auth.py", "src/middleware.py"],
                },
            ),
            StepRecord(
                step_id="implement",
                agent_id="engineer",
                task_id="chain-root-123-implement-d2",
                completed_at="2026-02-19T10:15:00+00:00",
                summary="Implemented auth service and middleware",
                files_modified=["src/auth.py", "src/middleware.py", "tests/test_auth.py"],
                commit_shas=["abc1234", "def5678"],
            ),
            StepRecord(
                step_id="code_review",
                agent_id="architect",
                task_id="chain-root-123-code_review-d3",
                completed_at="2026-02-19T10:20:00+00:00",
                summary="Code review passed",
                verdict="approved",
            ),
        ],
    )


class TestStepRecord:
    def test_defaults(self):
        record = StepRecord(
            step_id="plan",
            agent_id="architect",
            task_id="task-1",
            completed_at="2026-02-19T10:00:00+00:00",
            summary="test",
        )
        assert record.verdict is None
        assert record.plan is None
        assert record.files_modified == []
        assert record.commit_shas == []
        assert record.findings is None
        assert record.error is None


class TestChainState:
    def test_defaults(self):
        state = ChainState(
            root_task_id="root-1",
            user_goal="test goal",
            workflow="default",
        )
        assert state.steps == []
        assert state.current_step is None
        assert state.attempt == 1
        assert state.implementation_branch is None


class TestSaveAndLoad:
    def test_save_and_load_roundtrip(self, workspace):
        state = ChainState(
            root_task_id="root-1",
            user_goal="Test goal",
            workflow="default",
            implementation_branch="feature/test",
            steps=[
                StepRecord(
                    step_id="plan",
                    agent_id="architect",
                    task_id="task-1",
                    completed_at="2026-02-19T10:00:00+00:00",
                    summary="Planned",
                    verdict="approved",
                    plan={"objectives": ["do thing"]},
                ),
            ],
            attempt=2,
        )

        save_chain_state(workspace, state)
        loaded = load_chain_state(workspace, "root-1")

        assert loaded is not None
        assert loaded.root_task_id == "root-1"
        assert loaded.user_goal == "Test goal"
        assert loaded.workflow == "default"
        assert loaded.implementation_branch == "feature/test"
        assert loaded.attempt == 2
        assert len(loaded.steps) == 1
        assert loaded.steps[0].step_id == "plan"
        assert loaded.steps[0].verdict == "approved"
        assert loaded.steps[0].plan == {"objectives": ["do thing"]}

    def test_load_nonexistent_returns_none(self, workspace):
        result = load_chain_state(workspace, "nonexistent")
        assert result is None

    def test_load_corrupt_file_returns_none(self, workspace):
        state_dir = workspace / ".agent-communication" / "chain-state"
        state_dir.mkdir(parents=True)
        (state_dir / "corrupt.json").write_text("not json")

        result = load_chain_state(workspace, "corrupt")
        assert result is None

    def test_load_missing_root_task_id_returns_none(self, workspace):
        """Structurally valid JSON missing required root_task_id."""
        state_dir = workspace / ".agent-communication" / "chain-state"
        state_dir.mkdir(parents=True)
        (state_dir / "bad.json").write_text('{"steps": []}')

        result = load_chain_state(workspace, "bad")
        assert result is None

    def test_load_tolerates_unknown_step_fields(self, workspace):
        """Future schema fields in step records don't crash deserialization."""
        state_dir = workspace / ".agent-communication" / "chain-state"
        state_dir.mkdir(parents=True)
        data = {
            "root_task_id": "root-1",
            "user_goal": "test",
            "workflow": "default",
            "steps": [{
                "step_id": "plan",
                "agent_id": "architect",
                "task_id": "t1",
                "completed_at": "2026-02-19T10:00:00+00:00",
                "summary": "Planned",
                "future_field": "should be ignored",
                "another_new_field": 42,
            }],
        }
        (state_dir / "root-1.json").write_text(json.dumps(data))

        result = load_chain_state(workspace, "root-1")
        assert result is not None
        assert len(result.steps) == 1
        assert result.steps[0].step_id == "plan"
        assert not hasattr(result.steps[0], "future_field")

    def test_save_creates_directory(self, workspace):
        state = ChainState(
            root_task_id="root-1",
            user_goal="test",
            workflow="default",
        )
        save_chain_state(workspace, state)

        path = _chain_state_path(workspace, "root-1")
        assert path.exists()

    def test_append_preserves_existing_steps(self, workspace):
        """Saving a state with 2 steps then loading preserves both."""
        state = ChainState(
            root_task_id="root-1",
            user_goal="test",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="plan", agent_id="architect",
                    task_id="t1", completed_at="2026-02-19T10:00:00+00:00",
                    summary="step 1",
                ),
                StepRecord(
                    step_id="implement", agent_id="engineer",
                    task_id="t2", completed_at="2026-02-19T10:05:00+00:00",
                    summary="step 2",
                ),
            ],
        )
        save_chain_state(workspace, state)
        loaded = load_chain_state(workspace, "root-1")
        assert len(loaded.steps) == 2
        assert loaded.steps[0].step_id == "plan"
        assert loaded.steps[1].step_id == "implement"


class TestAppendStep:
    @patch("agent_framework.core.chain_state._collect_files_modified")
    @patch("agent_framework.core.chain_state._collect_commit_shas")
    def test_creates_new_chain_state(self, mock_shas, mock_files, workspace, sample_task):
        mock_files.return_value = ["src/auth.py"]
        mock_shas.return_value = ["abc1234"]

        state = append_step(
            workspace=workspace,
            task=sample_task,
            agent_id="engineer",
            response_content="Implemented auth service",
        )

        assert state.root_task_id == "root-123"
        assert state.user_goal == "Add JWT-based authentication to the API"
        assert state.workflow == "default"
        assert len(state.steps) == 1
        assert state.steps[0].step_id == "implement"
        assert state.steps[0].agent_id == "engineer"
        assert state.steps[0].files_modified == ["src/auth.py"]

    @patch("agent_framework.core.chain_state._collect_files_modified")
    @patch("agent_framework.core.chain_state._collect_commit_shas")
    def test_appends_to_existing_chain_state(self, mock_shas, mock_files, workspace, sample_task):
        mock_files.return_value = []
        mock_shas.return_value = []

        # Create initial state
        initial = ChainState(
            root_task_id="root-123",
            user_goal="test",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="plan", agent_id="architect",
                    task_id="t1", completed_at="2026-02-19T10:00:00+00:00",
                    summary="Planned",
                ),
            ],
        )
        save_chain_state(workspace, initial)

        state = append_step(
            workspace=workspace,
            task=sample_task,
            agent_id="engineer",
            response_content="Implemented",
        )

        assert len(state.steps) == 2
        assert state.steps[0].step_id == "plan"
        assert state.steps[1].step_id == "implement"

    @patch("agent_framework.core.chain_state._collect_files_modified")
    @patch("agent_framework.core.chain_state._collect_commit_shas")
    def test_captures_plan_from_task(self, mock_shas, mock_files, workspace, sample_task):
        mock_files.return_value = []
        mock_shas.return_value = []

        # Set task to plan step so plan gets included
        sample_task.context["workflow_step"] = "plan"

        state = append_step(
            workspace=workspace,
            task=sample_task,
            agent_id="architect",
            response_content="Plan complete",
        )

        assert state.steps[0].plan is not None
        assert "Add user authentication" in state.steps[0].plan["objectives"]

    @patch("agent_framework.core.chain_state._collect_files_modified")
    @patch("agent_framework.core.chain_state._collect_commit_shas")
    def test_captures_verdict(self, mock_shas, mock_files, workspace, sample_task):
        mock_files.return_value = []
        mock_shas.return_value = []

        state = append_step(
            workspace=workspace,
            task=sample_task,
            agent_id="engineer",
            response_content="Done",
        )

        assert state.steps[0].verdict == "approved"

    @patch("agent_framework.core.chain_state._collect_files_modified")
    @patch("agent_framework.core.chain_state._collect_commit_shas")
    def test_captures_structured_findings(self, mock_shas, mock_files, workspace, sample_task):
        mock_files.return_value = []
        mock_shas.return_value = []

        sample_task.context["structured_findings"] = {
            "findings": [
                {"file": "src/auth.py", "severity": "HIGH", "description": "Missing input validation"},
            ]
        }

        state = append_step(
            workspace=workspace,
            task=sample_task,
            agent_id="qa",
            response_content="Found issues",
        )

        assert state.steps[0].findings is not None
        assert len(state.steps[0].findings) == 1
        assert state.steps[0].findings[0]["severity"] == "HIGH"


class TestBuildStepSummary:
    def test_uses_plan_objectives_for_plan_step(self, sample_task):
        sample_task.context["workflow_step"] = "plan"
        summary = _build_step_summary(sample_task, "", "plan")
        assert "Add user authentication" in summary
        assert "files_to_modify" in summary.lower() or "src/auth.py" in summary

    def test_includes_verdict(self, sample_task):
        summary = _build_step_summary(sample_task, "", "implement")
        assert "approved" in summary.lower()

    def test_falls_back_to_response_content(self):
        task = Task(
            id="t1", type=TaskType.IMPLEMENTATION, status=TaskStatus.COMPLETED,
            priority=1, created_by="test", assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="test", description="test",
            context={},
        )
        summary = _build_step_summary(task, "I implemented the authentication feature.", "implement")
        assert "authentication" in summary.lower()

    def test_skips_tool_call_noise(self):
        task = Task(
            id="t1", type=TaskType.IMPLEMENTATION, status=TaskStatus.COMPLETED,
            priority=1, created_by="test", assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="test", description="test",
            context={},
        )
        response = "```json\n{}\n```\nTool call result\nReading file\nImplemented the auth feature successfully."
        summary = _build_step_summary(task, response, "implement")
        assert "auth feature" in summary.lower()


class TestRenderForStep:
    def test_render_for_implement(self, chain_state_with_steps):
        # Remove implement and code_review steps to simulate engineer receiving plan
        chain_state_with_steps.steps = chain_state_with_steps.steps[:1]

        result = render_for_step(chain_state_with_steps, "implement")
        assert "PLAN" in result
        assert "Add user authentication" in result
        assert "Create auth service" in result

    def test_render_for_code_review(self, chain_state_with_steps):
        result = render_for_step(chain_state_with_steps, "code_review")
        assert "CODE REVIEW CONTEXT" in result
        assert "PLAN OBJECTIVES" in result
        assert "FILES CHANGED" in result
        assert "src/auth.py" in result

    def test_render_for_qa_review(self, chain_state_with_steps):
        result = render_for_step(chain_state_with_steps, "qa_review")
        assert "QA REVIEW CONTEXT" in result
        assert "SUCCESS CRITERIA" in result
        assert "FILES CHANGED" in result
        assert "CODE REVIEW RESULT" in result
        assert "approved" in result

    def test_render_for_create_pr(self, chain_state_with_steps):
        result = render_for_step(chain_state_with_steps, "create_pr")
        assert "PR SUMMARY" in result
        assert "JWT" in result
        assert "src/auth.py" in result

    def test_render_for_fix_cycle(self, chain_state_with_steps):
        # Set code_review verdict to needs_fix
        chain_state_with_steps.steps[2].verdict = "needs_fix"
        chain_state_with_steps.steps[2].findings = [
            {"file": "src/auth.py", "severity": "HIGH", "description": "Missing error handling"},
        ]

        result = render_for_step(chain_state_with_steps, "implement")
        assert "FIX CYCLE" in result
        assert "Missing error handling" in result

    def test_render_generic_fallback(self, chain_state_with_steps):
        result = render_for_step(chain_state_with_steps, "unknown_step")
        assert "CHAIN STATE" in result
        assert "plan" in result
        assert "implement" in result

    def test_empty_steps_returns_empty(self):
        state = ChainState(
            root_task_id="root-1",
            user_goal="test",
            workflow="default",
        )
        result = render_for_step(state, "implement")
        assert result == ""

    def test_truncation(self, chain_state_with_steps):
        """Rendered output respects the budget limit."""
        # Create a huge plan to trigger truncation
        chain_state_with_steps.steps[0].plan = {
            "objectives": [f"Objective {i}: " + "x" * 200 for i in range(100)],
            "approach": [f"Step {i}: " + "y" * 200 for i in range(100)],
            "files_to_modify": [f"src/file_{i}.py" for i in range(100)],
            "risks": ["risk"],
            "success_criteria": ["criteria"],
        }
        # Remove later steps to trigger implement rendering
        chain_state_with_steps.steps = chain_state_with_steps.steps[:1]

        result = render_for_step(chain_state_with_steps, "implement")
        assert len(result) <= CHAIN_STATE_MAX_PROMPT_CHARS + 50  # small margin for truncation msg


class TestFindStep:
    def test_finds_most_recent(self, chain_state_with_steps):
        step = _find_step(chain_state_with_steps, "plan")
        assert step is not None
        assert step.step_id == "plan"

    def test_returns_none_for_missing(self, chain_state_with_steps):
        step = _find_step(chain_state_with_steps, "nonexistent")
        assert step is None

    def test_finds_latest_when_multiple(self):
        state = ChainState(
            root_task_id="root-1",
            user_goal="test",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="implement", agent_id="engineer", task_id="t1",
                    completed_at="2026-02-19T10:00:00+00:00", summary="first",
                ),
                StepRecord(
                    step_id="implement", agent_id="engineer", task_id="t2",
                    completed_at="2026-02-19T10:05:00+00:00", summary="second",
                ),
            ],
        )
        step = _find_step(state, "implement")
        assert step.task_id == "t2"
        assert step.summary == "second"
