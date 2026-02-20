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
    build_workflow_summary,
    load_chain_state,
    save_chain_state,
    render_for_step,
    _build_step_summary,
    _find_step,
    _iso_delta_seconds,
    _parse_shortstat,
    _render_tool_stats,
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


class TestToolStats:
    def test_tool_stats_default_none(self):
        record = StepRecord(
            step_id="implement", agent_id="engineer", task_id="t1",
            completed_at="2026-02-19T10:00:00+00:00", summary="test",
        )
        assert record.tool_stats is None

    def test_tool_stats_roundtrip(self, workspace):
        """tool_stats survives save → load cycle."""
        tool_stats = {
            "total_calls": 42,
            "tool_distribution": {"Read": 20, "Edit": 10, "Grep": 7, "Bash": 5},
            "duplicate_reads": {"/config.py": 3},
            "edit_density": 0.238,
        }
        state = ChainState(
            root_task_id="root-1",
            user_goal="test",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="implement", agent_id="engineer", task_id="t1",
                    completed_at="2026-02-19T10:00:00+00:00", summary="done",
                    tool_stats=tool_stats,
                ),
            ],
        )
        save_chain_state(workspace, state)
        loaded = load_chain_state(workspace, "root-1")
        assert loaded.steps[0].tool_stats is not None
        assert loaded.steps[0].tool_stats["total_calls"] == 42
        assert loaded.steps[0].tool_stats["duplicate_reads"] == {"/config.py": 3}

    def test_render_tool_stats_with_data(self):
        record = StepRecord(
            step_id="implement", agent_id="engineer", task_id="t1",
            completed_at="2026-02-19T10:00:00+00:00", summary="done",
            tool_stats={
                "total_calls": 74,
                "tool_distribution": {"Read": 35, "Grep": 10, "Bash": 12, "Edit": 8, "Write": 5, "Glob": 4},
                "duplicate_reads": {"/config.py": 4, "/models.py": 3},
                "edit_density": 0.176,
            },
        )
        lines = _render_tool_stats(record)
        text = "\n".join(lines)
        assert "TOOL USAGE" in text
        assert "74 calls" in text
        assert "Read: 35" in text
        assert "2 duplicate reads" in text
        assert "/config.py: 4x" in text
        assert "17.6%" in text

    def test_render_tool_stats_none(self):
        record = StepRecord(
            step_id="implement", agent_id="engineer", task_id="t1",
            completed_at="2026-02-19T10:00:00+00:00", summary="done",
        )
        assert _render_tool_stats(record) == []

    def test_render_tool_stats_zero_calls(self):
        record = StepRecord(
            step_id="implement", agent_id="engineer", task_id="t1",
            completed_at="2026-02-19T10:00:00+00:00", summary="done",
            tool_stats={"total_calls": 0, "tool_distribution": {}},
        )
        assert _render_tool_stats(record) == []

    def test_code_review_renders_tool_stats(self):
        """Code review context includes tool stats from implement step."""
        state = ChainState(
            root_task_id="root-1",
            user_goal="test",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="plan", agent_id="architect", task_id="t1",
                    completed_at="2026-02-19T10:00:00+00:00", summary="planned",
                    plan={"objectives": ["do stuff"]},
                ),
                StepRecord(
                    step_id="implement", agent_id="engineer", task_id="t2",
                    completed_at="2026-02-19T10:05:00+00:00", summary="done",
                    files_modified=["src/a.py"],
                    tool_stats={
                        "total_calls": 50,
                        "tool_distribution": {"Read": 30, "Edit": 20},
                        "duplicate_reads": {},
                        "edit_density": 0.4,
                    },
                ),
            ],
        )
        result = render_for_step(state, "code_review")
        assert "TOOL USAGE" in result
        assert "50 calls" in result

class TestParseShortstat:
    def test_full_output(self):
        assert _parse_shortstat("3 files changed, 10 insertions(+), 2 deletions(-)") == (10, 2)

    def test_insertions_only(self):
        assert _parse_shortstat("1 file changed, 5 insertions(+)") == (5, 0)

    def test_deletions_only(self):
        assert _parse_shortstat("2 files changed, 7 deletions(-)") == (0, 7)

    def test_singular_forms(self):
        assert _parse_shortstat("1 file changed, 1 insertion(+), 1 deletion(-)") == (1, 1)

    def test_no_match(self):
        assert _parse_shortstat("1 file changed") == (0, 0)

    def test_empty_string(self):
        assert _parse_shortstat("") == (0, 0)

    def test_large_numbers(self):
        assert _parse_shortstat("15 files changed, 1041 insertions(+), 203 deletions(-)") == (1041, 203)


class TestQAToolStats:
    def test_qa_review_renders_tool_stats(self):
        """QA review context includes tool stats from implement step."""
        state = ChainState(
            root_task_id="root-1",
            user_goal="test",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="plan", agent_id="architect", task_id="t1",
                    completed_at="2026-02-19T10:00:00+00:00", summary="planned",
                    plan={"success_criteria": ["tests pass"]},
                ),
                StepRecord(
                    step_id="implement", agent_id="engineer", task_id="t2",
                    completed_at="2026-02-19T10:05:00+00:00", summary="done",
                    files_modified=["src/a.py"],
                    tool_stats={
                        "total_calls": 30,
                        "tool_distribution": {"Read": 20, "Edit": 10},
                        "duplicate_reads": {"/a.py": 2},
                        "edit_density": 0.333,
                    },
                ),
                StepRecord(
                    step_id="code_review", agent_id="architect", task_id="t3",
                    completed_at="2026-02-19T10:10:00+00:00", summary="approved",
                    verdict="approved",
                ),
            ],
        )
        result = render_for_step(state, "qa_review")
        assert "TOOL USAGE" in result
        assert "30 calls" in result


class TestImplementSummaryFallback:
    """_render_for_implement falls back to plan_step.summary when plan is None."""

    def test_summary_fallback_when_plan_is_none(self):
        """Plan extraction failed but chain state has a text summary."""
        state = ChainState(
            root_task_id="root-1",
            user_goal="Add auth",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="plan", agent_id="architect", task_id="t1",
                    completed_at="2026-02-19T10:00:00+00:00",
                    summary="Plan objectives: Add JWT auth; Create middleware\nFiles to modify: src/auth.py",
                    plan=None,
                ),
            ],
        )
        result = render_for_step(state, "implement")
        assert "IMPLEMENTATION CONTEXT" in result
        assert "PLAN (from upstream agent)" in result
        assert "JWT auth" in result

    def test_returns_empty_when_no_plan_and_no_summary(self):
        """Both plan and summary are empty — fall through to legacy."""
        state = ChainState(
            root_task_id="root-1",
            user_goal="test",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="plan", agent_id="architect", task_id="t1",
                    completed_at="2026-02-19T10:00:00+00:00",
                    summary="",
                    plan=None,
                ),
            ],
        )
        result = render_for_step(state, "implement")
        assert result == ""

    def test_returns_empty_when_no_plan_step(self):
        """No plan step at all — fall through to legacy."""
        state = ChainState(
            root_task_id="root-1",
            user_goal="test",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="implement", agent_id="engineer", task_id="t1",
                    completed_at="2026-02-19T10:00:00+00:00",
                    summary="Implemented",
                ),
            ],
        )
        result = render_for_step(state, "implement")
        assert result == ""

    def test_structured_plan_still_preferred(self, chain_state_with_steps):
        """When plan dict exists, it's rendered instead of summary text."""
        chain_state_with_steps.steps = chain_state_with_steps.steps[:1]
        result = render_for_step(chain_state_with_steps, "implement")
        # Structured plan renders objectives as bullets, not the summary text
        assert "Add user authentication" in result
        assert "PLAN (from upstream agent)" not in result

    def test_summary_fallback_respects_truncation(self):
        """Long summary fallback is truncated to budget."""
        state = ChainState(
            root_task_id="root-1",
            user_goal="test",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="plan", agent_id="architect", task_id="t1",
                    completed_at="2026-02-19T10:00:00+00:00",
                    summary="x" * (CHAIN_STATE_MAX_PROMPT_CHARS + 1000),
                    plan=None,
                ),
            ],
        )
        result = render_for_step(state, "implement")
        assert len(result) <= CHAIN_STATE_MAX_PROMPT_CHARS + 50


class TestStepRecordTiming:
    def test_started_at_and_duration_defaults(self):
        record = StepRecord(
            step_id="plan", agent_id="architect", task_id="t1",
            completed_at="2026-02-19T10:00:00+00:00", summary="test",
        )
        assert record.started_at is None
        assert record.duration_seconds is None

    def test_started_at_and_duration_roundtrip(self, workspace):
        """started_at and duration_seconds survive save → load."""
        state = ChainState(
            root_task_id="root-1",
            user_goal="test",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="plan", agent_id="architect", task_id="t1",
                    completed_at="2026-02-19T10:05:00+00:00",
                    summary="planned",
                    started_at="2026-02-19T10:00:00+00:00",
                    duration_seconds=300.0,
                ),
            ],
        )
        save_chain_state(workspace, state)
        loaded = load_chain_state(workspace, "root-1")
        assert loaded.steps[0].started_at == "2026-02-19T10:00:00+00:00"
        assert loaded.steps[0].duration_seconds == 300.0

    @patch("agent_framework.core.chain_state._collect_files_modified")
    @patch("agent_framework.core.chain_state._collect_commit_shas")
    def test_append_step_computes_duration(self, mock_shas, mock_files, workspace, sample_task):
        """append_step computes duration_seconds from started_at → completed_at."""
        mock_files.return_value = []
        mock_shas.return_value = []

        started = datetime(2026, 2, 19, 10, 0, 0, tzinfo=timezone.utc)
        state = append_step(
            workspace=workspace,
            task=sample_task,
            agent_id="engineer",
            response_content="Done",
            started_at=started,
        )

        record = state.steps[0]
        assert record.started_at == started.isoformat()
        assert record.duration_seconds is not None
        assert record.duration_seconds > 0

    @patch("agent_framework.core.chain_state._collect_files_modified")
    @patch("agent_framework.core.chain_state._collect_commit_shas")
    def test_append_step_none_started_at(self, mock_shas, mock_files, workspace, sample_task):
        """When started_at is not provided, both timing fields are None."""
        mock_files.return_value = []
        mock_shas.return_value = []

        state = append_step(
            workspace=workspace,
            task=sample_task,
            agent_id="engineer",
            response_content="Done",
        )

        record = state.steps[0]
        assert record.started_at is None
        assert record.duration_seconds is None


class TestBuildWorkflowSummary:
    def test_empty_chain(self):
        state = ChainState(
            root_task_id="root-1",
            user_goal="test",
            workflow="default",
        )
        summary = build_workflow_summary(state)
        assert summary["root_task_id"] == "root-1"
        assert summary["outcome"] == "empty"
        assert summary["steps"] == []
        assert summary["total_lines_added"] == 0
        assert summary["total_duration_seconds"] is None

    def test_happy_path_five_steps(self):
        """Full workflow chain → correct waterfall structure."""
        state = ChainState(
            root_task_id="root-1",
            user_goal="Add auth",
            workflow="default",
            attempt=1,
            steps=[
                StepRecord(
                    step_id="plan", agent_id="architect", task_id="t1",
                    started_at="2026-02-19T10:00:00+00:00",
                    completed_at="2026-02-19T10:00:27+00:00",
                    duration_seconds=27.0,
                    summary="planned",
                    verdict="approved",
                ),
                StepRecord(
                    step_id="implement", agent_id="engineer", task_id="t2",
                    started_at="2026-02-19T10:00:30+00:00",
                    completed_at="2026-02-19T10:05:30+00:00",
                    duration_seconds=300.0,
                    summary="implemented",
                    files_modified=["src/auth.py", "src/middleware.py"],
                    lines_added=500, lines_removed=20,
                ),
                StepRecord(
                    step_id="code_review", agent_id="architect", task_id="t3",
                    started_at="2026-02-19T10:06:00+00:00",
                    completed_at="2026-02-19T10:07:00+00:00",
                    duration_seconds=60.0,
                    summary="approved",
                    verdict="approved",
                ),
                StepRecord(
                    step_id="qa_review", agent_id="qa", task_id="t4",
                    started_at="2026-02-19T10:07:30+00:00",
                    completed_at="2026-02-19T10:09:30+00:00",
                    duration_seconds=120.0,
                    summary="tests pass",
                    verdict="approved",
                ),
                StepRecord(
                    step_id="create_pr", agent_id="qa", task_id="t5",
                    started_at="2026-02-19T10:10:00+00:00",
                    completed_at="2026-02-19T10:10:30+00:00",
                    duration_seconds=30.0,
                    summary="PR created",
                ),
            ],
        )

        summary = build_workflow_summary(state)

        assert summary["root_task_id"] == "root-1"
        assert summary["workflow"] == "default"
        assert summary["attempt"] == 1
        assert summary["outcome"] == "completed"
        assert len(summary["steps"]) == 5
        assert summary["total_lines_added"] == 500
        assert summary["total_lines_removed"] == 20
        assert summary["files_modified_count"] == 2
        # Total duration: 10:00:00 → 10:10:30 = 630s
        assert summary["total_duration_seconds"] == 630.0

        # Spot-check individual steps
        plan_step = summary["steps"][0]
        assert plan_step["step_id"] == "plan"
        assert plan_step["agent_id"] == "architect"
        assert plan_step["duration_seconds"] == 27.0
        assert plan_step["outcome"] == "completed"

    def test_with_retry_both_steps_appear(self):
        """Plan fails + retries → both steps in waterfall with distinct outcomes."""
        state = ChainState(
            root_task_id="root-1",
            user_goal="test",
            workflow="default",
            attempt=2,
            steps=[
                StepRecord(
                    step_id="plan", agent_id="architect", task_id="t1",
                    started_at="2026-02-19T10:00:00+00:00",
                    completed_at="2026-02-19T10:00:27+00:00",
                    duration_seconds=27.0,
                    summary="plan failed",
                    error="Context exhausted",
                ),
                StepRecord(
                    step_id="plan", agent_id="architect", task_id="t1-retry",
                    started_at="2026-02-19T10:01:00+00:00",
                    completed_at="2026-02-19T10:03:00+00:00",
                    duration_seconds=120.0,
                    summary="plan succeeded on retry",
                    verdict="approved",
                ),
                StepRecord(
                    step_id="implement", agent_id="engineer", task_id="t2",
                    started_at="2026-02-19T10:03:30+00:00",
                    completed_at="2026-02-19T10:08:30+00:00",
                    duration_seconds=300.0,
                    summary="implemented",
                    lines_added=200,
                ),
            ],
        )

        summary = build_workflow_summary(state)

        assert len(summary["steps"]) == 3
        assert summary["steps"][0]["outcome"] == "failed"
        assert summary["steps"][1]["outcome"] == "completed"
        assert summary["steps"][2]["outcome"] == "completed"
        assert summary["attempt"] == 2
        # Total: 10:00:00 → 10:08:30 = 510s
        assert summary["total_duration_seconds"] == 510.0

    def test_no_changes_verdict(self):
        """Single plan step with no_changes → chain outcome reflects it."""
        state = ChainState(
            root_task_id="root-1",
            user_goal="test",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="plan", agent_id="architect", task_id="t1",
                    started_at="2026-02-19T10:00:00+00:00",
                    completed_at="2026-02-19T10:00:15+00:00",
                    duration_seconds=15.0,
                    summary="no changes needed",
                    verdict="no_changes",
                ),
            ],
        )

        summary = build_workflow_summary(state)

        assert summary["outcome"] == "no_changes"
        assert len(summary["steps"]) == 1
        assert summary["steps"][0]["outcome"] == "completed"
        assert summary["total_lines_added"] == 0

    def test_rejected_step(self):
        """Step with needs_fix verdict → step outcome is 'rejected'."""
        state = ChainState(
            root_task_id="root-1",
            user_goal="test",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="code_review", agent_id="architect", task_id="t1",
                    completed_at="2026-02-19T10:00:00+00:00",
                    summary="issues found",
                    verdict="needs_fix",
                ),
            ],
        )

        summary = build_workflow_summary(state)
        assert summary["steps"][0]["outcome"] == "rejected"

    def test_falls_back_to_completed_at_when_no_started_at(self):
        """Total duration uses completed_at timestamps when started_at is absent."""
        state = ChainState(
            root_task_id="root-1",
            user_goal="test",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="plan", agent_id="architect", task_id="t1",
                    completed_at="2026-02-19T10:00:00+00:00",
                    summary="planned",
                ),
                StepRecord(
                    step_id="implement", agent_id="engineer", task_id="t2",
                    completed_at="2026-02-19T10:05:00+00:00",
                    summary="done",
                    lines_added=100,
                    files_modified=["a.py"],
                ),
            ],
        )

        summary = build_workflow_summary(state)
        # Falls back to completed_at: 10:00:00 → 10:05:00 = 300s
        assert summary["total_duration_seconds"] == 300.0
        assert summary["files_modified_count"] == 1

    def test_aggregates_files_across_steps(self):
        """files_modified_count deduplicates across steps."""
        state = ChainState(
            root_task_id="root-1",
            user_goal="test",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="implement", agent_id="engineer", task_id="t1",
                    completed_at="2026-02-19T10:00:00+00:00",
                    summary="first pass",
                    files_modified=["a.py", "b.py"],
                    lines_added=50,
                ),
                StepRecord(
                    step_id="implement", agent_id="engineer", task_id="t2",
                    completed_at="2026-02-19T10:05:00+00:00",
                    summary="fix pass",
                    files_modified=["a.py", "c.py"],
                    lines_added=30,
                ),
            ],
        )

        summary = build_workflow_summary(state)
        assert summary["files_modified_count"] == 3  # a.py, b.py, c.py
        assert summary["total_lines_added"] == 80


class TestIsoDeltaSeconds:
    def test_valid_timestamps(self):
        assert _iso_delta_seconds("2026-02-19T10:00:00+00:00", "2026-02-19T10:05:00+00:00") == 300.0

    def test_invalid_start(self):
        assert _iso_delta_seconds("not-a-date", "2026-02-19T10:00:00+00:00") is None

    def test_none_input(self):
        assert _iso_delta_seconds(None, "2026-02-19T10:00:00+00:00") is None
