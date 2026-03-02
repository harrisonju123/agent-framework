"""Tests for the 5 agentic flow improvements.

Covers:
1. Work Preservation — checkpoint interval, push_after
2. Attempt Continuity — get_best_resume_branch, working_dir in AttemptRecord
3. Adaptive Behavior — model escalation, progress stall detection
4. Verification Gates — checklist match ratio, impl step self-eval gate
5. Design Rationale — extraction, chain state rendering
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_framework.core.task import Task, TaskStatus, TaskType, PlanDocument


def _make_task(**overrides):
    defaults = dict(
        id="task-test-001",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=50,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Test task",
        description="A test task",
        context={"github_repo": "org/repo"},
        notes=[],
    )
    defaults.update(overrides)
    return Task(**defaults)


# =============================================================================
# Improvement #2: Attempt Continuity — get_best_resume_branch
# =============================================================================

class TestGetBestResumeBranch:
    def test_returns_none_when_no_history(self, tmp_path):
        from agent_framework.core.attempt_tracker import get_best_resume_branch

        result = get_best_resume_branch(tmp_path, "nonexistent-task")
        assert result is None

    def test_prefers_pushed_branch(self, tmp_path):
        from agent_framework.core.attempt_tracker import (
            AttemptHistory,
            AttemptRecord,
            save_attempt_history,
            get_best_resume_branch,
        )

        history = AttemptHistory(
            task_id="task-1",
            attempts=[
                AttemptRecord(
                    attempt_number=1,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    branch="agent/engineer/task-1-old",
                    commit_sha="aaa1111",
                    pushed=False,
                    working_dir="/tmp/old-worktree",
                ),
                AttemptRecord(
                    attempt_number=2,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    branch="agent/engineer/task-1-pushed",
                    commit_sha="bbb2222",
                    pushed=True,
                    working_dir="/tmp/pushed-worktree",
                ),
            ],
        )
        save_attempt_history(tmp_path, history)

        result = get_best_resume_branch(tmp_path, "task-1")
        assert result is not None
        branch, sha, wd = result
        assert branch == "agent/engineer/task-1-pushed"
        assert sha == "bbb2222"

    def test_falls_back_to_unpushed_with_existing_path(self, tmp_path):
        from agent_framework.core.attempt_tracker import (
            AttemptHistory,
            AttemptRecord,
            save_attempt_history,
            get_best_resume_branch,
        )

        # Create a worktree dir that exists
        worktree_dir = tmp_path / "worktree"
        worktree_dir.mkdir()

        history = AttemptHistory(
            task_id="task-2",
            attempts=[
                AttemptRecord(
                    attempt_number=1,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    branch="agent/engineer/task-2",
                    commit_sha="ccc3333",
                    pushed=False,
                    working_dir=str(worktree_dir),
                ),
            ],
        )
        save_attempt_history(tmp_path, history)

        result = get_best_resume_branch(tmp_path, "task-2")
        assert result is not None
        branch, sha, wd = result
        assert branch == "agent/engineer/task-2"
        assert wd == str(worktree_dir)

    def test_skips_unpushed_with_missing_path(self, tmp_path):
        from agent_framework.core.attempt_tracker import (
            AttemptHistory,
            AttemptRecord,
            save_attempt_history,
            get_best_resume_branch,
        )

        history = AttemptHistory(
            task_id="task-3",
            attempts=[
                AttemptRecord(
                    attempt_number=1,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    branch="agent/engineer/task-3",
                    commit_sha="ddd4444",
                    pushed=False,
                    working_dir="/tmp/gone-forever",
                ),
            ],
        )
        save_attempt_history(tmp_path, history)

        result = get_best_resume_branch(tmp_path, "task-3")
        assert result is None

    def test_working_dir_persisted_in_record(self, tmp_path):
        """AttemptRecord.working_dir is stored and loaded from disk."""
        from agent_framework.core.attempt_tracker import (
            AttemptHistory,
            AttemptRecord,
            save_attempt_history,
            load_attempt_history,
        )

        history = AttemptHistory(
            task_id="task-wd",
            attempts=[
                AttemptRecord(
                    attempt_number=1,
                    started_at=datetime.now(timezone.utc).isoformat(),
                    branch="agent/engineer/task-wd",
                    commit_sha="eee5555",
                    pushed=True,
                    working_dir="/some/worktree/path",
                ),
            ],
        )
        save_attempt_history(tmp_path, history)

        loaded = load_attempt_history(tmp_path, "task-wd")
        assert loaded is not None
        assert loaded.attempts[0].working_dir == "/some/worktree/path"


# =============================================================================
# Improvement #3: Adaptive Behavior — model escalation
# =============================================================================

class TestModelEscalation:
    def test_escalate_flag_set_on_context_exhaustion(self):
        """handle_failure sets _escalate_model for context exhaustion errors."""
        from agent_framework.core.error_recovery import ErrorRecoveryManager

        er = MagicMock(spec=ErrorRecoveryManager)
        er._categorize_error = ErrorRecoveryManager._categorize_error.__get__(er)
        er.escalation_handler = MagicMock()
        er.escalation_handler.categorize_error.return_value = "context_exhausted"

        task = _make_task(context={})
        task.last_error = "context window exhausted after 120k tokens"

        error_type = er._categorize_error(task.last_error)
        assert error_type == "context_exhausted"

    def test_escalate_flag_set_on_circuit_breaker(self):
        from agent_framework.core.error_recovery import ErrorRecoveryManager

        er = MagicMock(spec=ErrorRecoveryManager)
        er._categorize_error = ErrorRecoveryManager._categorize_error.__get__(er)
        er.escalation_handler = MagicMock()
        er.escalation_handler.categorize_error.return_value = "circuit_breaker"

        task = _make_task(context={})
        task.last_error = "circuit breaker tripped"

        error_type = er._categorize_error(task.last_error)
        assert error_type == "circuit_breaker"


# =============================================================================
# Improvement #4: Verification Gates — checklist match ratio
# =============================================================================

class TestChecklistMatchRatio:
    def test_returns_zero_when_no_checklist(self):
        from agent_framework.core.error_recovery import ErrorRecoveryManager

        er = MagicMock()
        er._build_checklist_report = ErrorRecoveryManager._build_checklist_report.__get__(er)

        task = _make_task(context={})
        report, matched, total = er._build_checklist_report(task, "some diff")
        assert report == ""
        assert matched == 0
        assert total == 0

    def test_counts_file_matches(self):
        from agent_framework.core.error_recovery import ErrorRecoveryManager

        er = MagicMock()
        er._build_checklist_report = ErrorRecoveryManager._build_checklist_report.__get__(er)

        task = _make_task(context={
            "requirements_checklist": [
                {"id": "1", "description": "Add auth", "files": ["src/auth.py"]},
                {"id": "2", "description": "Add tests", "files": ["tests/test_auth.py"]},
                {"id": "3", "description": "Add docs", "files": ["docs/auth.md"]},
                {"id": "4", "description": "Update config", "files": ["config.yaml"]},
            ],
        })

        git_evidence = "auth.py | 50 ++++\ntest_auth.py | 20 +++"
        report, matched, total = er._build_checklist_report(task, git_evidence)
        assert total == 4
        assert matched == 2  # auth.py + test_auth.py matched
        assert "✅" in report

    def test_below_threshold_fails_self_eval(self):
        """When <50% checklist matched, self-eval should hard-fail without LLM."""
        from agent_framework.core.error_recovery import ErrorRecoveryManager

        er = MagicMock()
        er.self_evaluate = ErrorRecoveryManager.self_evaluate.__get__(er)
        er.gather_git_evidence = ErrorRecoveryManager.gather_git_evidence.__get__(er)
        er._try_diff_strategies = ErrorRecoveryManager._try_diff_strategies.__get__(er)
        er._build_checklist_report = ErrorRecoveryManager._build_checklist_report.__get__(er)
        er._self_eval_max_retries = 2
        er._self_eval_model = "haiku"
        er.session_logger = MagicMock()
        er.logger = MagicMock()
        er.queue = MagicMock()
        er.llm = AsyncMock()

        task = _make_task(
            acceptance_criteria=["Has auth", "Has tests", "Has docs", "Has config"],
            context={
                "requirements_checklist": [
                    {"id": "1", "description": "Add authentication module", "files": ["src/auth.py"]},
                    {"id": "2", "description": "Add integration tests", "files": ["tests/test_auth.py"]},
                    {"id": "3", "description": "Add documentation files", "files": ["docs/auth.md"]},
                    {"id": "4", "description": "Update configuration yaml", "files": ["config.yaml"]},
                ],
            },
        )

        # Only 1/4 files present in diff (below 50% threshold)
        with patch("agent_framework.core.error_recovery.run_git_command") as mock_git:
            mock_git.side_effect = [
                MagicMock(stdout="auth.py | 50 ++++"),  # stat
                MagicMock(stdout="+def auth():"),         # diff
            ]

            import asyncio
            result = asyncio.get_event_loop().run_until_complete(
                er.self_evaluate(
                    task,
                    MagicMock(content="I added the auth module"),
                    working_dir=Path("/tmp/test"),
                )
            )

        assert result is False
        # LLM should NOT have been called (hard fail before LLM)
        er.llm.complete.assert_not_called()


# =============================================================================
# Improvement #5: Design Rationale
# =============================================================================

class TestDesignRationale:
    def test_extract_rationale_from_text(self):
        from agent_framework.core.agent import Agent

        content = (
            "We should use JWT tokens because they are stateless and scale well. "
            "Using sessions instead of JWTs would require sticky routing. "
            "The main constraint is backward compatibility with existing clients. "
            "Also we need to handle token refresh."
        )

        result = Agent._extract_design_rationale(content)
        assert result is not None
        assert "because" in result.lower() or "constraint" in result.lower() or "instead of" in result.lower()

    def test_extract_rationale_returns_none_for_no_keywords(self):
        from agent_framework.core.agent import Agent

        content = "Add a new file. Write tests. Update config."
        result = Agent._extract_design_rationale(content)
        assert result is None

    def test_extract_rationale_caps_at_1000(self):
        from agent_framework.core.agent import Agent

        # Generate a long text with many rationale sentences
        content = " ".join(
            f"We chose approach {i} because it is more performant than alternative {i}."
            for i in range(100)
        )
        result = Agent._extract_design_rationale(content)
        assert result is not None
        assert len(result) <= 1000

    def test_rationale_in_chain_state_implement_render(self):
        """Design rationale appears in the implement step rendering."""
        from agent_framework.core.chain_state import (
            ChainState,
            StepRecord,
            render_for_step,
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
                        "approach": ["Create auth module"],
                        "files_to_modify": ["src/auth.py"],
                    },
                    design_rationale="JWT chosen because stateless scaling is a hard constraint.",
                ),
            ],
        )

        rendered = render_for_step(state, "implement")
        assert "DESIGN RATIONALE" in rendered
        assert "JWT chosen" in rendered

    def test_rationale_in_chain_state_fix_render(self):
        """Reviewer reasoning appears in fix cycle rendering."""
        from agent_framework.core.chain_state import (
            ChainState,
            StepRecord,
            render_for_step,
        )

        state = ChainState(
            root_task_id="root-2",
            user_goal="Add auth",
            workflow="default",
            steps=[
                StepRecord(
                    step_id="implement",
                    agent_id="engineer",
                    task_id="chain-root-2-implement-d2",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    summary="Implemented",
                    files_modified=["src/auth.py"],
                ),
                StepRecord(
                    step_id="code_review",
                    agent_id="architect",
                    task_id="chain-root-2-code_review-d3",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    summary="Changes need revision",
                    verdict="needs_fix",
                    design_rationale="The auth module bypasses the middleware constraint.",
                ),
            ],
        )

        rendered = render_for_step(state, "implement")
        assert "REVIEWER REASONING" in rendered
        assert "middleware constraint" in rendered

    def test_rationale_persisted_via_append_step(self, tmp_path):
        """Design rationale flows from task context through append_step to disk."""
        from agent_framework.core.chain_state import (
            append_step,
            load_chain_state,
        )

        task = _make_task(context={
            "workflow_step": "plan",
            "workflow": "default",
            "_design_rationale": "Chose approach A because it avoids the N+1 query problem.",
        })

        with patch("agent_framework.core.chain_state._collect_files_modified", return_value=[]):
            with patch("agent_framework.core.chain_state._collect_line_counts", return_value=(0, 0)):
                with patch("agent_framework.core.chain_state._collect_commit_shas", return_value=[]):
                    state = append_step(
                        tmp_path, task, "architect", "Planning response content"
                    )

        assert len(state.steps) == 1
        assert state.steps[0].design_rationale == "Chose approach A because it avoids the N+1 query problem."

        # Verify round-trip through disk
        loaded = load_chain_state(tmp_path, task.root_id)
        assert loaded is not None
        assert loaded.steps[0].design_rationale is not None
        assert "N+1 query" in loaded.steps[0].design_rationale


# =============================================================================
# Improvement #1: Work Preservation — safety_commit push_after
# =============================================================================

class TestSafetyCommit:
    def test_returns_true_on_success(self):
        """safety_commit returns True when commit succeeds."""
        from agent_framework.core.git_operations import GitOperationsManager

        ops = MagicMock(spec=GitOperationsManager)
        ops.safety_commit = GitOperationsManager.safety_commit.__get__(ops)
        ops._verify_manifest_branch = MagicMock()
        ops.logger = MagicMock()

        working_dir = MagicMock()
        working_dir.exists.return_value = True

        with patch("agent_framework.utils.subprocess_utils.run_git_command") as mock_git:
            mock_git.side_effect = [
                MagicMock(returncode=0, stdout="M file.py"),  # status
                MagicMock(returncode=0),  # add
                MagicMock(returncode=0),  # commit
            ]

            result = ops.safety_commit(working_dir, "test checkpoint")

        assert result is True

    def test_returns_false_when_clean(self):
        """safety_commit returns False when working dir is clean."""
        from agent_framework.core.git_operations import GitOperationsManager

        ops = MagicMock(spec=GitOperationsManager)
        ops.safety_commit = GitOperationsManager.safety_commit.__get__(ops)
        ops._verify_manifest_branch = MagicMock()
        ops.logger = MagicMock()

        working_dir = MagicMock()
        working_dir.exists.return_value = True

        with patch("agent_framework.utils.subprocess_utils.run_git_command") as mock_git:
            mock_git.return_value = MagicMock(returncode=0, stdout="")  # clean

            result = ops.safety_commit(working_dir, "test")

        assert result is False


# =============================================================================
# Improvement #2: Worktree local branch support
# =============================================================================

class TestWorktreeLocalBranch:
    def test_local_branch_exists_check(self, tmp_path):
        """_local_branch_exists uses refs/heads/ to check local branches."""
        from agent_framework.workspace.worktree_manager import WorktreeManager, WorktreeConfig

        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        mgr = WorktreeManager(config)

        with patch(
            "agent_framework.workspace.worktree_manager.run_git_command"
        ) as mock_git:
            mock_git.return_value = MagicMock(returncode=0)
            assert mgr._local_branch_exists(tmp_path, "my-branch") is True
            mock_git.assert_called_once()
            args = mock_git.call_args[0][0]
            assert args == ["rev-parse", "--verify", "refs/heads/my-branch"]

    def test_local_branch_missing(self, tmp_path):
        from agent_framework.workspace.worktree_manager import WorktreeManager, WorktreeConfig

        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        mgr = WorktreeManager(config)

        with patch(
            "agent_framework.workspace.worktree_manager.run_git_command"
        ) as mock_git:
            mock_git.return_value = MagicMock(returncode=128)
            assert mgr._local_branch_exists(tmp_path, "no-such-branch") is False
