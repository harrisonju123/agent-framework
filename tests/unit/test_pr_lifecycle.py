"""Tests for autonomous PR lifecycle management."""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from agent_framework.core.agent import Agent, AgentConfig
from agent_framework.core.config import WorkflowDefinition
from agent_framework.core.pr_lifecycle import (
    CICheckResult,
    CIStatus,
    PRLifecycleManager,
)
from agent_framework.core.task import Task, TaskStatus, TaskType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(task_id="task-abc", parent_task_id=None, **ctx_overrides):
    context = {"workflow": "default", "github_repo": "org/repo", **ctx_overrides}
    return Task(
        id=task_id,
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Implement feature X",
        description="Build the thing.",
        context=context,
        parent_task_id=parent_task_id,
    )


def _make_manager(**overrides):
    defaults = dict(
        queue=MagicMock(),
        workspace="/tmp/ws",
        repo_configs={
            "org/repo": {"auto_merge": True, "merge_strategy": "squash", "max_ci_fix_attempts": 3},
        },
        pr_lifecycle_config={
            "ci_poll_interval": 0,  # No wait in tests
            "ci_poll_max_wait": 1,
            "max_ci_fix_attempts": 3,
            "auto_approve": True,
            "delete_branch_on_merge": True,
        },
        logger_instance=MagicMock(),
    )
    defaults.update(overrides)
    return PRLifecycleManager(**defaults)


@pytest.fixture
def manager():
    return _make_manager()


@pytest.fixture
def agent(tmp_path):
    """Minimal Agent for integration tests — follows test_subtask_pr_lifecycle pattern."""
    config = AgentConfig(
        id="engineer",
        name="Engineer",
        queue="engineer",
        prompt="You are an engineer.",
    )
    a = Agent.__new__(Agent)
    a.config = config
    a.queue = MagicMock()
    a.workspace = tmp_path
    a._workflows_config = {
        "default": WorkflowDefinition(description="default", agents=["engineer"]),
    }
    a._agents_config = [SimpleNamespace(id="engineer")]
    a._team_mode_enabled = False
    a._mcp_enabled = False
    a._optimization_config = {}
    a._guidance_cache = {}
    a.logger = MagicMock()
    a._session_logger = MagicMock()
    a._active_worktree = None
    a.worktree_manager = None
    a.multi_repo_manager = None
    a.jira_client = None
    a._agent_definition = None
    a._pr_lifecycle_manager = None
    # Initialize GitOperationsManager
    from agent_framework.core.git_operations import GitOperationsManager
    a._git_ops = GitOperationsManager(
        config=a.config,
        workspace=a.workspace,
        queue=a.queue,
        logger=a.logger,
        session_logger=a._session_logger if hasattr(a, '_session_logger') else None,
    )
    return a


# ===========================================================================
# should_manage
# ===========================================================================

class TestShouldManage:
    def test_skips_without_pr_url(self, manager):
        task = _make_task()
        assert manager.should_manage(task) is False

    def test_skips_subtasks(self, manager):
        task = _make_task(parent_task_id="parent-1", pr_url="https://github.com/org/repo/pull/10")
        assert manager.should_manage(task) is False

    def test_skips_non_auto_merge_repos(self):
        mgr = _make_manager(repo_configs={"org/repo": {"auto_merge": False}})
        task = _make_task(pr_url="https://github.com/org/repo/pull/10")
        assert mgr.should_manage(task) is False

    def test_skips_unknown_repos(self, manager):
        task = _make_task(
            github_repo="unknown/repo",
            pr_url="https://github.com/unknown/repo/pull/10",
        )
        assert manager.should_manage(task) is False

    def test_returns_true_for_auto_merge_repo_with_pr(self, manager):
        task = _make_task(pr_url="https://github.com/org/repo/pull/10")
        assert manager.should_manage(task) is True

    def test_skips_without_github_repo(self, manager):
        task = _make_task(pr_url="https://github.com/org/repo/pull/10")
        del task.context["github_repo"]
        assert manager.should_manage(task) is False


# ===========================================================================
# CI polling
# ===========================================================================

class TestCIPolling:
    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_passing_checks(self, mock_run, manager):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([
                {"name": "build", "state": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "lint", "state": "COMPLETED", "conclusion": "SUCCESS"},
            ]),
            stderr="",
        )
        result = manager._fetch_ci_status("org/repo", "10")
        assert result.status == CIStatus.PASSING

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_failing_checks(self, mock_run, manager):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([
                {"name": "build", "state": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "tests", "state": "COMPLETED", "conclusion": "FAILURE"},
            ]),
            stderr="",
        )
        result = manager._fetch_ci_status("org/repo", "10")
        assert result.status == CIStatus.FAILING
        assert "tests" in result.failed_checks

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_pending_checks(self, mock_run, manager):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([
                {"name": "build", "state": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "deploy", "state": "PENDING", "conclusion": ""},
            ]),
            stderr="",
        )
        result = manager._fetch_ci_status("org/repo", "10")
        assert result.status == CIStatus.PENDING

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_empty_checks_treated_as_passing(self, mock_run, manager):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=json.dumps([]), stderr=""
        )
        result = manager._fetch_ci_status("org/repo", "10")
        assert result.status == CIStatus.PASSING

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_command_failure_returns_error(self, mock_run, manager):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="gh: not found"
        )
        result = manager._fetch_ci_status("org/repo", "10")
        assert result.status == CIStatus.ERROR

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_invalid_json_returns_error(self, mock_run, manager):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="not json", stderr=""
        )
        result = manager._fetch_ci_status("org/repo", "10")
        assert result.status == CIStatus.ERROR

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_nonzero_exit_with_valid_json_parses_checks(self, mock_run, manager):
        """gh exits 1 when checks fail but still writes valid JSON — must parse it."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout=json.dumps([
                {"name": "tests", "state": "COMPLETED", "conclusion": "FAILURE"},
            ]),
            stderr="Some checks were not successful",
        )
        result = manager._fetch_ci_status("org/repo", "10")
        assert result.status == CIStatus.FAILING
        assert "tests" in result.failed_checks

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_error_conclusion_treated_as_failure(self, mock_run, manager):
        """conclusion=ERROR must be classified as failing, not silently ignored."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([
                {"name": "deploy", "state": "COMPLETED", "conclusion": "ERROR"},
            ]),
            stderr="",
        )
        result = manager._fetch_ci_status("org/repo", "10")
        assert result.status == CIStatus.FAILING
        assert "deploy" in result.failed_checks

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_startup_failure_conclusion_treated_as_failure(self, mock_run, manager):
        """conclusion=STARTUP_FAILURE must be classified as failing."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps([
                {"name": "build", "state": "COMPLETED", "conclusion": "STARTUP_FAILURE"},
            ]),
            stderr="",
        )
        result = manager._fetch_ci_status("org/repo", "10")
        assert result.status == CIStatus.FAILING
        assert "build" in result.failed_checks


# ===========================================================================
# CI fix task creation
# ===========================================================================

class TestCIFixTaskCreation:
    def test_deterministic_task_id(self, manager):
        task = _make_task(pr_url="https://github.com/org/repo/pull/10")
        ci_result = CICheckResult(
            status=CIStatus.FAILING,
            failed_checks=["tests"],
            failure_logs="Error: test_foo failed",
        )
        manager._create_ci_fix_task(task, ci_result, 1, "engineer")

        pushed = manager._queue.push.call_args
        fix_task = pushed[0][0]
        assert fix_task.id == f"ci-fix-{task.id[:12]}-c1"
        assert fix_task.type == TaskType.FIX
        assert fix_task.status == TaskStatus.PENDING
        assert pushed[0][1] == "engineer"

    def test_failure_logs_in_description(self, manager):
        task = _make_task(pr_url="https://github.com/org/repo/pull/10")
        ci_result = CICheckResult(
            status=CIStatus.FAILING,
            failed_checks=["lint"],
            failure_logs="src/foo.py:42: E501 line too long",
        )
        manager._create_ci_fix_task(task, ci_result, 2, "engineer")

        fix_task = manager._queue.push.call_args[0][0]
        assert "E501" in fix_task.description
        assert "attempt 2" in fix_task.description

    def test_context_propagation(self, manager):
        task = _make_task(
            pr_url="https://github.com/org/repo/pull/10",
            implementation_branch="feature/abc",
            jira_key="PLUTO-42",
        )
        ci_result = CICheckResult(
            status=CIStatus.FAILING, failed_checks=["build"], failure_logs=""
        )
        manager._create_ci_fix_task(task, ci_result, 1, "engineer")

        fix_ctx = manager._queue.push.call_args[0][0].context
        assert fix_ctx["github_repo"] == "org/repo"
        assert fix_ctx["pr_url"] == "https://github.com/org/repo/pull/10"
        assert fix_ctx["implementation_branch"] == "feature/abc"
        assert fix_ctx["jira_key"] == "PLUTO-42"
        assert fix_ctx["ci_fix_count"] == 1


# ===========================================================================
# Escalation
# ===========================================================================

class TestEscalation:
    def test_escalation_queued_to_architect(self, manager):
        task = _make_task(pr_url="https://github.com/org/repo/pull/10")
        ci_result = CICheckResult(
            status=CIStatus.FAILING,
            failed_checks=["tests"],
            failure_logs="persistent failure",
        )
        manager._escalate_ci_failure(task, ci_result, 3, "engineer")

        pushed = manager._queue.push.call_args
        esc_task = pushed[0][0]
        assert esc_task.type == TaskType.ESCALATION
        assert pushed[0][1] == "architect"
        assert esc_task.id == f"ci-escalation-{task.id[:12]}"
        assert "3 fix attempts" in esc_task.description


# ===========================================================================
# Merge conflict detection
# ===========================================================================

class TestMergeConflicts:
    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_detects_conflicting(self, mock_run, manager):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"mergeable": "CONFLICTING"}),
            stderr="",
        )
        assert manager._has_merge_conflicts("org/repo", "10") is True

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_no_conflict(self, mock_run, manager):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"mergeable": "MERGEABLE"}),
            stderr="",
        )
        assert manager._has_merge_conflicts("org/repo", "10") is False

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_command_failure_returns_false(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="err")
        assert manager._has_merge_conflicts("org/repo", "10") is False


# ===========================================================================
# Rebase
# ===========================================================================

class TestRebase:
    @patch("agent_framework.core.pr_lifecycle.run_git_command")
    def test_successful_rebase(self, mock_git, manager):
        mock_git.return_value = MagicMock(returncode=0, stdout="", stderr="")
        task = _make_task(
            pr_url="https://github.com/org/repo/pull/10",
            implementation_branch="feature/abc",
        )
        assert manager._rebase_on_main(task, "org/repo", "10") is True
        # Verify sequence: fetch, checkout, rebase, push
        calls = [c[0][0] for c in mock_git.call_args_list]
        assert calls[0] == ["fetch", "origin", "main"]
        assert calls[1] == ["checkout", "feature/abc"]
        assert calls[2] == ["rebase", "origin/main"]
        assert calls[3] == ["push", "--force-with-lease"]

    @patch("agent_framework.core.pr_lifecycle.run_git_command")
    def test_rebase_failure_aborts(self, mock_git, manager):
        def side_effect(args, **kwargs):
            if args[0] == "rebase" and args[1] == "origin/main":
                return MagicMock(returncode=1, stdout="", stderr="conflict")
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_git.side_effect = side_effect
        task = _make_task(
            pr_url="https://github.com/org/repo/pull/10",
            implementation_branch="feature/abc",
        )
        assert manager._rebase_on_main(task, "org/repo", "10") is False
        # Should have called rebase --abort
        abort_calls = [
            c for c in mock_git.call_args_list
            if c[0][0] == ["rebase", "--abort"]
        ]
        assert len(abort_calls) == 1

    def test_no_branch_returns_false(self, manager):
        task = _make_task(pr_url="https://github.com/org/repo/pull/10")
        assert manager._rebase_on_main(task, "org/repo", "10") is False


# ===========================================================================
# PR approval
# ===========================================================================

class TestApproval:
    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_approve_calls_gh(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=0)
        manager._approve_pr("org/repo", "10")
        cmd = mock_run.call_args[0][0]
        assert cmd[:3] == ["gh", "pr", "review"]
        assert "--approve" in cmd

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_approve_failure_logged(self, mock_run, manager):
        from agent_framework.utils.subprocess_utils import SubprocessError
        mock_run.side_effect = SubprocessError("cmd", 1, "error")
        manager._approve_pr("org/repo", "10")
        manager._log.warning.assert_called()


# ===========================================================================
# Merge
# ===========================================================================

class TestMergePR:
    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_successful_merge(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        assert manager._merge_pr("org/repo", "10") is True
        cmd = mock_run.call_args[0][0]
        assert "--squash" in cmd
        assert "--auto" in cmd
        assert "--delete-branch" in cmd

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_merge_with_rebase_strategy(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        assert manager._merge_pr("org/repo", "10", strategy="rebase") is True
        cmd = mock_run.call_args[0][0]
        assert "--rebase" in cmd
        assert "--squash" not in cmd

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_merge_failure(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="merge conflict")
        assert manager._merge_pr("org/repo", "10") is False

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_delete_branch_flag_respected(self, mock_run):
        mgr = _make_manager(pr_lifecycle_config={
            "ci_poll_interval": 0,
            "ci_poll_max_wait": 1,
            "delete_branch_on_merge": False,
        })
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mgr._merge_pr("org/repo", "10")
        cmd = mock_run.call_args[0][0]
        assert "--delete-branch" not in cmd


# ===========================================================================
# Full manage() flow
# ===========================================================================

class TestManageFlow:
    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_passing_ci_merges(self, mock_run, manager):
        """CI passing + no conflicts → approve + merge."""
        task = _make_task(pr_url="https://github.com/org/repo/pull/10")

        # Respond to different gh commands
        def route_command(cmd, **kwargs):
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "pr checks" in cmd_str:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps([
                        {"name": "build", "state": "COMPLETED", "conclusion": "SUCCESS"},
                    ]),
                    stderr="",
                )
            if "pr view" in cmd_str:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps({"mergeable": "MERGEABLE"}),
                    stderr="",
                )
            # approve and merge
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = route_command
        assert manager.manage(task, "engineer") is True

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_failing_ci_queues_fix(self, mock_run, manager):
        """CI failing → creates fix task, returns False."""
        task = _make_task(pr_url="https://github.com/org/repo/pull/10")

        def route_command(cmd, **kwargs):
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "pr checks" in cmd_str and "--json" in cmd_str:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps([
                        {"name": "tests", "state": "COMPLETED", "conclusion": "FAILURE"},
                    ]),
                    stderr="",
                )
            # Failure log fetch
            return MagicMock(returncode=0, stdout="test_foo FAILED", stderr="")

        mock_run.side_effect = route_command
        assert manager.manage(task, "engineer") is False
        manager._queue.push.assert_called_once()
        fix_task = manager._queue.push.call_args[0][0]
        assert fix_task.type == TaskType.FIX

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_exhausted_ci_fixes_escalates(self, mock_run, manager):
        """After max_ci_fix_attempts, escalate instead of fix."""
        task = _make_task(
            pr_url="https://github.com/org/repo/pull/10",
            ci_fix_count=3,
        )

        def route_command(cmd, **kwargs):
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "pr checks" in cmd_str and "--json" in cmd_str:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps([
                        {"name": "tests", "state": "COMPLETED", "conclusion": "FAILURE"},
                    ]),
                    stderr="",
                )
            return MagicMock(returncode=0, stdout="FAILED", stderr="")

        mock_run.side_effect = route_command
        assert manager.manage(task, "engineer") is False
        esc_task = manager._queue.push.call_args[0][0]
        assert esc_task.type == TaskType.ESCALATION

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_check_error_conclusion_queues_fix(self, mock_run, manager):
        """conclusion=ERROR on a check (→ CIStatus.FAILING) queues a fix task."""
        task = _make_task(pr_url="https://github.com/org/repo/pull/10")

        def route_command(cmd, **kwargs):
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "pr checks" in cmd_str and "--json" in cmd_str:
                return MagicMock(
                    returncode=1,
                    stdout=json.dumps([
                        {"name": "tests", "state": "COMPLETED", "conclusion": "ERROR"},
                    ]),
                    stderr="Some checks were not successful",
                )
            return MagicMock(returncode=0, stdout="error details", stderr="")

        mock_run.side_effect = route_command
        assert manager.manage(task, "engineer") is False
        manager._queue.push.assert_called_once()
        fix_task = manager._queue.push.call_args[0][0]
        assert fix_task.type == TaskType.FIX

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_check_error_conclusion_exhausted_fixes_escalates(self, mock_run, manager):
        """conclusion=ERROR on a check (→ CIStatus.FAILING) escalates when fixes exhausted."""
        task = _make_task(
            pr_url="https://github.com/org/repo/pull/10",
            ci_fix_count=3,
        )

        def route_command(cmd, **kwargs):
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "pr checks" in cmd_str and "--json" in cmd_str:
                return MagicMock(
                    returncode=1,
                    stdout=json.dumps([
                        {"name": "tests", "state": "COMPLETED", "conclusion": "ERROR"},
                    ]),
                    stderr="Some checks were not successful",
                )
            return MagicMock(returncode=0, stdout="error details", stderr="")

        mock_run.side_effect = route_command
        assert manager.manage(task, "engineer") is False
        esc_task = manager._queue.push.call_args[0][0]
        assert esc_task.type == TaskType.ESCALATION

    def test_infrastructure_error_ci_leaves_pr_for_manual_review(self, manager):
        """CIStatus.ERROR (gh CLI failure) leaves PR for manual review — no fix task queued."""
        task = _make_task(pr_url="https://github.com/org/repo/pull/10")
        manager._poll_ci_checks = MagicMock(return_value=CICheckResult(
            status=CIStatus.ERROR, failed_checks=[], failure_logs="gh: authentication failed"
        ))
        assert manager.manage(task, "engineer") is False
        manager._queue.push.assert_not_called()
        manager._log.warning.assert_called()

    def test_infrastructure_error_ci_exhausted_fixes_no_task(self, manager):
        """CIStatus.ERROR with exhausted fixes still just leaves PR — no escalation."""
        task = _make_task(
            pr_url="https://github.com/org/repo/pull/10",
            ci_fix_count=3,
        )
        manager._poll_ci_checks = MagicMock(return_value=CICheckResult(
            status=CIStatus.ERROR, failed_checks=[], failure_logs="gh: authentication failed"
        ))
        assert manager.manage(task, "engineer") is False
        manager._queue.push.assert_not_called()
        manager._log.warning.assert_called()

    def test_failing_with_empty_checks_leaves_pr_for_manual_review(self, manager):
        """FAILING + empty failed_checks → no fix task, left for manual review."""
        task = _make_task(pr_url="https://github.com/org/repo/pull/10")
        manager._poll_ci_checks = MagicMock(return_value=CICheckResult(
            status=CIStatus.FAILING, failed_checks=[], failure_logs=""
        ))
        assert manager.manage(task, "engineer") is False
        manager._queue.push.assert_not_called()
        manager._log.warning.assert_called()


# ===========================================================================
# Agent integration
# ===========================================================================

class TestAgentIntegration:
    def test_lifecycle_called_in_post_completion(self, agent):
        """_manage_pr_lifecycle is called during post-completion for non-subtasks."""
        mock_manager = MagicMock()
        mock_manager.should_manage.return_value = True
        mock_manager.manage.return_value = True
        agent._git_ops._pr_lifecycle_manager = mock_manager

        task = _make_task(pr_url="https://github.com/org/repo/pull/10")
        agent._git_ops._sync_jira_status = MagicMock()
        agent._git_ops.manage_pr_lifecycle(task)

        mock_manager.should_manage.assert_called_once_with(task)
        mock_manager.manage.assert_called_once_with(task, "engineer")
        # JIRA should be updated to Done after successful merge
        agent._git_ops._sync_jira_status.assert_called_once()

    def test_lifecycle_skipped_without_manager(self, agent):
        """No-op when _pr_lifecycle_manager is None."""
        task = _make_task(pr_url="https://github.com/org/repo/pull/10")
        # Should not raise
        agent._git_ops.manage_pr_lifecycle(task)

    def test_lifecycle_skipped_when_should_manage_false(self, agent):
        """No-op when should_manage returns False."""
        mock_manager = MagicMock()
        mock_manager.should_manage.return_value = False
        agent._git_ops._pr_lifecycle_manager = mock_manager

        task = _make_task()
        agent._git_ops.manage_pr_lifecycle(task)

        mock_manager.manage.assert_not_called()

    def test_lifecycle_error_does_not_propagate(self, agent):
        """Errors in PR lifecycle are caught, don't crash the agent."""
        mock_manager = MagicMock()
        mock_manager.should_manage.return_value = True
        mock_manager.manage.side_effect = RuntimeError("boom")
        agent._git_ops._pr_lifecycle_manager = mock_manager

        task = _make_task(pr_url="https://github.com/org/repo/pull/10")
        # Should not raise
        agent._git_ops.manage_pr_lifecycle(task)
        agent.logger.error.assert_called()

    def test_jira_not_updated_when_not_merged(self, agent):
        """JIRA status NOT updated when merge returns False."""
        mock_manager = MagicMock()
        mock_manager.should_manage.return_value = True
        mock_manager.manage.return_value = False
        agent._git_ops._pr_lifecycle_manager = mock_manager

        agent._git_ops._sync_jira_status = MagicMock()
        task = _make_task(pr_url="https://github.com/org/repo/pull/10")
        agent._git_ops.manage_pr_lifecycle(task)

        agent._git_ops._sync_jira_status.assert_not_called()


# ===========================================================================
# Strategy validation
# ===========================================================================

class TestStrategyValidation:
    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_invalid_strategy_returns_false(self, mock_run, manager):
        """Invalid merge strategy rejects without calling gh."""
        assert manager._merge_pr("org/repo", "10", strategy="yolo") is False
        mock_run.assert_not_called()
        manager._log.error.assert_called()

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_valid_strategies_accepted(self, mock_run, manager):
        """All three valid strategies are accepted."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        for strategy in ("squash", "merge", "rebase"):
            assert manager._merge_pr("org/repo", "10", strategy=strategy) is True


# ===========================================================================
# Repo cwd resolution
# ===========================================================================

class TestResolveRepoCwd:
    def test_uses_multi_repo_manager_when_available(self):
        mock_mrm = MagicMock()
        mock_mrm.ensure_repo.return_value = "/repos/org/repo"
        mgr = _make_manager(multi_repo_manager=mock_mrm)
        cwd = mgr._resolve_repo_cwd("org/repo")
        assert str(cwd) == "/repos/org/repo"
        mock_mrm.ensure_repo.assert_called_once_with("org/repo")

    def test_falls_back_to_workspace(self):
        mgr = _make_manager()
        cwd = mgr._resolve_repo_cwd("org/repo")
        assert str(cwd) == "/tmp/ws"

    def test_falls_back_on_ensure_repo_error(self):
        mock_mrm = MagicMock()
        mock_mrm.ensure_repo.side_effect = RuntimeError("clone failed")
        mgr = _make_manager(multi_repo_manager=mock_mrm)
        cwd = mgr._resolve_repo_cwd("org/repo")
        assert str(cwd) == "/tmp/ws"

    @patch("agent_framework.core.pr_lifecycle.run_git_command")
    def test_rebase_uses_repo_cwd(self, mock_git):
        """Rebase operations should run in the repo clone, not the framework dir."""
        mock_mrm = MagicMock()
        mock_mrm.ensure_repo.return_value = "/repos/org/repo"
        mgr = _make_manager(multi_repo_manager=mock_mrm)
        mock_git.return_value = MagicMock(returncode=0, stdout="", stderr="")
        task = _make_task(
            pr_url="https://github.com/org/repo/pull/10",
            implementation_branch="feature/abc",
        )
        mgr._rebase_on_main(task, "org/repo", "10")
        # All git commands should use the repo clone path
        for c in mock_git.call_args_list:
            assert c[1].get("cwd") == Path("/repos/org/repo")


# ===========================================================================
# CI fix reentry path
# ===========================================================================

class TestCIFixReentry:
    def test_fix_task_carries_ci_fix_count(self, manager):
        """CI fix task propagates ci_fix_count so reentry increments correctly."""
        task = _make_task(pr_url="https://github.com/org/repo/pull/10")
        ci_result = CICheckResult(
            status=CIStatus.FAILING, failed_checks=["tests"], failure_logs=""
        )
        manager._create_ci_fix_task(task, ci_result, 2, "engineer")
        fix_task = manager._queue.push.call_args[0][0]
        # When the fix task completes and re-enters manage(), this count is read
        assert fix_task.context["ci_fix_count"] == 2
        assert fix_task.context["pr_url"] == "https://github.com/org/repo/pull/10"

    @patch("agent_framework.core.pr_lifecycle.run_command")
    def test_reentry_with_elevated_count_escalates(self, mock_run, manager):
        """Fix task at max count triggers escalation on next CI failure."""
        task = _make_task(
            pr_url="https://github.com/org/repo/pull/10",
            ci_fix_count=3,  # Already at max
        )
        def route_command(cmd, **kwargs):
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
            if "pr checks" in cmd_str and "--json" in cmd_str:
                return MagicMock(
                    returncode=0,
                    stdout=json.dumps([
                        {"name": "tests", "state": "COMPLETED", "conclusion": "FAILURE"},
                    ]),
                    stderr="",
                )
            return MagicMock(returncode=0, stdout="FAILED", stderr="")

        mock_run.side_effect = route_command
        manager.manage(task, "engineer")
        queued = manager._queue.push.call_args[0][0]
        assert queued.type == TaskType.ESCALATION


# ===========================================================================
# Helpers
# ===========================================================================

class TestHelpers:
    def test_extract_pr_number(self):
        assert PRLifecycleManager._extract_pr_number(
            "https://github.com/org/repo/pull/42"
        ) == "42"

    def test_extract_pr_number_trailing_slash(self):
        assert PRLifecycleManager._extract_pr_number(
            "https://github.com/org/repo/pull/99/"
        ) == "99"
