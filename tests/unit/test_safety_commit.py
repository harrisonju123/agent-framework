"""Tests for safety_commit() and its integration points in Agent."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_framework.core.agent import Agent
from agent_framework.core.git_operations import GitOperationsManager
from agent_framework.core.llm_executor import LLMExecutionManager
from agent_framework.core.post_completion import PostCompletionManager
from agent_framework.core.task import Task, TaskStatus, TaskType

_PATCH_RUN_GIT = "agent_framework.utils.subprocess_utils.run_git_command"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def git_ops(mock_logger, tmp_path):
    return GitOperationsManager(
        config=MagicMock(id="engineer", base_id="engineer"),
        workspace=tmp_path,
        queue=MagicMock(queue_dir=Path("/mock/queue"), completed_dir=Path("/mock/queue/completed")),
        logger=mock_logger,
    )


@pytest.fixture
def task():
    return Task(
        id="sc-1",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=1,
        created_by="test",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Test task",
        description="Test",
        context={"github_repo": "org/repo", "workflow_step": "implement"},
    )


# ---------------------------------------------------------------------------
# Core safety_commit() tests
# ---------------------------------------------------------------------------

class TestSafetyCommit:

    def test_with_changes(self, git_ops, tmp_path):
        """Dirty working tree -> add + commit, returns True."""
        with patch(_PATCH_RUN_GIT) as mock_git:
            mock_git.return_value = MagicMock(returncode=0, stdout=" M file.py\n", stderr="")
            result = git_ops.safety_commit(tmp_path, "checkpoint")

        assert result is True
        assert mock_git.call_count == 3  # status, add, commit
        commit_args = mock_git.call_args_list[2][0][0]
        assert commit_args[0] == "commit"
        assert "[auto-commit]" in commit_args[2]
        assert "checkpoint" in commit_args[2]

    def test_clean_tree(self, git_ops, tmp_path):
        """Empty porcelain status -> no-op, returns False."""
        with patch(_PATCH_RUN_GIT) as mock_git:
            mock_git.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = git_ops.safety_commit(tmp_path, "checkpoint")

        assert result is False
        assert mock_git.call_count == 1  # only status

    def test_nonexistent_dir(self, git_ops):
        """Path doesn't exist -> returns False, no git calls."""
        with patch(_PATCH_RUN_GIT) as mock_git:
            result = git_ops.safety_commit(Path("/nonexistent/path"), "checkpoint")

        assert result is False
        mock_git.assert_not_called()

    def test_git_error_swallowed(self, git_ops, tmp_path):
        """run_git_command raises -> returns False, no exception propagated."""
        with patch(_PATCH_RUN_GIT) as mock_git:
            mock_git.side_effect = RuntimeError("git broke")
            result = git_ops.safety_commit(tmp_path, "checkpoint")

        assert result is False

    def test_message_format(self, git_ops, tmp_path):
        """Commit message uses [auto-commit] prefix."""
        with patch(_PATCH_RUN_GIT) as mock_git:
            mock_git.return_value = MagicMock(returncode=0, stdout=" M f.py\n", stderr="")
            git_ops.safety_commit(tmp_path, "my reason")

        commit_call = mock_git.call_args_list[2]
        msg = commit_call[0][0][2]
        assert msg == "[auto-commit] my reason"

    def test_status_command_fails(self, git_ops, tmp_path):
        """git status returns non-zero -> returns False."""
        with patch(_PATCH_RUN_GIT) as mock_git:
            mock_git.return_value = MagicMock(returncode=128, stdout="", stderr="fatal")
            result = git_ops.safety_commit(tmp_path, "checkpoint")

        assert result is False
        assert mock_git.call_count == 1

    def test_add_fails(self, git_ops, tmp_path):
        """git add fails -> returns False, no commit attempted."""
        with patch(_PATCH_RUN_GIT) as mock_git:
            status_result = MagicMock(returncode=0, stdout=" M f.py\n", stderr="")
            add_result = MagicMock(returncode=1, stdout="", stderr="add failed")
            mock_git.side_effect = [status_result, add_result]
            result = git_ops.safety_commit(tmp_path, "checkpoint")

        assert result is False
        assert mock_git.call_count == 2  # status + add, no commit


# ---------------------------------------------------------------------------
# _auto_commit_wip delegation test
# ---------------------------------------------------------------------------

class TestAutoCommitWipDelegation:

    @pytest.mark.asyncio
    async def test_delegates_to_llm_executor(self):
        """_auto_commit_wip delegates to LLMExecutionManager.auto_commit_wip."""
        a = MagicMock()
        a._auto_commit_wip = Agent._auto_commit_wip.__get__(a)
        a._llm_executor = MagicMock()
        a._llm_executor.auto_commit_wip = AsyncMock()

        t = Task(
            id="wip-1", type=TaskType.IMPLEMENTATION, status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="eng",
            created_at=datetime.now(timezone.utc), title="T", description="D",
            context={},
        )
        wd = Path("/tmp/work")
        await a._auto_commit_wip(t, wd, 15)

        a._llm_executor.auto_commit_wip.assert_awaited_once_with(t, wd, 15)

    @pytest.mark.asyncio
    async def test_auto_commit_wip_calls_safety_commit(self):
        """LLMExecutionManager.auto_commit_wip calls git_ops.safety_commit."""
        git_ops_mock = MagicMock()
        git_ops_mock.safety_commit.return_value = True
        session_logger = MagicMock()

        mgr = LLMExecutionManager(
            config=MagicMock(id="eng", base_id="engineer"),
            llm=MagicMock(),
            git_ops=git_ops_mock,
            logger=MagicMock(),
            session_logger=session_logger,
            activity_manager=MagicMock(),
        )

        t = Task(
            id="wip-1", type=TaskType.IMPLEMENTATION, status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="eng",
            created_at=datetime.now(timezone.utc), title="T", description="D",
            context={},
        )
        wd = Path("/tmp/work")
        await mgr.auto_commit_wip(t, wd, 15)

        git_ops_mock.safety_commit.assert_called_once_with(
            wd, "WIP: auto-save before circuit breaker (15 consecutive Bash calls)"
        )
        session_logger.log.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_session_log_when_nothing_committed(self):
        """Session log is NOT written when safety_commit returns False."""
        git_ops_mock = MagicMock()
        git_ops_mock.safety_commit.return_value = False
        session_logger = MagicMock()

        mgr = LLMExecutionManager(
            config=MagicMock(id="eng", base_id="engineer"),
            llm=MagicMock(),
            git_ops=git_ops_mock,
            logger=MagicMock(),
            session_logger=session_logger,
            activity_manager=MagicMock(),
        )

        t = Task(
            id="wip-2", type=TaskType.IMPLEMENTATION, status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="eng",
            created_at=datetime.now(timezone.utc), title="T", description="D",
            context={},
        )
        await mgr.auto_commit_wip(t, Path("/tmp/work"), 10)

        session_logger.log.assert_not_called()


# ---------------------------------------------------------------------------
# Periodic checkpoint tests
# ---------------------------------------------------------------------------

def _make_slow_llm(on_tool_activity_ref):
    """Mock LLM whose complete() captures on_tool_activity then hangs."""
    llm = MagicMock()

    async def _complete(*args, **kwargs):
        on_tool_activity_ref.append(kwargs.get("on_tool_activity"))
        await asyncio.sleep(999)

    llm.complete = _complete
    llm.cancel = MagicMock()
    llm.get_partial_output = MagicMock(return_value="")
    return llm


async def _setup_and_get_callback(mgr, task, working_dir=None):
    """Start LLM execution via LLMExecutionManager and return (result_task, on_tool_activity callback)."""
    cb_ref = []
    mgr.llm = _make_slow_llm(cb_ref)

    async def _never_interrupt():
        await asyncio.sleep(999)

    # Use a MagicMock as working_dir by default -- .exists() returns truthy MagicMock
    wd = working_dir if working_dir is not None else MagicMock()

    async def _run():
        return await mgr.execute(
            task, "prompt", wd, None,
            watch_for_interruption_coro=_never_interrupt,
            max_consecutive_tool_calls=999,
            max_consecutive_diagnostic_calls=999,
            exploration_alert_threshold=999,
        )

    result_task = asyncio.create_task(_run())

    for _ in range(50):
        if cb_ref:
            break
        await asyncio.sleep(0.01)
    assert cb_ref, "on_tool_activity callback not captured"

    return result_task, cb_ref[0]


@pytest.fixture
def checkpoint_mgr():
    """LLMExecutionManager wired for periodic checkpoint tests."""
    git_ops_mock = MagicMock()
    git_ops_mock.safety_commit.return_value = False
    git_ops_mock.worktree_env_vars = None
    git_ops_mock.active_worktree = None

    config = MagicMock()
    config.id = "test-agent"
    config.base_id = "engineer"

    mgr = LLMExecutionManager(
        config=config,
        llm=MagicMock(),
        git_ops=git_ops_mock,
        logger=MagicMock(),
        session_logger=MagicMock(),
        activity_manager=MagicMock(),
    )
    return mgr


class TestPeriodicCheckpoint:

    @pytest.mark.asyncio
    async def test_fires_at_interval(self, checkpoint_mgr, task):
        """safety_commit called at tool call #25 but not at #1 or #24."""
        result_task, on_tool = await _setup_and_get_callback(checkpoint_mgr, task)

        # Fire 24 diverse tool calls (no checkpoint expected)
        for i in range(24):
            on_tool("Read" if i % 2 == 0 else "Write", f"file_{i}")

        checkpoint_mgr.git_ops.safety_commit.assert_not_called()

        # Tool call #25 triggers checkpoint
        on_tool("Read", "file_25")

        checkpoint_mgr.git_ops.safety_commit.assert_called_once()
        call_args = checkpoint_mgr.git_ops.safety_commit.call_args
        assert "periodic checkpoint" in call_args[0][1]
        assert "tool call 25" in call_args[0][1]

        # Cleanup
        result_task.cancel()
        try:
            await result_task
        except (asyncio.CancelledError, Exception):
            pass

    @pytest.mark.asyncio
    async def test_fires_again_at_50(self, checkpoint_mgr, task):
        """safety_commit called again at tool call #50."""
        result_task, on_tool = await _setup_and_get_callback(checkpoint_mgr, task)

        for i in range(50):
            on_tool("Read" if i % 2 == 0 else "Write", f"file_{i}")

        assert checkpoint_mgr.git_ops.safety_commit.call_count == 2

        result_task.cancel()
        try:
            await result_task
        except (asyncio.CancelledError, Exception):
            pass

    @pytest.mark.asyncio
    async def test_skips_missing_dir(self, checkpoint_mgr):
        """No crash if working_dir doesn't exist at checkpoint time."""
        missing_task = Task(
            id="sc-miss", type=TaskType.IMPLEMENTATION, status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="eng",
            created_at=datetime.now(timezone.utc), title="T", description="D",
            context={"github_repo": "org/repo"},
        )

        result_task, on_tool = await _setup_and_get_callback(
            checkpoint_mgr, missing_task, working_dir=Path("/nonexistent/dir")
        )

        # Fire 25 tool calls -- should not crash
        for i in range(25):
            on_tool("Read", f"file_{i}")

        # safety_commit should NOT have been called (dir doesn't exist)
        checkpoint_mgr.git_ops.safety_commit.assert_not_called()

        result_task.cancel()
        try:
            await result_task
        except (asyncio.CancelledError, Exception):
            pass


# ---------------------------------------------------------------------------
# Exit-path wiring tests
# ---------------------------------------------------------------------------

class TestPostCompletionCommit:

    @pytest.mark.asyncio
    async def test_commits_for_implementation_step(self, tmp_path):
        """safety_commit is called before mark_completed for implementation steps."""
        a = MagicMock()
        a._handle_successful_response = Agent._handle_successful_response.__get__(a)
        a._is_implementation_step = MagicMock(return_value=True)
        git_ops_mock = MagicMock()
        git_ops_mock.safety_commit.return_value = True
        a._git_ops = git_ops_mock
        a._optimization_config = {}
        a._run_sandbox_tests = AsyncMock(return_value=None)
        a._self_eval_enabled = False
        a._error_recovery = MagicMock()
        a._error_recovery.has_deliverables.return_value = True
        config = MagicMock()
        config.id = "eng"
        config.base_id = "engineer"
        a.config = config
        a._agent_definition = None
        a.workspace = tmp_path
        a.logger = MagicMock()
        queue = MagicMock()
        a.queue = queue
        a.activity_manager = MagicMock()
        a._session_logger = MagicMock()
        a._context_window_manager = None
        a._sync_jira_status = MagicMock()

        # Wire up real PostCompletionManager so safety_commit is called internally
        a._post_completion = PostCompletionManager(
            config=config,
            queue=queue,
            workspace=tmp_path,
            logger=a.logger,
            session_logger=a._session_logger,
            activity_manager=a.activity_manager,
            review_cycle=MagicMock(),
            workflow_router=MagicMock(),
            git_ops=git_ops_mock,
            budget=MagicMock(),
            error_recovery=MagicMock(),
            optimization_config={},
            session_logging_enabled=False,
            session_logs_dir=tmp_path / "logs",
        )
        a._analytics = MagicMock()
        a._analytics.extract_summary = AsyncMock(return_value=None)
        a._analytics.extract_and_store_memories = MagicMock()
        a._analytics.analyze_tool_patterns = MagicMock()

        t = Task(
            id="post-1", type=TaskType.IMPLEMENTATION, status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="eng",
            created_at=datetime.now(timezone.utc), title="T", description="D",
            context={"workflow_step": "implement"},
        )
        resp = MagicMock(content="done", model_used="test")

        await a._handle_successful_response(t, resp, datetime.now(timezone.utc), working_dir=tmp_path)

        git_ops_mock.safety_commit.assert_called_once()
        assert "task completion" in git_ops_mock.safety_commit.call_args[0][1]

    @pytest.mark.asyncio
    async def test_skips_non_implementation_step(self, tmp_path):
        """safety_commit NOT called for non-implementation steps (e.g., plan)."""
        a = MagicMock()
        a._handle_successful_response = Agent._handle_successful_response.__get__(a)
        a._is_implementation_step = MagicMock(return_value=False)
        git_ops_mock = MagicMock()
        a._git_ops = git_ops_mock
        a._optimization_config = {}
        a._run_sandbox_tests = AsyncMock(return_value=None)
        a._self_eval_enabled = False
        config = MagicMock()
        config.id = "arch"
        config.base_id = "architect"
        a.config = config
        a._agent_definition = None
        a.workspace = tmp_path
        a.logger = MagicMock()
        queue = MagicMock()
        a.queue = queue
        a.activity_manager = MagicMock()
        a._session_logger = MagicMock()
        a._context_window_manager = None
        a._sync_jira_status = MagicMock()

        # Wire up real PostCompletionManager
        a._post_completion = PostCompletionManager(
            config=config,
            queue=queue,
            workspace=tmp_path,
            logger=a.logger,
            session_logger=a._session_logger,
            activity_manager=a.activity_manager,
            review_cycle=MagicMock(),
            workflow_router=MagicMock(),
            git_ops=git_ops_mock,
            budget=MagicMock(),
            error_recovery=MagicMock(),
            optimization_config={},
            session_logging_enabled=False,
            session_logs_dir=tmp_path / "logs",
        )
        a._analytics = MagicMock()
        a._analytics.extract_summary = AsyncMock(return_value=None)
        a._analytics.extract_and_store_memories = MagicMock()
        a._analytics.analyze_tool_patterns = MagicMock()

        t = Task(
            id="post-2", type="planning", status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="arch",
            created_at=datetime.now(timezone.utc), title="T", description="D",
            context={"workflow_step": "plan"},
        )
        resp = MagicMock(content="plan output", model_used="test")

        await a._handle_successful_response(t, resp, datetime.now(timezone.utc), working_dir=tmp_path)

        git_ops_mock.safety_commit.assert_not_called()


class TestInterruptionPathCommit:

    @pytest.mark.asyncio
    @patch("agent_framework.core.attempt_tracker.record_attempt", return_value=None)
    async def test_commits_before_reset(self, mock_record, task, tmp_path):
        """_finalize_failed_attempt is called in the interruption path before reset_to_pending."""
        # Build a real LLMExecutionManager with a mock LLM that hangs
        finalize_mock = MagicMock()

        git_ops_mock = MagicMock()
        git_ops_mock.safety_commit.return_value = True
        git_ops_mock.worktree_env_vars = None
        git_ops_mock.active_worktree = None

        config = MagicMock()
        config.id = "test-agent"
        config.base_id = "engineer"

        llm = MagicMock()

        async def _slow_complete(*args, **kwargs):
            await asyncio.sleep(999)

        llm.complete = _slow_complete
        llm.cancel = MagicMock()
        llm.get_partial_output = MagicMock(return_value="")

        mgr = LLMExecutionManager(
            config=config,
            llm=llm,
            git_ops=git_ops_mock,
            logger=MagicMock(),
            session_logger=MagicMock(),
            activity_manager=MagicMock(),
        )

        # Make the watcher return immediately (simulate interruption)
        async def _interrupt_now():
            return

        result = await mgr.execute(
            task, "prompt", tmp_path, None,
            watch_for_interruption_coro=_interrupt_now,
            finalize_failed_attempt_cb=finalize_mock,
        )

        assert result is None  # Interrupted
        # finalize_failed_attempt_cb is called with partial output
        finalize_mock.assert_called_once()
        call_args = finalize_mock.call_args
        assert call_args[0][0] is task
        assert call_args[0][1] == tmp_path


class TestFailurePathCommit:

    @pytest.mark.asyncio
    @patch("agent_framework.core.attempt_tracker.record_attempt", return_value=None)
    async def test_commits_before_retry(self, mock_record, tmp_path):
        """_finalize_failed_attempt is called in _handle_failed_response to preserve work."""
        from agent_framework.core.error_recovery import ErrorRecoveryManager

        a = MagicMock()
        a.workspace = tmp_path
        a._git_ops = MagicMock()
        a._git_ops.safety_commit.return_value = True
        a._handle_failure = AsyncMock()
        a._budget = MagicMock()
        a._budget.estimate_cost.return_value = 0.0
        a._model_success_store = None
        a.logger = MagicMock()
        a._session_logger = MagicMock()
        a.activity_manager = MagicMock()
        a.config = MagicMock()
        a.config.id = "eng"
        a.config.base_id = "eng"
        a._review_cycle = MagicMock()
        a._context_window_manager = None

        er = ErrorRecoveryManager(
            config=a.config, queue=MagicMock(), llm=MagicMock(),
            logger=a.logger, session_logger=a._session_logger,
            retry_handler=MagicMock(), escalation_handler=MagicMock(),
            workspace=a.workspace, budget_manager=a._budget,
            activity_manager=a.activity_manager,
        )
        er.handle_failure = a._handle_failure
        a._error_recovery = er

        a._handle_failed_response = Agent._handle_failed_response.__get__(a)
        a._finalize_failed_attempt = Agent._finalize_failed_attempt.__get__(a)
        a._extract_partial_progress = Agent._extract_partial_progress

        t = Task(
            id="fail-1", type=TaskType.IMPLEMENTATION, status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="eng",
            created_at=datetime.now(timezone.utc), title="T", description="D",
            context={},
        )
        resp = MagicMock(
            content="partial", error="timeout", model_used="test",
            latency_ms=1000, finish_reason="error", success=False,
            reported_cost_usd=0.01,
        )

        # Use tmp_path (real dir) so .exists() returns True
        await a._handle_failed_response(t, resp, working_dir=tmp_path)

        # Work preservation now happens via record_attempt inside _finalize_failed_attempt
        mock_record.assert_called_once()


class TestCleanupPathCommit:

    def test_commits_before_cleanup_worktree(self):
        """safety_commit is called in _cleanup_task_execution before cleanup_worktree."""
        a = MagicMock()
        a._cleanup_task_execution = Agent._cleanup_task_execution.__get__(a)
        a._git_ops = MagicMock()
        a._git_ops.active_worktree = Path("/tmp/worktree")
        a._git_ops.safety_commit.return_value = True
        a.config = MagicMock()
        a.config.id = "eng"
        a.activity_manager = MagicMock()
        a.queue = MagicMock()

        t = Task(
            id="clean-1", type=TaskType.IMPLEMENTATION, status=TaskStatus.COMPLETED,
            priority=1, created_by="test", assigned_to="eng",
            created_at=datetime.now(timezone.utc), title="T", description="D",
            context={},
        )

        a._cleanup_task_execution(t, MagicMock())

        a._git_ops.safety_commit.assert_called_once()
        assert "cleanup" in a._git_ops.safety_commit.call_args[0][1]
        # Verify cleanup_worktree was also called (after safety_commit)
        a._git_ops.cleanup_worktree.assert_called_once()

    def test_skips_when_no_active_worktree(self):
        """safety_commit is NOT called when there's no active worktree."""
        a = MagicMock()
        a._cleanup_task_execution = Agent._cleanup_task_execution.__get__(a)
        a._git_ops = MagicMock()
        a._git_ops.active_worktree = None
        a.config = MagicMock()
        a.config.id = "eng"
        a.activity_manager = MagicMock()
        a.queue = MagicMock()

        t = Task(
            id="clean-2", type=TaskType.IMPLEMENTATION, status=TaskStatus.COMPLETED,
            priority=1, created_by="test", assigned_to="eng",
            created_at=datetime.now(timezone.utc), title="T", description="D",
            context={},
        )

        a._cleanup_task_execution(t, MagicMock())

        a._git_ops.safety_commit.assert_not_called()


class TestExceptionHandlerCommit:

    @pytest.mark.asyncio
    async def test_commits_before_error_recovery(self, tmp_path):
        """safety_commit is called in _handle_task exception handler before retry."""
        a = MagicMock()
        a._handle_task = Agent._handle_task.__get__(a)
        a._git_ops = MagicMock()
        a._git_ops.safety_commit.return_value = True
        a._error_recovery = MagicMock()
        a._error_recovery.gather_git_evidence.return_value = None
        a._handle_failure = AsyncMock()
        a._normalize_workflow = MagicMock()
        a._validate_task_or_reject = MagicMock(return_value=True)
        a._setup_task_context = MagicMock()
        a._setup_context_window_manager_for_task = MagicMock()
        a._initialize_task_execution = MagicMock(side_effect=RuntimeError("boom"))
        a._context_window_manager = None
        a._session_logger = MagicMock()
        a._cleanup_task_execution = MagicMock()
        a.config = MagicMock()
        a.config.id = "eng"
        a.logger = MagicMock()
        a.queue = MagicMock()
        a.activity_manager = MagicMock()

        t = Task(
            id="exc-1", type=TaskType.IMPLEMENTATION, status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="eng",
            created_at=datetime.now(timezone.utc), title="T", description="D",
            context={},
        )

        # Exception fires before working_dir is assigned -- working_dir stays None
        await a._handle_task(t, lock=MagicMock())

        # working_dir is None -> safety_commit should NOT be called (no UnboundLocalError)
        a._git_ops.safety_commit.assert_not_called()
        # _handle_failure should still be called
        a._handle_failure.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("agent_framework.core.attempt_tracker.record_attempt", return_value=None)
    async def test_commits_when_working_dir_exists(self, mock_record, tmp_path):
        """_finalize_failed_attempt is called when exception fires after working_dir is set."""
        from agent_framework.core.error_recovery import ErrorRecoveryManager

        a = MagicMock()
        a._handle_task = Agent._handle_task.__get__(a)
        a._finalize_failed_attempt = Agent._finalize_failed_attempt.__get__(a)
        a._extract_partial_progress = Agent._extract_partial_progress
        a.workspace = tmp_path
        a._git_ops = MagicMock()
        a._git_ops.safety_commit.return_value = True
        a._get_validated_working_directory = MagicMock(return_value=tmp_path)
        a._handle_failure = AsyncMock()
        a._normalize_workflow = MagicMock()
        a._validate_task_or_reject = MagicMock(return_value=True)
        a._setup_task_context = MagicMock()
        a._setup_context_window_manager_for_task = MagicMock()
        a._initialize_task_execution = MagicMock()
        # Fail after working_dir is assigned
        a._try_index_codebase = MagicMock(side_effect=RuntimeError("indexing failed"))
        a._context_window_manager = None
        a._session_logger = MagicMock()
        a._cleanup_task_execution = MagicMock()
        a.config = MagicMock()
        a.config.id = "eng"
        a.config.base_id = "eng"
        a.logger = MagicMock()
        a.queue = MagicMock()
        a.activity_manager = MagicMock()

        er = ErrorRecoveryManager(
            config=a.config, queue=MagicMock(), llm=MagicMock(),
            logger=a.logger, session_logger=a._session_logger,
            retry_handler=MagicMock(), escalation_handler=MagicMock(),
            workspace=a.workspace,
        )
        er.handle_failure = a._handle_failure
        a._error_recovery = er

        t = Task(
            id="exc-2", type=TaskType.IMPLEMENTATION, status=TaskStatus.IN_PROGRESS,
            priority=1, created_by="test", assigned_to="eng",
            created_at=datetime.now(timezone.utc), title="T", description="D",
            context={},
        )

        await a._handle_task(t, lock=MagicMock())

        # Work preservation now happens via record_attempt inside _finalize_failed_attempt
        mock_record.assert_called_once()
