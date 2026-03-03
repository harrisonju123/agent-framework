"""Tests for task_analytics.py — TaskAnalyticsManager."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_framework.core.task_analytics import (
    TaskAnalyticsManager,
    SUMMARY_CONTEXT_MAX_CHARS,
    SUMMARY_MAX_LENGTH,
    _CONVERSATIONAL_PREFIXES,
)
from agent_framework.core.task import Task, TaskStatus, TaskType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(**ctx_overrides):
    ctx = {"github_repo": "org/repo", **ctx_overrides}
    return Task(
        id="task-analytics",
        title="Analytics task",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.COMPLETED,
        context=ctx,
        created_at=datetime.now(timezone.utc),
        created_by="architect",
        assigned_to="engineer",
        description="Analytics task description",
        priority=50,
    )


def _make_manager(**overrides):
    defaults = dict(
        config=MagicMock(id="engineer", base_id="engineer"),
        logger=MagicMock(),
        session_logger=MagicMock(),
        llm=AsyncMock(),
        memory_retriever=MagicMock(),
        tool_pattern_store=MagicMock(),
        optimization_config={},
        memory_enabled=True,
        tool_tips_enabled=True,
        session_logging_enabled=True,
        session_logs_dir=Path("/tmp/logs"),
        workspace=Path("/tmp/workspace"),
        feedback_bus=None,
        code_index_query=None,
    )
    defaults.update(overrides)
    return TaskAnalyticsManager(**defaults)


# ---------------------------------------------------------------------------
# get_repo_slug
# ---------------------------------------------------------------------------

class TestGetRepoSlug:
    def test_returns_slug(self):
        task = _make_task(github_repo="org/repo")
        assert TaskAnalyticsManager.get_repo_slug(task) == "org/repo"

    def test_returns_none_without_repo(self):
        task = _make_task()
        task.context.pop("github_repo", None)
        assert TaskAnalyticsManager.get_repo_slug(task) is None


# ---------------------------------------------------------------------------
# extract_and_store_memories
# ---------------------------------------------------------------------------

class TestExtractAndStoreMemories:
    def test_extracts_memories(self):
        mgr = _make_manager()
        mgr.memory_retriever.extract_memories_from_response.return_value = 3
        task = _make_task()
        response = MagicMock(content="Learned: use async patterns")
        mgr.extract_and_store_memories(task, response)
        mgr.memory_retriever.extract_memories_from_response.assert_called_once()
        mgr.session_logger.log.assert_called_once()

    def test_skips_when_disabled(self):
        mgr = _make_manager(memory_enabled=False)
        task = _make_task()
        response = MagicMock(content="some output")
        mgr.extract_and_store_memories(task, response)
        mgr.memory_retriever.extract_memories_from_response.assert_not_called()

    def test_skips_without_repo(self):
        mgr = _make_manager()
        task = _make_task()
        task.context.pop("github_repo")
        response = MagicMock(content="output")
        mgr.extract_and_store_memories(task, response)
        mgr.memory_retriever.extract_memories_from_response.assert_not_called()

    def test_feedback_bus_error_non_fatal(self):
        bus = MagicMock()
        bus.process.side_effect = RuntimeError("bus error")
        mgr = _make_manager(feedback_bus=bus)
        mgr.memory_retriever.extract_memories_from_response.return_value = 0
        task = _make_task()
        response = MagicMock(content="output")
        # Should not raise
        mgr.extract_and_store_memories(task, response)


# ---------------------------------------------------------------------------
# extract_summary
# ---------------------------------------------------------------------------

class TestExtractSummary:
    @pytest.mark.asyncio
    async def test_regex_extraction_jira_and_pr(self):
        mgr = _make_manager()
        task = _make_task()
        content = (
            "Created PROJ-123 and opened https://github.com/org/repo/pull/42. "
            "Modified src/foo.py and tests/test_foo.py."
        )
        result = await mgr.extract_summary(content, task)
        assert "PROJ-123" in result
        assert "42" in result

    @pytest.mark.asyncio
    async def test_empty_content(self):
        mgr = _make_manager()
        task = _make_task()
        result = await mgr.extract_summary("", task)
        assert "completed" in result.lower()

    @pytest.mark.asyncio
    async def test_none_content(self):
        mgr = _make_manager()
        task = _make_task()
        result = await mgr.extract_summary(None, task)
        assert "completed" in result.lower()

    @pytest.mark.asyncio
    async def test_recursion_depth_limit(self):
        mgr = _make_manager()
        task = _make_task()
        result = await mgr.extract_summary("some content", task, _recursion_depth=1)
        assert "completed" in result.lower()


# ---------------------------------------------------------------------------
# record_optimization_metrics
# ---------------------------------------------------------------------------

class TestRecordOptimizationMetrics:
    def test_writes_metrics_file(self, tmp_path):
        mgr = _make_manager(workspace=tmp_path)
        task = _make_task()

        mgr.record_optimization_metrics(
            task,
            legacy_prompt_length=5000,
            optimized_prompt_length=3000,
            should_use_optimization_cb=lambda t: True,
            get_active_optimizations_cb=lambda: {"compact_json": True},
        )

        metrics_file = tmp_path / ".agent-communication" / "metrics" / "optimization.jsonl"
        assert metrics_file.exists()
        data = json.loads(metrics_file.read_text().strip())
        assert data["savings_chars"] == 2000
        assert data["canary_active"] is True

    def test_permission_error_non_fatal(self, tmp_path):
        mgr = _make_manager(workspace=Path("/nonexistent/readonly"))
        task = _make_task()
        # Should not raise
        mgr.record_optimization_metrics(task, 5000, 3000)


# ---------------------------------------------------------------------------
# set_session_logger
# ---------------------------------------------------------------------------

class TestSetSessionLogger:
    def test_updates_logger(self):
        mgr = _make_manager()
        new_logger = MagicMock()
        mgr.set_session_logger(new_logger)
        assert mgr.session_logger is new_logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_summary_limits(self):
        assert SUMMARY_CONTEXT_MAX_CHARS == 2000
        assert SUMMARY_MAX_LENGTH == 500

    def test_conversational_prefixes(self):
        assert isinstance(_CONVERSATIONAL_PREFIXES, tuple)
        assert "thank" in _CONVERSATIONAL_PREFIXES


# ---------------------------------------------------------------------------
# _check_reread_waste
# ---------------------------------------------------------------------------

class TestCheckRereadWaste:
    """Static method that scans session log for read_cache_bypass events."""

    def test_detects_high_wasteful_rate(self, tmp_path):
        session_path = tmp_path / "session.jsonl"
        session_path.write_text(json.dumps({
            "event": "read_cache_bypass",
            "cached_files": 10,
            "wasteful_rereads": 5,
            "re_read_count": 6,
        }))
        from agent_framework.memory.tool_pattern_analyzer import ToolPatternAnalyzer
        analyzer = ToolPatternAnalyzer()
        rec = TaskAnalyticsManager._check_reread_waste(session_path, analyzer)
        assert rec is not None
        assert rec.pattern_id == "cross-step-reread"

    def test_no_detection_below_threshold(self, tmp_path):
        session_path = tmp_path / "session.jsonl"
        session_path.write_text(json.dumps({
            "event": "read_cache_bypass",
            "cached_files": 10,
            "wasteful_rereads": 1,
            "re_read_count": 2,
        }))
        from agent_framework.memory.tool_pattern_analyzer import ToolPatternAnalyzer
        analyzer = ToolPatternAnalyzer()
        rec = TaskAnalyticsManager._check_reread_waste(session_path, analyzer)
        assert rec is None

    def test_no_bypass_events(self, tmp_path):
        session_path = tmp_path / "session.jsonl"
        session_path.write_text(json.dumps({
            "event": "tool_call",
            "tool": "Read",
        }))
        from agent_framework.memory.tool_pattern_analyzer import ToolPatternAnalyzer
        analyzer = ToolPatternAnalyzer()
        rec = TaskAnalyticsManager._check_reread_waste(session_path, analyzer)
        assert rec is None

    def test_missing_session_file(self, tmp_path):
        session_path = tmp_path / "missing.jsonl"
        from agent_framework.memory.tool_pattern_analyzer import ToolPatternAnalyzer
        analyzer = ToolPatternAnalyzer()
        rec = TaskAnalyticsManager._check_reread_waste(session_path, analyzer)
        assert rec is None
