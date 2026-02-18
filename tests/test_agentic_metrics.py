"""Unit tests for DashboardDataProvider.get_agentic_metrics()."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_framework.web.data_provider import DashboardDataProvider
from agent_framework.web.models import AgenticMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workspace(tmpdir: str) -> Path:
    """Minimal workspace with agents.yaml."""
    workspace = Path(tmpdir)
    config_dir = workspace / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "agents.yaml").write_text(
        """
agents:
  - id: engineer
    name: Engineer
    queue: engineer
    enabled: true
    prompt: "test"
"""
    )
    return workspace


def _write_memory(workspace: Path, repo_slug: str, agent_type: str, entries: list) -> None:
    """Write a memory store JSON file."""
    mem_dir = workspace / ".agent-communication" / "memory" / repo_slug
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / f"{agent_type}.json").write_text(json.dumps(entries))


def _write_session(workspace: Path, task_id: str, events: list) -> None:
    """Write a session JSONL file."""
    sessions_dir = workspace / "logs" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(e) for e in events)
    (sessions_dir / f"{task_id}.jsonl").write_text(lines)


# ---------------------------------------------------------------------------
# Memory metrics
# ---------------------------------------------------------------------------


class TestMemoryMetrics:
    def test_empty_when_no_memory_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)
            metrics = provider.get_agentic_metrics()
            assert metrics.memory.total_entries == 0
            assert metrics.memory.stores_count == 0
            assert metrics.memory.categories == {}

    def test_counts_entries_and_categories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            entries = [
                {"category": "conventions", "content": "use snake_case"},
                {"category": "conventions", "content": "prefer pathlib"},
                {"category": "repo_structure", "content": "src/ layout"},
            ]
            _write_memory(workspace, "owner__repo", "engineer", entries)

            provider = DashboardDataProvider(workspace)
            metrics = provider.get_agentic_metrics()

            assert metrics.memory.total_entries == 3
            assert metrics.memory.stores_count == 1
            assert metrics.memory.categories["conventions"] == 2
            assert metrics.memory.categories["repo_structure"] == 1

    def test_multiple_stores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            _write_memory(workspace, "owner__repo-a", "engineer", [
                {"category": "test_commands", "content": "pytest"},
            ])
            _write_memory(workspace, "owner__repo-b", "architect", [
                {"category": "architectural_decisions", "content": "monolith"},
                {"category": "architectural_decisions", "content": "event-driven"},
            ])

            provider = DashboardDataProvider(workspace)
            metrics = provider.get_agentic_metrics()

            assert metrics.memory.total_entries == 3
            assert metrics.memory.stores_count == 2

    def test_skips_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            mem_dir = workspace / ".agent-communication" / "memory" / "repo"
            mem_dir.mkdir(parents=True, exist_ok=True)
            (mem_dir / "engineer.json").write_text("{invalid}")

            provider = DashboardDataProvider(workspace)
            metrics = provider.get_agentic_metrics()
            # Invalid file is skipped gracefully
            assert metrics.memory.total_entries == 0

    def test_skips_non_list_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            mem_dir = workspace / ".agent-communication" / "memory" / "repo"
            mem_dir.mkdir(parents=True, exist_ok=True)
            (mem_dir / "engineer.json").write_text(json.dumps({"key": "value"}))

            provider = DashboardDataProvider(workspace)
            metrics = provider.get_agentic_metrics()
            assert metrics.memory.total_entries == 0


# ---------------------------------------------------------------------------
# Self-eval metrics
# ---------------------------------------------------------------------------


class TestSelfEvalMetrics:
    def test_empty_when_no_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)
            metrics = provider.get_agentic_metrics()
            assert metrics.self_eval.total_evals == 0
            assert metrics.self_eval.passed == 0
            assert metrics.self_eval.failed == 0

    def test_counts_self_eval_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            _write_session(workspace, "task-001", [
                {"event": "self_eval", "passed": True},
                {"event": "self_eval", "passed": True},
                {"event": "self_eval", "passed": False},
                {"event": "tool_call", "tool": "Bash"},  # not counted
            ])

            provider = DashboardDataProvider(workspace)
            metrics = provider.get_agentic_metrics()

            assert metrics.self_eval.total_evals == 3
            assert metrics.self_eval.passed == 2
            assert metrics.self_eval.failed == 1
            assert metrics.self_eval.sessions_scanned == 1


# ---------------------------------------------------------------------------
# Replan metrics
# ---------------------------------------------------------------------------


class TestReplanMetrics:
    def test_empty_when_no_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)
            metrics = provider.get_agentic_metrics()
            assert metrics.replan.total_replans == 0
            assert metrics.replan.sessions_with_replans == 0

    def test_counts_replan_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            _write_session(workspace, "task-001", [
                {"event": "replan", "reason": "blocked"},
                {"event": "replan", "reason": "new_info"},
            ])
            _write_session(workspace, "task-002", [
                {"event": "replan", "reason": "blocked"},
            ])

            provider = DashboardDataProvider(workspace)
            metrics = provider.get_agentic_metrics()

            assert metrics.replan.total_replans == 3
            assert metrics.replan.sessions_with_replans == 2
            assert metrics.replan.sessions_scanned == 2


# ---------------------------------------------------------------------------
# Specialization metrics
# ---------------------------------------------------------------------------


class TestSpecializationMetrics:
    def test_empty_when_no_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)
            metrics = provider.get_agentic_metrics()
            assert metrics.specialization.profile_counts == {}

    def test_counts_profile_selections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            _write_session(workspace, "task-001", [
                {"event": "specialization_selected", "profile_id": "backend"},
            ])
            _write_session(workspace, "task-002", [
                {"event": "specialization_selected", "profile_id": "backend"},
            ])
            _write_session(workspace, "task-003", [
                {"event": "specialization_selected", "profile_id": "frontend"},
            ])

            provider = DashboardDataProvider(workspace)
            metrics = provider.get_agentic_metrics()

            assert metrics.specialization.profile_counts["backend"] == 2
            assert metrics.specialization.profile_counts["frontend"] == 1


# ---------------------------------------------------------------------------
# Debate metrics
# ---------------------------------------------------------------------------


class TestDebateMetrics:
    def test_empty_when_no_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)
            metrics = provider.get_agentic_metrics()
            assert metrics.debates.total_debates == 0

    def test_counts_debate_starts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            _write_session(workspace, "task-001", [
                {"event": "debate_start", "topic": "architecture choice"},
                {"event": "debate_start", "topic": "test strategy"},
            ])

            provider = DashboardDataProvider(workspace)
            metrics = provider.get_agentic_metrics()

            assert metrics.debates.total_debates == 2
            assert metrics.debates.sessions_scanned == 1


# ---------------------------------------------------------------------------
# Context budget metrics
# ---------------------------------------------------------------------------


class TestContextBudgetMetrics:
    def test_empty_when_no_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)
            metrics = provider.get_agentic_metrics()
            assert metrics.context_budget.budget_exceeded_count == 0
            assert metrics.context_budget.exceeded_by_agent == {}

    def test_counts_exceeded_events_from_activity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)

            # Write activity events directly to the event stream file
            from agent_framework.core.activity import ActivityManager, ActivityEvent
            manager = ActivityManager(workspace)
            manager.append_event(ActivityEvent(
                type="token_budget_exceeded",
                agent="engineer",
                task_id="task-001",
                title="Token budget exceeded: 55000 > 50000",
                timestamp=datetime.now(timezone.utc),
            ))
            manager.append_event(ActivityEvent(
                type="token_budget_exceeded",
                agent="engineer",
                task_id="task-002",
                title="Token budget exceeded: 60000 > 50000",
                timestamp=datetime.now(timezone.utc),
            ))
            manager.append_event(ActivityEvent(
                type="token_budget_exceeded",
                agent="architect",
                task_id="task-003",
                title="Token budget exceeded: 45000 > 40000",
                timestamp=datetime.now(timezone.utc),
            ))
            manager.append_event(ActivityEvent(
                type="complete",  # not counted
                agent="engineer",
                task_id="task-001",
                title="Task complete",
                timestamp=datetime.now(timezone.utc),
            ))

            provider = DashboardDataProvider(workspace)
            metrics = provider.get_agentic_metrics()

            assert metrics.context_budget.budget_exceeded_count == 3
            assert metrics.context_budget.exceeded_by_agent["engineer"] == 2
            assert metrics.context_budget.exceeded_by_agent["architect"] == 1


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------


class TestCaching:
    def test_returns_same_instance_within_ttl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)

            first = provider.get_agentic_metrics()
            second = provider.get_agentic_metrics()

            # Same object means the cache was hit
            assert first is second

    def test_cache_refreshes_after_ttl(self):
        """Manually expire the cache and verify a fresh object is returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)

            first = provider.get_agentic_metrics()

            # Expire the cache by zeroing the timestamp
            provider._agentic_metrics_cache_time = None

            second = provider.get_agentic_metrics()

            # Different object — cache was bypassed
            assert first is not second

    def test_metrics_schema_valid(self):
        """Returned value must be a valid AgenticMetrics instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)
            metrics = provider.get_agentic_metrics()
            assert isinstance(metrics, AgenticMetrics)
            # computed_at should be set
            assert metrics.computed_at is not None
