"""Unit tests for AgenticMetrics aggregation in DashboardDataProvider."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_framework.web.data_provider import DashboardDataProvider
from agent_framework.web.models import (
    AgenticMetrics,
    ContextBudgetMetrics,
    DebateMetrics,
    MemoryMetrics,
    ReplanMetrics,
    SelfEvalMetrics,
    SpecializationMetrics,
)


@pytest.fixture
def workspace(tmp_path):
    """Create a minimal workspace directory structure."""
    comm_dir = tmp_path / ".agent-communication"
    comm_dir.mkdir()
    (comm_dir / "queues").mkdir()
    (comm_dir / "completed").mkdir()
    (comm_dir / "memory").mkdir()
    (comm_dir / "profile-registry").mkdir()
    return tmp_path


@pytest.fixture
def provider(workspace):
    """Create a DashboardDataProvider with mocked dependencies."""
    with (
        patch("agent_framework.web.data_provider.ActivityManager") as MockActivity,
        patch("agent_framework.web.data_provider.FileQueue"),
        patch("agent_framework.web.data_provider.CircuitBreaker"),
    ):
        mock_activity = MagicMock()
        mock_activity.get_recent_events.return_value = []
        MockActivity.return_value = mock_activity
        p = DashboardDataProvider(workspace)
        # Expose mock so tests can configure it
        p._mock_activity = mock_activity
        return p


def _write_memory_store(path: Path, entries: list) -> None:
    """Helper to write a memory store JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries))


def _write_task_file(path: Path, data: dict) -> None:
    """Helper to write a task JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class TestMemoryMetrics:
    def test_empty_when_no_memory_dir(self, provider, workspace):
        import shutil
        shutil.rmtree(workspace / ".agent-communication" / "memory")

        result = provider._compute_memory_metrics()

        assert result.total_entries == 0
        assert result.stores_count == 0
        assert result.categories == {}

    def test_counts_entries_and_categories(self, provider, workspace):
        store_path = workspace / ".agent-communication" / "memory" / "myorg__repo" / "engineer.json"
        _write_memory_store(store_path, [
            {"category": "conventions", "content": "Use snake_case"},
            {"category": "conventions", "content": "4-space indent"},
            {"category": "repo_structure", "content": "Tests in tests/"},
        ])

        result = provider._compute_memory_metrics()

        assert result.total_entries == 3
        assert result.stores_count == 1
        assert result.categories["conventions"] == 2
        assert result.categories["repo_structure"] == 1

    def test_aggregates_across_multiple_stores(self, provider, workspace):
        mem_base = workspace / ".agent-communication" / "memory"
        _write_memory_store(mem_base / "repo_a" / "engineer.json", [
            {"category": "conventions", "content": "foo"},
        ])
        _write_memory_store(mem_base / "repo_b" / "architect.json", [
            {"category": "architectural_decisions", "content": "bar"},
            {"category": "architectural_decisions", "content": "baz"},
        ])

        result = provider._compute_memory_metrics()

        assert result.total_entries == 3
        assert result.stores_count == 2

    def test_skips_invalid_json(self, provider, workspace):
        store_path = workspace / ".agent-communication" / "memory" / "repo" / "bad.json"
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text("not valid json {{{")

        result = provider._compute_memory_metrics()

        assert result.total_entries == 0


class TestSelfEvalMetrics:
    def test_empty_when_no_tasks(self, provider):
        result = provider._compute_self_eval_metrics()

        assert result.tasks_evaluated == 0
        assert result.total_retries == 0

    def test_counts_tasks_with_self_eval(self, provider, workspace):
        completed = workspace / ".agent-communication" / "completed"
        _write_task_file(completed / "task-1.json", {
            "context": {"_self_eval_count": 2},
        })
        _write_task_file(completed / "task-2.json", {
            "context": {"_self_eval_count": 0},  # 0 means no eval
        })
        _write_task_file(completed / "task-3.json", {
            "context": {"_self_eval_count": 1},
        })

        result = provider._compute_self_eval_metrics()

        assert result.tasks_evaluated == 2
        assert result.total_retries == 3

    def test_handles_missing_context(self, provider, workspace):
        completed = workspace / ".agent-communication" / "completed"
        _write_task_file(completed / "task-1.json", {"id": "t1"})  # no context key

        result = provider._compute_self_eval_metrics()

        assert result.tasks_evaluated == 0
        assert result.total_retries == 0


class TestReplanMetrics:
    def test_empty_when_no_tasks(self, provider):
        result = provider._compute_replan_metrics()

        assert result.tasks_replanned == 0
        assert result.total_replan_attempts == 0

    def test_counts_tasks_with_replan_history(self, provider, workspace):
        completed = workspace / ".agent-communication" / "completed"
        _write_task_file(completed / "task-a.json", {
            "replan_history": [{"attempt": 1, "error": "oops"}],
        })
        _write_task_file(completed / "task-b.json", {
            "replan_history": [
                {"attempt": 1, "error": "oops"},
                {"attempt": 2, "error": "still failing"},
            ],
        })
        _write_task_file(completed / "task-c.json", {
            "replan_history": [],  # empty — no replan
        })

        result = provider._compute_replan_metrics()

        assert result.tasks_replanned == 2
        assert result.total_replan_attempts == 3

    def test_searches_queue_dir_too(self, provider, workspace):
        queues = workspace / ".agent-communication" / "queues" / "engineer"
        _write_task_file(queues / "task-q.json", {
            "replan_history": [{"attempt": 1, "error": "x"}],
        })

        result = provider._compute_replan_metrics()

        assert result.tasks_replanned == 1


class TestSpecializationMetrics:
    def test_empty_when_no_registry(self, provider, workspace):
        import os
        registry = workspace / ".agent-communication" / "profile-registry" / "profiles.json"
        if registry.exists():
            os.unlink(registry)

        result = provider._compute_specialization_metrics()

        assert result.profiles_cached == 0
        assert result.total_matches == 0

    def test_counts_profiles_and_matches(self, provider, workspace):
        registry = workspace / ".agent-communication" / "profile-registry" / "profiles.json"
        registry.write_text(json.dumps([
            {"match_count": 5, "profile": {"id": "backend"}},
            {"match_count": 3, "profile": {"id": "frontend"}},
            {"match_count": 0, "profile": {"id": "infra"}},
        ]))

        result = provider._compute_specialization_metrics()

        assert result.profiles_cached == 3
        assert result.total_matches == 8


class TestDebateMetrics:
    def test_empty_when_no_memory(self, provider, workspace):
        result = provider._compute_debate_metrics()

        assert result.debates_recorded == 0
        assert result.high_confidence_count == 0

    def test_counts_debate_tagged_entries(self, provider, workspace):
        store_path = workspace / ".agent-communication" / "memory" / "repo" / "engineer.json"
        _write_memory_store(store_path, [
            {
                "category": "architectural_decisions",
                "content": "Use PostgreSQL\nConfidence: high",
                "tags": ["debate"],
            },
            {
                "category": "architectural_decisions",
                "content": "Avoid Redis\nConfidence: medium",
                "tags": ["debate"],
            },
            {
                "category": "conventions",
                "content": "Use snake_case",
                "tags": [],  # not a debate
            },
        ])

        result = provider._compute_debate_metrics()

        assert result.debates_recorded == 2
        assert result.high_confidence_count == 1

    def test_only_high_confidence_counted(self, provider, workspace):
        store_path = workspace / ".agent-communication" / "memory" / "r" / "a.json"
        _write_memory_store(store_path, [
            {
                "category": "architectural_decisions",
                "content": "Decision A\nConfidence: low",
                "tags": ["debate"],
            },
            {
                "category": "architectural_decisions",
                "content": "Decision B\nConfidence: high",
                "tags": ["debate"],
            },
            {
                "category": "architectural_decisions",
                "content": "Decision C\nConfidence: high",
                "tags": ["debate"],
            },
        ])

        result = provider._compute_debate_metrics()

        assert result.debates_recorded == 3
        assert result.high_confidence_count == 2


class TestContextBudgetMetrics:
    def test_zero_when_no_critical_events(self, provider):
        provider._mock_activity.get_recent_events.return_value = []

        result = provider._compute_context_budget_metrics()

        assert result.critical_events == 0

    def test_counts_critical_budget_events(self, provider):
        mock_events = [
            MagicMock(type="context_budget_critical"),
            MagicMock(type="context_budget_critical"),
            MagicMock(type="start"),
            MagicMock(type="complete"),
        ]
        provider._mock_activity.get_recent_events.return_value = mock_events

        result = provider._compute_context_budget_metrics()

        assert result.critical_events == 2


class TestGetAgenticMetrics:
    def test_returns_complete_metrics(self, provider):
        result = provider.get_agentic_metrics()

        assert isinstance(result, AgenticMetrics)
        assert isinstance(result.memory, MemoryMetrics)
        assert isinstance(result.self_eval, SelfEvalMetrics)
        assert isinstance(result.replan, ReplanMetrics)
        assert isinstance(result.specialization, SpecializationMetrics)
        assert isinstance(result.debate, DebateMetrics)
        assert isinstance(result.context_budget, ContextBudgetMetrics)
        assert isinstance(result.computed_at, datetime)

    def test_caches_result_within_ttl(self, provider, workspace):
        """Second call within TTL returns same object (no re-scan)."""
        first = provider.get_agentic_metrics()
        second = provider.get_agentic_metrics()

        # Same object due to caching
        assert first is second

    def test_cache_expires_after_ttl(self, provider, workspace):
        """After TTL, metrics are recomputed."""
        first = provider.get_agentic_metrics()

        # Force cache expiry
        provider._agentic_metrics_cache_time = datetime(2000, 1, 1, tzinfo=timezone.utc)

        second = provider.get_agentic_metrics()

        # Different objects after cache miss
        assert first is not second
