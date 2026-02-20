"""Tests for PromptBuilder._inject_qa_warnings — Known Pitfalls injection."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_framework.core.prompt_builder import PromptBuilder, PromptContext
from agent_framework.memory.memory_store import MemoryStore
from agent_framework.memory.memory_retriever import MemoryRetriever
from agent_framework.core.task import Task, TaskType


def _make_config(base_id="engineer"):
    cfg = SimpleNamespace()
    cfg.id = base_id
    cfg.base_id = base_id
    cfg.prompt = "You are engineer."
    return cfg


def _make_task(repo="org/repo"):
    task = SimpleNamespace()
    task.id = "task-1"
    task.title = "Fix bug"
    task.description = "Fix the bug"
    task.type = TaskType.IMPLEMENTATION
    task.acceptance_criteria = []
    task.deliverables = []
    task.notes = []
    task.context = {"github_repo": repo}
    task.plan = None
    task.parent_task_id = None
    task.depends_on = []
    task.retry_count = 0
    task.last_error = None
    task.escalation_report = None
    task.replan_history = []
    task.root_id = "root-1"
    return task


@pytest.fixture
def store(tmp_path):
    return MemoryStore(workspace=tmp_path, enabled=True)


@pytest.fixture
def prompt_builder(store, tmp_path):
    config = _make_config()
    retriever = MemoryRetriever(store)
    ctx = PromptContext(
        config=config,
        workspace=tmp_path,
        mcp_enabled=False,
        memory_retriever=retriever,
    )
    return PromptBuilder(ctx)


class TestInjectQAWarnings:
    def test_injects_qa_warnings(self, store, prompt_builder):
        """QA warnings from memory should appear in prompt as Known Pitfalls."""
        store.remember(
            "org/repo", "engineer",
            category="qa_warnings",
            content="[HIGH] security in **/*.py: SQL injection risk (seen 3x)",
            tags=["security"],
        )

        task = _make_task()
        result = prompt_builder._inject_qa_warnings("Base prompt.", task)

        assert "Known Pitfalls" in result
        assert "SQL injection" in result

    def test_injects_missed_criteria(self, store, prompt_builder):
        """Missed criteria from memory should appear in Known Pitfalls."""
        store.remember(
            "org/repo", "engineer",
            category="missed_criteria",
            content="Commonly missed: All tests must pass",
            tags=["implementation"],
        )

        task = _make_task()
        result = prompt_builder._inject_qa_warnings("Base prompt.", task)

        assert "Known Pitfalls" in result
        assert "All tests must pass" in result

    def test_no_injection_when_empty(self, store, prompt_builder):
        """No Known Pitfalls section when no warnings exist."""
        task = _make_task()
        result = prompt_builder._inject_qa_warnings("Base prompt.", task)

        assert "Known Pitfalls" not in result
        assert result == "Base prompt."

    def test_no_injection_for_non_engineer(self, store, tmp_path):
        """QA warnings should only be injected for engineer agents."""
        config = _make_config(base_id="architect")
        retriever = MemoryRetriever(store)
        ctx = PromptContext(
            config=config,
            workspace=tmp_path,
            mcp_enabled=False,
            memory_retriever=retriever,
        )
        builder = PromptBuilder(ctx)

        store.remember(
            "org/repo", "architect",
            category="qa_warnings",
            content="Some warning",
        )

        task = _make_task()
        result = builder._inject_qa_warnings("Base prompt.", task)

        assert "Known Pitfalls" not in result

    def test_no_injection_without_repo(self, store, prompt_builder):
        """No injection when task has no github_repo."""
        task = _make_task()
        task.context = {}  # No repo

        store.remember(
            "org/repo", "engineer",
            category="qa_warnings",
            content="Some warning",
        )

        result = prompt_builder._inject_qa_warnings("Base prompt.", task)
        assert result == "Base prompt."

    def test_combines_qa_and_missed(self, store, prompt_builder):
        """Both QA warnings and missed criteria appear in the same section."""
        store.remember(
            "org/repo", "engineer",
            category="qa_warnings",
            content="[HIGH] performance issue in **/*.py (seen 2x)",
        )
        store.remember(
            "org/repo", "engineer",
            category="missed_criteria",
            content="Commonly missed: Run linter before commit",
        )

        task = _make_task()
        result = prompt_builder._inject_qa_warnings("Base prompt.", task)

        assert "Known Pitfalls" in result
        assert "performance" in result
        assert "linter" in result

    def test_no_injection_when_store_disabled(self, tmp_path):
        """No injection when memory store is disabled."""
        disabled_store = MemoryStore(workspace=tmp_path, enabled=False)
        config = _make_config()
        retriever = MemoryRetriever(disabled_store)
        ctx = PromptContext(
            config=config,
            workspace=tmp_path,
            mcp_enabled=False,
            memory_retriever=retriever,
        )
        builder = PromptBuilder(ctx)

        task = _make_task()
        result = builder._inject_qa_warnings("Base prompt.", task)
        assert result == "Base prompt."
