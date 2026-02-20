"""Tests for QA warnings injection in PromptBuilder."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_framework.memory.memory_store import MemoryStore
from agent_framework.core.prompt_builder import PromptBuilder, PromptContext
from agent_framework.core.task import Task, TaskStatus, TaskType


def _make_config(**overrides):
    config = MagicMock()
    config.id = "engineer-1"
    config.base_id = "engineer"
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def _make_task(**overrides):
    defaults = dict(
        id="test-task",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=5,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Test task",
        description="A test task",
        context={"github_repo": "myorg/myrepo"},
    )
    defaults.update(overrides)
    return Task(**defaults)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(workspace=tmp_path, enabled=True)


@pytest.fixture
def prompt_builder(tmp_path, store):
    config = _make_config()
    ctx = PromptContext(
        config=config,
        workspace=tmp_path,
        mcp_enabled=False,
        memory_store=store,
        optimization_config={},
    )
    return PromptBuilder(ctx)


class TestInjectQAWarnings:
    def test_injects_warnings_when_qa_recurring_memories_exist(self, prompt_builder, store):
        # Store some qa_recurring memories
        store.remember(
            "myorg/myrepo", "shared", "qa_recurring",
            "[HIGH] correctness in *.py: Missing null checks",
            tags=["correctness", "high"],
        )
        store.remember(
            "myorg/myrepo", "shared", "qa_recurring",
            "[MEDIUM] testing in *.py: No edge case tests",
            tags=["testing", "medium"],
        )

        task = _make_task()
        result = prompt_builder._inject_qa_warnings("Base prompt", task)

        assert "## QA WARNINGS (from previous tasks)" in result
        assert "Missing null checks" in result
        assert "No edge case tests" in result

    def test_no_injection_without_memories(self, prompt_builder):
        task = _make_task()
        result = prompt_builder._inject_qa_warnings("Base prompt", task)
        assert result == "Base prompt"

    def test_no_injection_for_non_engineer(self, tmp_path, store):
        config = _make_config(base_id="qa")
        ctx = PromptContext(
            config=config,
            workspace=tmp_path,
            mcp_enabled=False,
            memory_store=store,
            optimization_config={},
        )
        builder = PromptBuilder(ctx)

        store.remember(
            "myorg/myrepo", "shared", "qa_recurring",
            "[HIGH] correctness: issue",
            tags=["correctness"],
        )

        task = _make_task()
        result = builder._inject_qa_warnings("Base prompt", task)
        assert result == "Base prompt"

    def test_no_injection_without_repo_slug(self, prompt_builder, store):
        store.remember(
            "myorg/myrepo", "shared", "qa_recurring",
            "[HIGH] correctness: issue",
        )

        task = _make_task(context={})  # No github_repo
        result = prompt_builder._inject_qa_warnings("Base prompt", task)
        assert result == "Base prompt"

    def test_no_injection_without_memory_store(self, tmp_path):
        config = _make_config()
        ctx = PromptContext(
            config=config,
            workspace=tmp_path,
            mcp_enabled=False,
            memory_store=None,
            optimization_config={},
        )
        builder = PromptBuilder(ctx)

        task = _make_task()
        result = builder._inject_qa_warnings("Base prompt", task)
        assert result == "Base prompt"

    def test_respects_char_limit(self, prompt_builder, store):
        # Store many warnings to exceed char limit
        for i in range(20):
            store.remember(
                "myorg/myrepo", "shared", "qa_recurring",
                f"[HIGH] category{i} in *.py: A very long warning message that describes a complex issue #{i}",
                tags=[f"cat{i}"],
            )

        task = _make_task()
        result = prompt_builder._inject_qa_warnings("Base prompt", task)

        # Should have the header and some but not all warnings
        assert "## QA WARNINGS" in result
        warnings_section = result.split("## QA WARNINGS")[1]
        # Should be under the char limit (500 + header)
        assert len(warnings_section) < 700
