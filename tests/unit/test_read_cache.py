"""Tests for the read cache feature (cross-step file read dedup).

Covers:
- CLI backend: --append-system-prompt flag and AGENT_ROOT_TASK_ID env var
- Agent: _populate_read_cache() after step completion
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.llm.base import LLMRequest


class TestAppendSystemPromptInCLI:
    """Verify --append-system-prompt flag is wired correctly."""

    def test_append_system_prompt_field_on_request(self):
        req = LLMRequest(
            prompt="test",
            append_system_prompt="EFFICIENCY: don't re-read files",
        )
        assert req.append_system_prompt == "EFFICIENCY: don't re-read files"

    def test_append_system_prompt_none_by_default(self):
        req = LLMRequest(prompt="test")
        assert req.append_system_prompt is None


class TestRootTaskIdInEnv:
    """Verify AGENT_ROOT_TASK_ID is derived from request context."""

    def test_root_task_id_from_context(self):
        """When _root_task_id is in context, it should be preferred over task_id."""
        req = LLMRequest(
            prompt="test",
            context={"_root_task_id": "root-abc", "other": "stuff"},
        )
        task_id = "chain-root-abc-step1"
        # Simulate the env setup logic from claude_cli_backend
        root_id = req.context.get("_root_task_id", task_id) if req.context else task_id
        assert root_id == "root-abc"

    def test_root_task_id_falls_back_to_task_id(self):
        """When _root_task_id is not in context, task_id is used."""
        req = LLMRequest(prompt="test", context={"github_repo": "org/repo"})
        task_id = "standalone-task"
        root_id = req.context.get("_root_task_id", task_id) if req.context else task_id
        assert root_id == "standalone-task"

    def test_root_task_id_no_context(self):
        """When context is None, task_id is used."""
        req = LLMRequest(prompt="test")
        task_id = "task-no-ctx"
        root_id = req.context.get("_root_task_id", task_id) if req.context else task_id
        assert root_id == "task-no-ctx"


class TestPopulateReadCache:
    """Test Agent._populate_read_cache() via bound method."""

    @pytest.fixture
    def workspace(self, tmp_path):
        return tmp_path

    @pytest.fixture
    def agent(self, workspace):
        from agent_framework.core.agent import Agent

        agent = MagicMock()
        agent.workspace = workspace
        agent.config = MagicMock()
        agent.config.base_id = "engineer"
        agent.logger = MagicMock()
        agent._session_logger = MagicMock()
        agent._populate_read_cache = Agent._populate_read_cache.__get__(agent)
        return agent

    @pytest.fixture
    def task(self):
        return Task(
            id="chain-root1-impl-d0",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            created_by="test",
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title="Implement feature",
            description="Do the thing",
            context={
                "_root_task_id": "root1",
                "workflow_step": "implement",
            },
        )

    def test_populates_cache_file(self, agent, task, workspace):
        agent._session_logger.extract_file_reads.return_value = [
            "/src/server.py",
            "/src/models.py",
        ]

        agent._populate_read_cache(task)

        cache_file = workspace / ".agent-communication" / "read-cache" / "root1.json"
        assert cache_file.exists()
        data = json.loads(cache_file.read_text())
        assert data["root_task_id"] == "root1"
        assert "/src/server.py" in data["entries"]
        assert "/src/models.py" in data["entries"]
        assert data["entries"]["/src/server.py"]["read_by"] == "engineer"
        assert data["entries"]["/src/server.py"]["workflow_step"] == "implement"

    def test_preserves_existing_entries(self, agent, task, workspace):
        """Framework-populated entries don't overwrite LLM summaries."""
        cache_dir = workspace / ".agent-communication" / "read-cache"
        cache_dir.mkdir(parents=True)
        existing = {
            "root_task_id": "root1",
            "entries": {
                "/src/server.py": {
                    "summary": "Express server with routes",
                    "read_by": "architect",
                    "read_at": "2026-02-18T10:00:00Z",
                    "workflow_step": "plan",
                },
            },
        }
        (cache_dir / "root1.json").write_text(json.dumps(existing))

        agent._session_logger.extract_file_reads.return_value = [
            "/src/server.py",  # already cached — should NOT overwrite
            "/src/new_file.py",  # new — should be added
        ]

        agent._populate_read_cache(task)

        data = json.loads((cache_dir / "root1.json").read_text())
        # Original entry preserved with its summary
        assert data["entries"]["/src/server.py"]["summary"] == "Express server with routes"
        assert data["entries"]["/src/server.py"]["read_by"] == "architect"
        # New entry added
        assert "/src/new_file.py" in data["entries"]
        assert data["entries"]["/src/new_file.py"]["read_by"] == "engineer"

    def test_noop_when_no_reads(self, agent, task, workspace):
        agent._session_logger.extract_file_reads.return_value = []

        agent._populate_read_cache(task)

        cache_file = workspace / ".agent-communication" / "read-cache" / "root1.json"
        assert not cache_file.exists()

    def test_noop_when_all_already_cached(self, agent, task, workspace):
        """If every read is already in cache, skip the atomic write."""
        cache_dir = workspace / ".agent-communication" / "read-cache"
        cache_dir.mkdir(parents=True)
        existing = {
            "root_task_id": "root1",
            "entries": {
                "/src/server.py": {
                    "summary": "Already cached",
                    "read_by": "architect",
                    "read_at": "2026-02-18T10:00:00Z",
                    "workflow_step": "plan",
                },
            },
        }
        cache_path = cache_dir / "root1.json"
        cache_path.write_text(json.dumps(existing))
        mtime_before = cache_path.stat().st_mtime

        agent._session_logger.extract_file_reads.return_value = ["/src/server.py"]

        agent._populate_read_cache(task)

        # File should not have been rewritten
        assert cache_path.stat().st_mtime == mtime_before

    def test_tolerates_corrupted_cache(self, agent, task, workspace):
        """Corrupted JSON is replaced with fresh cache."""
        cache_dir = workspace / ".agent-communication" / "read-cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "root1.json").write_text("corrupted{{{")

        agent._session_logger.extract_file_reads.return_value = ["/src/file.py"]

        agent._populate_read_cache(task)

        data = json.loads((cache_dir / "root1.json").read_text())
        assert "/src/file.py" in data["entries"]

    def test_auto_generates_summaries_for_real_files(self, agent, task, workspace):
        """When files exist on disk, summaries are auto-generated."""
        src_dir = workspace / "src"
        src_dir.mkdir()
        py_file = src_dir / "models.py"
        py_file.write_text("class User:\n    pass\n\ndef get_user():\n    pass\n")

        agent._session_logger.extract_file_reads.return_value = [str(py_file)]

        agent._populate_read_cache(task)

        cache_file = workspace / ".agent-communication" / "read-cache" / "root1.json"
        data = json.loads(cache_file.read_text())
        entry = data["entries"][str(py_file)]
        assert "classes: User" in entry["summary"]
        assert "funcs: get_user" in entry["summary"]

    def test_auto_summary_empty_for_missing_file(self, agent, task, workspace):
        """Missing files get empty summary (graceful degradation)."""
        agent._session_logger.extract_file_reads.return_value = ["/nonexistent/file.py"]

        agent._populate_read_cache(task)

        cache_file = workspace / ".agent-communication" / "read-cache" / "root1.json"
        data = json.loads(cache_file.read_text())
        assert data["entries"]["/nonexistent/file.py"]["summary"] == ""


class TestDisplayPath:
    """Test PromptBuilder._display_path() strips workspace prefix."""

    @pytest.fixture
    def builder(self, tmp_path):
        from agent_framework.core.prompt_builder import PromptBuilder

        ctx = MagicMock()
        ctx.workspace = tmp_path
        config = MagicMock()
        builder = PromptBuilder.__new__(PromptBuilder)
        builder.ctx = ctx
        builder.config = config
        builder.logger = MagicMock()
        return builder

    def test_strips_workspace_prefix(self, builder, tmp_path):
        full = str(tmp_path / "src" / "main.py")
        assert builder._display_path(full) == "src/main.py"

    def test_preserves_external_path(self, builder):
        assert builder._display_path("/other/repo/file.py") == "/other/repo/file.py"

    def test_strips_trailing_slash(self, builder, tmp_path):
        full = str(tmp_path) + "/file.py"
        assert builder._display_path(full) == "file.py"


class TestRepoScopedCache:
    """Test repo-scoped read cache written by Agent._update_repo_cache."""

    @pytest.fixture
    def workspace(self, tmp_path):
        return tmp_path

    @pytest.fixture
    def agent(self, workspace):
        from agent_framework.core.agent import Agent

        agent = MagicMock()
        agent.workspace = workspace
        agent.config = MagicMock()
        agent.config.base_id = "architect"
        agent.logger = MagicMock()
        agent._session_logger = MagicMock()
        agent._populate_read_cache = Agent._populate_read_cache.__get__(agent)
        agent._update_repo_cache = Agent._update_repo_cache.__get__(agent)
        return agent

    @pytest.fixture
    def task_with_repo(self):
        return Task(
            id="chain-root1-plan-d0",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            created_by="test",
            assigned_to="architect",
            created_at=datetime.now(timezone.utc),
            title="Plan feature",
            description="Plan",
            context={
                "_root_task_id": "root1",
                "workflow_step": "plan",
                "github_repo": "justworkshr/myrepo",
            },
        )

    def test_populates_repo_cache(self, agent, task_with_repo, workspace):
        """_populate_read_cache writes both task-specific and repo-scoped cache."""
        agent._session_logger.extract_file_reads.return_value = [
            "/src/server.py",
            "/src/models.py",
        ]

        agent._populate_read_cache(task_with_repo)

        cache_dir = workspace / ".agent-communication" / "read-cache"
        # Task-specific cache
        assert (cache_dir / "root1.json").exists()
        # Repo-scoped cache
        repo_file = cache_dir / "_repo-justworkshr-myrepo.json"
        assert repo_file.exists()
        data = json.loads(repo_file.read_text())
        assert data["github_repo"] == "justworkshr/myrepo"
        assert "/src/server.py" in data["entries"]
        assert "/src/models.py" in data["entries"]

    def test_repo_cache_merges_across_tasks(self, agent, workspace):
        """Second task's entries merge into existing repo cache."""
        cache_dir = workspace / ".agent-communication" / "read-cache"
        cache_dir.mkdir(parents=True)

        # Simulate first task's repo cache
        existing_repo = {
            "github_repo": "justworkshr/myrepo",
            "entries": {
                "/src/old_file.py": {
                    "summary": "Old file",
                    "read_by": "architect",
                    "read_at": "2026-02-18T09:00:00+00:00",
                    "workflow_step": "plan",
                },
            },
        }
        (cache_dir / "_repo-justworkshr-myrepo.json").write_text(json.dumps(existing_repo))

        # Second task adds new entries
        new_entries = {
            "/src/new_file.py": {
                "summary": "New file",
                "read_by": "engineer",
                "read_at": "2026-02-18T10:00:00+00:00",
                "workflow_step": "implement",
            },
        }
        agent._update_repo_cache(cache_dir, "justworkshr/myrepo", new_entries)

        data = json.loads((cache_dir / "_repo-justworkshr-myrepo.json").read_text())
        assert "/src/old_file.py" in data["entries"]
        assert "/src/new_file.py" in data["entries"]

    def test_repo_cache_evicts_oldest(self, agent, workspace):
        """Entries beyond _MAX_REPO_CACHE_ENTRIES are evicted by read_at age."""
        from agent_framework.core.agent import _MAX_REPO_CACHE_ENTRIES

        cache_dir = workspace / ".agent-communication" / "read-cache"
        cache_dir.mkdir(parents=True)

        # Create repo cache at the limit
        entries = {}
        for i in range(_MAX_REPO_CACHE_ENTRIES):
            entries[f"/src/file_{i:04d}.py"] = {
                "summary": f"File {i}",
                "read_by": "architect",
                "read_at": f"2026-02-18T{i // 60:02d}:{i % 60:02d}:00+00:00",
                "workflow_step": "plan",
            }
        existing = {"github_repo": "org/repo", "entries": entries}
        (cache_dir / "_repo-org-repo.json").write_text(json.dumps(existing))

        # Add 10 more entries with newer timestamps
        new_entries = {}
        for i in range(10):
            new_entries[f"/src/brand_new_{i}.py"] = {
                "summary": f"Brand new {i}",
                "read_by": "engineer",
                "read_at": "2026-02-19T12:00:00+00:00",
                "workflow_step": "implement",
            }
        agent._update_repo_cache(cache_dir, "org/repo", new_entries)

        data = json.loads((cache_dir / "_repo-org-repo.json").read_text())
        assert len(data["entries"]) == _MAX_REPO_CACHE_ENTRIES
        # New entries should be present
        for i in range(10):
            assert f"/src/brand_new_{i}.py" in data["entries"]
        # Oldest entries (lowest read_at) should have been evicted
        assert "/src/file_0000.py" not in data["entries"]

    def test_no_repo_cache_without_github_repo(self, agent, workspace):
        """No repo cache when task lacks github_repo in context."""
        task = Task(
            id="chain-root2-plan-d0",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.IN_PROGRESS,
            priority=1,
            created_by="test",
            assigned_to="architect",
            created_at=datetime.now(timezone.utc),
            title="Plan feature",
            description="Plan",
            context={
                "_root_task_id": "root2",
                "workflow_step": "plan",
            },
        )
        agent._session_logger.extract_file_reads.return_value = ["/src/file.py"]

        agent._populate_read_cache(task)

        cache_dir = workspace / ".agent-communication" / "read-cache"
        assert (cache_dir / "root2.json").exists()
        # No repo cache files should exist
        repo_files = list(cache_dir.glob("_repo-*.json"))
        assert len(repo_files) == 0
