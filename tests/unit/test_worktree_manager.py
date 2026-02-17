"""Unit tests for WorktreeManager."""

import json
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import subprocess

from agent_framework.workspace.worktree_manager import (
    WorktreeManager,
    WorktreeConfig,
    WorktreeInfo,
    STALE_ACTIVE_THRESHOLD_SECONDS,
)


class TestWorktreeConfig:
    """Tests for WorktreeConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WorktreeConfig()
        assert config.enabled is False
        assert config.cleanup_on_complete is True
        assert config.cleanup_on_failure is False
        assert config.max_age_hours == 24
        assert config.max_worktrees == 20

    def test_config_with_custom_values(self):
        """Test custom configuration."""
        config = WorktreeConfig(
            enabled=True,
            root=Path("/custom/path"),
            cleanup_on_complete=False,
            cleanup_on_failure=True,
            max_age_hours=48,
            max_worktrees=10,
        )
        assert config.enabled is True
        assert config.cleanup_on_complete is False
        assert config.cleanup_on_failure is True
        assert config.max_age_hours == 48
        assert config.max_worktrees == 10

    def test_config_path_expansion(self):
        """Test that paths are expanded."""
        config = WorktreeConfig(root="~/test/worktrees")
        assert "~" not in str(config.root)


class TestWorktreeInfo:
    """Tests for WorktreeInfo."""

    def test_to_dict(self):
        """Test serialization to dict."""
        info = WorktreeInfo(
            path="/path/to/worktree",
            branch="feature/test",
            agent_id="engineer",
            task_id="task-123",
            created_at="2025-01-01T00:00:00",
            last_accessed="2025-01-01T01:00:00",
            base_repo="/path/to/base",
        )
        data = info.to_dict()
        assert data["path"] == "/path/to/worktree"
        assert data["branch"] == "feature/test"
        assert data["agent_id"] == "engineer"

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "path": "/path/to/worktree",
            "branch": "feature/test",
            "agent_id": "engineer",
            "task_id": "task-123",
            "created_at": "2025-01-01T00:00:00",
            "last_accessed": "2025-01-01T01:00:00",
            "base_repo": "/path/to/base",
        }
        info = WorktreeInfo.from_dict(data)
        assert info.path == "/path/to/worktree"
        assert info.branch == "feature/test"


class TestBranchNameValidation:
    """Tests for branch name validation."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a WorktreeManager with temp directory."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        return WorktreeManager(config=config)

    def test_validate_branch_name_valid(self, manager):
        """Test valid branch names."""
        valid_names = [
            "main",
            "feature/test",
            "agent/engineer/PROJ-123",
            "fix_bug-123",
        ]
        for name in valid_names:
            assert manager._validate_branch_name(name) == name

    def test_validate_branch_name_empty(self, manager):
        """Test empty branch name raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            manager._validate_branch_name("")

    def test_validate_branch_name_invalid_chars(self, manager):
        """Test invalid characters raise error."""
        with pytest.raises(ValueError, match="Invalid branch name"):
            manager._validate_branch_name("feature/test@branch")

    def test_validate_branch_name_path_traversal(self, manager):
        """Test path traversal attempt raises error."""
        # The regex check catches '.' before the '..' check
        with pytest.raises(ValueError, match="Invalid branch name"):
            manager._validate_branch_name("feature/../main")

    def test_validate_branch_name_too_long(self, manager):
        """Test excessively long branch name raises error."""
        with pytest.raises(ValueError, match="too long"):
            manager._validate_branch_name("a" * 256)

    def test_validate_branch_name_leading_slash(self, manager):
        """Test leading slash raises error."""
        with pytest.raises(ValueError, match="cannot start or end with"):
            manager._validate_branch_name("/feature/test")

    def test_validate_branch_name_trailing_slash(self, manager):
        """Test trailing slash raises error."""
        with pytest.raises(ValueError, match="cannot start or end with"):
            manager._validate_branch_name("feature/test/")


class TestWorktreeKeyGeneration:
    """Tests for worktree key generation."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a WorktreeManager with temp directory."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        return WorktreeManager(config=config)

    def test_get_worktree_key(self, manager):
        """Test worktree key generation."""
        key = manager._get_worktree_key("engineer", "task-12345678-abcd")
        assert key == "engineer-task-12345678-abcd"

    def test_get_worktree_key_jira_prefix(self, manager):
        """Test worktree key extracts JIRA ticket from jira-prefixed IDs."""
        key = manager._get_worktree_key("engineer", "jira-ME-429-1770446158")
        assert key == "engineer-ME-429"

    def test_get_worktree_path(self, manager):
        """Test worktree path generation."""
        path = manager._get_worktree_path("owner/repo", "engineer", "task-12345678")
        assert "owner" in str(path)
        assert "repo" in str(path)
        assert "engineer-task-12345678" in str(path)


class TestRegistryOperations:
    """Tests for registry load/save."""

    def test_empty_registry_on_init(self, tmp_path):
        """Test registry is empty when no file exists."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)
        assert len(manager._registry) == 0

    def test_load_existing_registry(self, tmp_path):
        """Test loading existing registry file."""
        worktree_root = tmp_path / "worktrees"
        worktree_root.mkdir(parents=True)

        # Create registry file
        registry_data = {
            "engineer-task-123": {
                "path": "/path/to/worktree",
                "branch": "feature/test",
                "agent_id": "engineer",
                "task_id": "task-12345678",
                "created_at": "2025-01-01T00:00:00",
                "last_accessed": "2025-01-01T01:00:00",
                "base_repo": "/path/to/base",
            }
        }
        registry_file = worktree_root / ".worktree-registry.json"
        registry_file.write_text(json.dumps(registry_data))

        config = WorktreeConfig(enabled=True, root=worktree_root)
        manager = WorktreeManager(config=config)

        assert len(manager._registry) == 1
        assert "engineer-task-123" in manager._registry

    def test_save_registry(self, tmp_path):
        """Test saving registry to file."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        # Add entry
        manager._registry["test-key"] = WorktreeInfo(
            path="/test/path",
            branch="test-branch",
            agent_id="test-agent",
            task_id="test-task",
            created_at="2025-01-01T00:00:00",
            last_accessed="2025-01-01T00:00:00",
            base_repo="/base/repo",
        )
        manager._save_registry()

        # Verify file exists
        registry_file = tmp_path / "worktrees" / ".worktree-registry.json"
        assert registry_file.exists()

        # Verify content
        data = json.loads(registry_file.read_text())
        assert "test-key" in data


class TestListWorktrees:
    """Tests for listing worktrees."""

    def test_list_worktrees_empty(self, tmp_path):
        """Test listing empty worktrees."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)
        assert manager.list_worktrees() == []

    def test_list_worktrees_with_entries(self, tmp_path):
        """Test listing worktrees with entries."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        # Add entries
        manager._registry["key1"] = WorktreeInfo(
            path="/path1", branch="b1", agent_id="a1", task_id="t1",
            created_at="2025-01-01T00:00:00", last_accessed="2025-01-01T00:00:00",
            base_repo="/base",
        )
        manager._registry["key2"] = WorktreeInfo(
            path="/path2", branch="b2", agent_id="a2", task_id="t2",
            created_at="2025-01-01T00:00:00", last_accessed="2025-01-01T00:00:00",
            base_repo="/base",
        )

        worktrees = manager.list_worktrees()
        assert len(worktrees) == 2


class TestGetStats:
    """Tests for worktree statistics."""

    def test_get_stats_empty(self, tmp_path):
        """Test stats with no worktrees."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        stats = manager.get_stats()
        assert stats["total_registered"] == 0
        assert stats["active"] == 0
        assert stats["orphaned"] == 0

    def test_get_stats_with_worktrees(self, tmp_path):
        """Test stats with registered worktrees."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees", max_worktrees=10)
        manager = WorktreeManager(config=config)

        # Add an entry (path doesn't exist)
        manager._registry["key1"] = WorktreeInfo(
            path="/nonexistent", branch="b1", agent_id="a1", task_id="t1",
            created_at="2025-01-01T00:00:00", last_accessed="2025-01-01T00:00:00",
            base_repo="/base",
        )

        stats = manager.get_stats()
        assert stats["total_registered"] == 1
        assert stats["active"] == 0  # Path doesn't exist
        assert stats["orphaned"] == 1
        assert stats["max_worktrees"] == 10


class TestGetWorktreeForTask:
    """Tests for finding worktree by task ID."""

    def test_find_existing_worktree(self, tmp_path):
        """Test finding an existing worktree."""
        worktree_path = tmp_path / "worktree"
        worktree_path.mkdir()

        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        manager._registry["agent-task-123"] = WorktreeInfo(
            path=str(worktree_path), branch="b1", agent_id="agent", task_id="task-12345678",
            created_at="2025-01-01T00:00:00+00:00", last_accessed="2025-01-01T00:00:00+00:00",
            base_repo="/base",
        )

        result = manager.get_worktree_for_task("task-12345678")
        assert result == worktree_path

    def test_find_nonexistent_worktree(self, tmp_path):
        """Test finding a worktree that doesn't exist."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        result = manager.get_worktree_for_task("nonexistent-task")
        assert result is None


class TestIdentifierValidation:
    """Tests for agent_id and task_id validation."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a WorktreeManager with temp directory."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        return WorktreeManager(config=config)

    def test_validate_identifier_valid(self, manager):
        """Test valid identifiers."""
        valid_ids = ["engineer", "agent-1", "task_123", "my_agent"]
        for id_val in valid_ids:
            assert manager._validate_identifier(id_val, "test") == id_val

    def test_validate_identifier_empty(self, manager):
        """Test empty identifier raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            manager._validate_identifier("", "agent_id")

    def test_validate_identifier_path_traversal(self, manager):
        """Test path traversal attempt raises error."""
        with pytest.raises(ValueError, match="Invalid"):
            manager._validate_identifier("../etc", "agent_id")

    def test_validate_identifier_slash(self, manager):
        """Test slash in identifier raises error."""
        with pytest.raises(ValueError, match="Invalid"):
            manager._validate_identifier("agent/bad", "agent_id")


class TestOwnerRepoValidation:
    """Tests for owner_repo validation."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create a WorktreeManager with temp directory."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        return WorktreeManager(config=config)

    def test_validate_owner_repo_valid(self, manager):
        """Test valid owner/repo format."""
        valid = ["owner/repo", "my-org/my-repo", "user123/project_1"]
        for repo in valid:
            assert manager._validate_owner_repo(repo) == repo

    def test_validate_owner_repo_invalid_format(self, manager):
        """Test invalid format raises error."""
        with pytest.raises(ValueError, match="Invalid repository format"):
            manager._validate_owner_repo("just-repo")

    def test_validate_owner_repo_path_traversal(self, manager):
        """Test path traversal raises error."""
        with pytest.raises(ValueError, match="Invalid repository"):
            manager._validate_owner_repo("../etc/passwd")

    def test_validate_owner_repo_multiple_slashes(self, manager):
        """Test multiple slashes raises error."""
        with pytest.raises(ValueError, match="Invalid repository format"):
            manager._validate_owner_repo("owner/repo/extra")


class TestPhantomWorktreeCleanup:
    """Tests for phantom worktree handling (both path and base_repo gone)."""

    def test_remove_worktree_directory_phantom_returns_true(self, tmp_path):
        """Phantom worktree (path + base_repo both gone) returns True without shelling out."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        nonexistent_path = tmp_path / "gone-worktree"
        nonexistent_base = tmp_path / "gone-base-repo"

        result = manager._remove_worktree_directory(nonexistent_path, nonexistent_base)
        assert result is True

    def test_remove_worktree_directory_phantom_no_base_repo(self, tmp_path):
        """Phantom worktree with None base_repo returns True."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        nonexistent_path = tmp_path / "gone-worktree"
        result = manager._remove_worktree_directory(nonexistent_path, None)
        assert result is True

    def test_cleanup_orphaned_purges_phantom_from_registry(self, tmp_path):
        """Stale phantom worktrees are purged from registry without subprocess calls."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees", max_age_hours=0)
        manager = WorktreeManager(config=config)

        # Register a phantom entry (both path and base_repo point nowhere)
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        manager._registry["phantom-key"] = WorktreeInfo(
            path=str(tmp_path / "gone-worktree"),
            branch="feature/gone",
            agent_id="engineer",
            task_id="task-phantom",
            created_at=old_ts,
            last_accessed=old_ts,
            base_repo=str(tmp_path / "gone-base-repo"),
        )

        with patch.object(manager, '_run_git') as mock_git:
            result = manager.cleanup_orphaned_worktrees()
            # Should NOT call git — phantom path is already gone
            mock_git.assert_not_called()

        assert result["registered"] == 1
        assert "phantom-key" not in manager._registry


class TestActiveWorktreeProtection:
    """Tests for active worktree eviction protection."""

    def test_worktree_info_active_defaults_false(self):
        """Active flag defaults to False for backwards compatibility."""
        info = WorktreeInfo(
            path="/p", branch="b", agent_id="a", task_id="t",
            created_at="2025-01-01T00:00:00", last_accessed="2025-01-01T00:00:00",
            base_repo="/base",
        )
        assert info.active is False

    def test_worktree_info_from_dict_without_active_field(self):
        """from_dict handles registries that predate the active field."""
        data = {
            "path": "/p", "branch": "b", "agent_id": "a", "task_id": "t",
            "created_at": "2025-01-01T00:00:00", "last_accessed": "2025-01-01T00:00:00",
            "base_repo": "/base",
        }
        info = WorktreeInfo.from_dict(data)
        assert info.active is False

    def test_worktree_info_active_serializes(self):
        """Active flag round-trips through to_dict/from_dict."""
        info = WorktreeInfo(
            path="/p", branch="b", agent_id="a", task_id="t",
            created_at="2025-01-01T00:00:00", last_accessed="2025-01-01T00:00:00",
            base_repo="/base", active=True,
        )
        data = info.to_dict()
        assert data["active"] is True
        restored = WorktreeInfo.from_dict(data)
        assert restored.active is True

    def test_enforce_capacity_skips_active_worktrees(self, tmp_path):
        """Active worktrees are not evicted even when they're the oldest."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees", max_worktrees=2)
        manager = WorktreeManager(config=config)

        now = datetime.now(timezone.utc)
        # 30 min old — older than the inactive entry, but well within the 2h stale threshold
        old_ts = (now - timedelta(minutes=30)).isoformat()
        recent_ts = (now - timedelta(minutes=5)).isoformat()

        # Oldest entry — but active
        manager._registry["active-old"] = WorktreeInfo(
            path=str(tmp_path / "wt-active"), branch="b1", agent_id="a1", task_id="t1",
            created_at=old_ts, last_accessed=old_ts, base_repo=str(tmp_path),
            active=True,
        )
        # Recent entry — inactive
        manager._registry["inactive-recent"] = WorktreeInfo(
            path=str(tmp_path / "wt-inactive"), branch="b2", agent_id="a2", task_id="t2",
            created_at=recent_ts, last_accessed=recent_ts, base_repo=str(tmp_path),
            active=False,
        )

        with patch.object(manager, '_remove_worktree_directory', return_value=True):
            with patch.object(manager, '_load_registry'):
                manager._enforce_capacity_limit()

        # Active worktree survives, inactive one evicted
        assert "active-old" in manager._registry
        assert "inactive-recent" not in manager._registry

    def test_enforce_capacity_evicts_stale_active_worktrees(self, tmp_path):
        """Active worktrees past the staleness threshold are evicted."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees", max_worktrees=1)
        manager = WorktreeManager(config=config)

        stale_ts = (
            datetime.now(timezone.utc) - timedelta(seconds=STALE_ACTIVE_THRESHOLD_SECONDS + 60)
        ).isoformat()

        manager._registry["stale-active"] = WorktreeInfo(
            path=str(tmp_path / "wt-stale"), branch="b1", agent_id="a1", task_id="t1",
            created_at=stale_ts, last_accessed=stale_ts, base_repo=str(tmp_path),
            active=True,
        )

        with patch.object(manager, '_remove_worktree_directory', return_value=True):
            with patch.object(manager, '_load_registry'):
                manager._enforce_capacity_limit()

        assert "stale-active" not in manager._registry

    def test_cleanup_orphaned_skips_active_worktrees(self, tmp_path):
        """cleanup_orphaned_worktrees skips worktrees that are actively in use."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees", max_age_hours=0)
        manager = WorktreeManager(config=config)

        # Create a worktree directory so it's not a phantom
        wt_path = tmp_path / "wt-active"
        wt_path.mkdir()

        # 30 min old — past max_age_hours=0 but within the 2h stale-active threshold
        old_ts = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        manager._registry["active-key"] = WorktreeInfo(
            path=str(wt_path), branch="b1", agent_id="a1", task_id="t1",
            created_at=old_ts, last_accessed=old_ts, base_repo=str(tmp_path),
            active=True,
        )

        with patch.object(manager, '_remove_worktree_directory', return_value=True) as mock_remove:
            result = manager.cleanup_orphaned_worktrees()

        # Active worktree should not be removed
        assert "active-key" in manager._registry
        mock_remove.assert_not_called()
        assert result["registered"] == 0

    def test_mark_worktree_inactive(self, tmp_path):
        """mark_worktree_inactive clears the active flag and saves."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        wt_path = tmp_path / "wt"
        manager._registry["key1"] = WorktreeInfo(
            path=str(wt_path), branch="b1", agent_id="a1", task_id="t1",
            created_at="2025-01-01T00:00:00", last_accessed="2025-01-01T00:00:00",
            base_repo="/base", active=True,
        )

        with patch.object(manager, '_save_registry') as mock_save:
            manager.mark_worktree_inactive(wt_path)

        assert manager._registry["key1"].active is False
        mock_save.assert_called_once()

    def test_mark_worktree_inactive_nonexistent_path(self, tmp_path):
        """mark_worktree_inactive is a no-op for unknown paths."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        with patch.object(manager, '_save_registry') as mock_save:
            manager.mark_worktree_inactive(tmp_path / "does-not-exist")

        mock_save.assert_not_called()
