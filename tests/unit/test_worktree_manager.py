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

    def test_get_worktree_key_truncates_long_ids(self, manager):
        """Deeply nested chain IDs get capped before they blow up filesystem paths."""
        long_id = "chain-" * 13 + "original"  # 86 chars
        cap = manager._MAX_WORKTREE_KEY_LENGTH
        assert len(long_id) > cap
        key = manager._get_worktree_key("engineer", long_id)
        ticket_part = key[len("engineer-"):]
        assert len(ticket_part) <= cap
        assert not ticket_part.endswith("-")  # rstrip cleans up dash-terminated truncations

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

    def test_remove_worktree_directory_path_gone_base_repo_exists(self, tmp_path):
        """Path removed by another process but base_repo still exists — prune and return True."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        nonexistent_path = tmp_path / "gone-worktree"
        base_repo = tmp_path / "base-repo"
        base_repo.mkdir()

        with patch.object(manager, '_run_git') as mock_git:
            result = manager._remove_worktree_directory(nonexistent_path, base_repo)

        assert result is True
        mock_git.assert_called_once_with(["worktree", "prune"], cwd=base_repo, timeout=30)

    def test_remove_worktree_directory_path_gone_prune_fails(self, tmp_path):
        """Prune failure is swallowed — still returns True since path is already gone."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        nonexistent_path = tmp_path / "gone-worktree"
        base_repo = tmp_path / "base-repo"
        base_repo.mkdir()

        with patch.object(manager, '_run_git', side_effect=subprocess.CalledProcessError(1, "git")):
            result = manager._remove_worktree_directory(nonexistent_path, base_repo)

        assert result is True

    def test_cleanup_orphaned_purges_phantom_from_registry(self, tmp_path):
        """Stale worktrees with missing paths are purged from registry without subprocess calls."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees", max_age_hours=0)
        manager = WorktreeManager(config=config)

        # Register an entry whose path is gone but base_repo still exists
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        base_repo = tmp_path / "base-repo"
        base_repo.mkdir()
        manager._registry["phantom-key"] = WorktreeInfo(
            path=str(tmp_path / "gone-worktree"),
            branch="feature/gone",
            agent_id="engineer",
            task_id="task-phantom",
            created_at=old_ts,
            last_accessed=old_ts,
            base_repo=str(base_repo),
        )

        with patch.object(manager, '_run_git') as mock_git, \
             patch.object(manager, '_load_registry'):
            result = manager.cleanup_orphaned_worktrees()
            # Path-gone short-circuit in cleanup_orphaned_worktrees bypasses _remove_worktree_directory
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

        with patch.object(manager, '_remove_worktree_directory', return_value=True) as mock_remove, \
             patch.object(manager, '_load_registry'):
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


class TestReloadRegistry:
    """Tests for reload_registry() cross-process visibility."""

    def test_reload_picks_up_external_changes(self, tmp_path):
        """Registry changes written by another process are visible after reload."""
        worktree_root = tmp_path / "worktrees"
        worktree_root.mkdir(parents=True)

        config = WorktreeConfig(enabled=True, root=worktree_root)
        manager = WorktreeManager(config=config)
        assert len(manager._registry) == 0

        # Simulate another process writing to the registry file
        registry_data = {
            "engineer-task-ext": {
                "path": str(tmp_path / "wt-ext"),
                "branch": "agent/engineer/EXT-1",
                "agent_id": "engineer",
                "task_id": "task-ext",
                "created_at": "2025-01-01T00:00:00+00:00",
                "last_accessed": "2025-01-01T00:00:00+00:00",
                "base_repo": str(tmp_path),
                "active": True,
            }
        }
        registry_file = worktree_root / ".worktree-registry.json"
        registry_file.write_text(json.dumps(registry_data))

        # Before reload — still empty
        assert len(manager._registry) == 0

        manager.reload_registry()

        assert len(manager._registry) == 1
        assert "engineer-task-ext" in manager._registry
        assert manager._registry["engineer-task-ext"].branch == "agent/engineer/EXT-1"

    def test_reload_delegates_to_load_registry(self, tmp_path):
        """reload_registry() calls _load_registry() internally."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        with patch.object(manager, '_load_registry') as mock_load:
            manager.reload_registry()
            mock_load.assert_called_once()


class TestParseBranchConflictPath:
    """Tests for _parse_branch_conflict_path()."""

    @pytest.fixture
    def manager(self, tmp_path):
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        return WorktreeManager(config=config)

    def test_parses_standard_error_message(self, manager):
        """Extracts path from standard git 'already checked out' error."""
        error = "fatal: 'agent/engineer/ME-123' is already checked out at '/home/user/.agent-workspaces/worktrees/owner/repo/engineer-ME-123'"
        result = manager._parse_branch_conflict_path(error)
        assert result == Path("/home/user/.agent-workspaces/worktrees/owner/repo/engineer-ME-123")

    def test_returns_none_for_unrelated_error(self, manager):
        """Returns None when error doesn't match the conflict pattern."""
        error = "fatal: not a git repository"
        result = manager._parse_branch_conflict_path(error)
        assert result is None

    def test_returns_none_for_empty_string(self, manager):
        """Returns None for empty error message."""
        result = manager._parse_branch_conflict_path("")
        assert result is None


class TestCreateWorktreeBranchConflict:
    """Tests for branch conflict recovery in create_worktree()."""

    def test_returns_existing_path_on_branch_conflict(self, tmp_path):
        """When git says branch is already checked out, return that path."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        # Create a fake base repo and existing worktree path
        base_repo = tmp_path / "base"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()
        existing_wt = tmp_path / "existing-worktree"
        existing_wt.mkdir()

        conflict_msg = f"fatal: 'feature/test' is already checked out at '{existing_wt}'"
        error = subprocess.CalledProcessError(128, "git")
        error.stderr = conflict_msg.encode()

        with patch.object(manager, '_enforce_capacity_limit'), \
             patch.object(manager, '_run_git', side_effect=error):
            result = manager.create_worktree(
                base_repo=base_repo,
                branch_name="feature/test",
                agent_id="engineer",
                task_id="task-conflict",
                owner_repo="owner/repo",
            )

        assert result == existing_wt

        # Verify the conflict worktree was registered
        key = manager._get_worktree_key("engineer", "task-conflict")
        assert key in manager._registry
        info = manager._registry[key]
        assert info.branch == "feature/test"
        assert info.path == str(existing_wt)
        assert info.active is True

    def test_raises_on_non_conflict_error(self, tmp_path):
        """Non-conflict CalledProcessError still raises RuntimeError."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        base_repo = tmp_path / "base"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()

        error = subprocess.CalledProcessError(128, "git")
        error.stderr = b"fatal: some other git error"

        with patch.object(manager, '_enforce_capacity_limit'), \
             patch.object(manager, '_run_git', side_effect=error):
            with pytest.raises(RuntimeError, match="some other git error"):
                manager.create_worktree(
                    base_repo=base_repo,
                    branch_name="feature/test",
                    agent_id="engineer",
                    task_id="task-other",
                    owner_repo="owner/repo",
                )

    def test_raises_when_conflict_path_does_not_exist(self, tmp_path):
        """If the parsed conflict path doesn't exist on disk, still raise."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        base_repo = tmp_path / "base"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()

        ghost_path = tmp_path / "ghost-worktree"
        conflict_msg = f"fatal: 'feature/test' is already checked out at '{ghost_path}'"
        error = subprocess.CalledProcessError(128, "git")
        error.stderr = conflict_msg.encode()

        with patch.object(manager, '_enforce_capacity_limit'), \
             patch.object(manager, '_run_git', side_effect=error):
            with pytest.raises(RuntimeError, match="already checked out"):
                manager.create_worktree(
                    base_repo=base_repo,
                    branch_name="feature/test",
                    agent_id="engineer",
                    task_id="task-ghost",
                    owner_repo="owner/repo",
                )

    def test_raises_when_stderr_is_none(self, tmp_path):
        """CalledProcessError with None stderr falls through to RuntimeError."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        base_repo = tmp_path / "base"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()

        error = subprocess.CalledProcessError(128, "git")
        error.stderr = None

        with patch.object(manager, '_enforce_capacity_limit'), \
             patch.object(manager, '_run_git', side_effect=error):
            with pytest.raises(RuntimeError):
                manager.create_worktree(
                    base_repo=base_repo,
                    branch_name="feature/test",
                    agent_id="engineer",
                    task_id="task-none-stderr",
                    owner_repo="owner/repo",
                )


class TestCreateWorktreeStartPoint:
    """Tests for create_worktree() with start_point parameter."""

    def test_uses_start_point_when_available_on_remote(self, tmp_path):
        """New branch is based on origin/<start_point> when it exists on remote."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        base_repo = tmp_path / "base"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()

        git_calls = []

        def mock_run_git(args, cwd, timeout=30):
            git_calls.append(args)
            return MagicMock()

        def mock_remote_branch_exists(repo, branch):
            return branch == "agent/engineer/ME-1-abc"

        with patch.object(manager, '_enforce_capacity_limit'), \
             patch.object(manager, '_run_git', side_effect=mock_run_git), \
             patch.object(manager, '_branch_exists', return_value=False), \
             patch.object(manager, '_remote_branch_exists', side_effect=mock_remote_branch_exists):
            manager.create_worktree(
                base_repo=base_repo,
                branch_name="agent/architect/ME-1-def12345",
                agent_id="architect",
                task_id="task-review",
                owner_repo="owner/repo",
                start_point="agent/engineer/ME-1-abc",
            )

        # The worktree add command should use origin/<start_point> as base
        worktree_add = [c for c in git_calls if c[:2] == ["worktree", "add"]]
        assert len(worktree_add) == 1
        assert "origin/agent/engineer/ME-1-abc" in worktree_add[0]

    def test_falls_back_to_default_branch_when_start_point_not_on_remote(self, tmp_path):
        """Falls back to default branch when start_point isn't pushed yet."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        base_repo = tmp_path / "base"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()

        git_calls = []

        def mock_run_git(args, cwd, timeout=30):
            git_calls.append(args)
            return MagicMock()

        with patch.object(manager, '_enforce_capacity_limit'), \
             patch.object(manager, '_run_git', side_effect=mock_run_git), \
             patch.object(manager, '_branch_exists', return_value=False), \
             patch.object(manager, '_remote_branch_exists', return_value=False), \
             patch.object(manager, '_get_default_branch', return_value="main"):
            manager.create_worktree(
                base_repo=base_repo,
                branch_name="agent/architect/ME-1-def12345",
                agent_id="architect",
                task_id="task-review-fallback",
                owner_repo="owner/repo",
                start_point="agent/engineer/ME-1-abc",
            )

        worktree_add = [c for c in git_calls if c[:2] == ["worktree", "add"]]
        assert len(worktree_add) == 1
        assert "origin/main" in worktree_add[0]

    def test_no_start_point_uses_default_branch(self, tmp_path):
        """Without start_point, new branch is based on default branch (existing behavior)."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        base_repo = tmp_path / "base"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()

        git_calls = []

        def mock_run_git(args, cwd, timeout=30):
            git_calls.append(args)
            return MagicMock()

        with patch.object(manager, '_enforce_capacity_limit'), \
             patch.object(manager, '_run_git', side_effect=mock_run_git), \
             patch.object(manager, '_branch_exists', return_value=False), \
             patch.object(manager, '_get_default_branch', return_value="main"):
            manager.create_worktree(
                base_repo=base_repo,
                branch_name="agent/engineer/ME-2-xyz",
                agent_id="engineer",
                task_id="task-normal",
                owner_repo="owner/repo",
            )

        worktree_add = [c for c in git_calls if c[:2] == ["worktree", "add"]]
        assert len(worktree_add) == 1
        assert "origin/main" in worktree_add[0]


class TestRemoteBranchExists:
    """Tests for _remote_branch_exists — remote-only branch check."""

    @pytest.fixture
    def manager(self, tmp_path):
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        return WorktreeManager(config=config)

    def test_returns_true_when_remote_ref_exists(self, manager, tmp_path):
        """origin/<branch> resolves → True."""
        with patch("agent_framework.workspace.worktree_manager.run_git_command") as mock_git:
            mock_git.return_value = MagicMock(returncode=0)
            assert manager._remote_branch_exists(tmp_path, "feature/test") is True
            mock_git.assert_called_once_with(
                ["rev-parse", "--verify", "origin/feature/test"],
                cwd=tmp_path,
                check=False,
                timeout=10,
            )

    def test_returns_false_when_remote_ref_missing(self, manager, tmp_path):
        """origin/<branch> not found → False."""
        with patch("agent_framework.workspace.worktree_manager.run_git_command") as mock_git:
            mock_git.return_value = MagicMock(returncode=128)
            assert manager._remote_branch_exists(tmp_path, "unpushed-branch") is False

    def test_returns_false_on_timeout(self, manager, tmp_path):
        """Timeout → False (graceful degradation)."""
        with patch("agent_framework.workspace.worktree_manager.run_git_command") as mock_git:
            mock_git.side_effect = subprocess.TimeoutExpired("git", 10)
            assert manager._remote_branch_exists(tmp_path, "slow-branch") is False


class TestBranchExists:
    """Tests for _branch_exists — local-then-remote branch check."""

    @pytest.fixture
    def manager(self, tmp_path):
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        return WorktreeManager(config=config)

    def test_returns_true_when_local_ref_exists(self, manager, tmp_path):
        """Local rev-parse succeeds → True without checking remote."""
        with patch("agent_framework.workspace.worktree_manager.run_git_command") as mock_git:
            mock_git.return_value = MagicMock(returncode=0)
            assert manager._branch_exists(tmp_path, "feature/local") is True
            mock_git.assert_called_once()

    def test_falls_through_to_remote_on_local_miss(self, manager, tmp_path):
        """Local rev-parse fails → delegates to _remote_branch_exists."""
        with patch("agent_framework.workspace.worktree_manager.run_git_command") as mock_git:
            # First call (local) fails, second call (remote) succeeds
            mock_git.side_effect = [
                MagicMock(returncode=128),
                MagicMock(returncode=0),
            ]
            assert manager._branch_exists(tmp_path, "feature/remote-only") is True
            assert mock_git.call_count == 2

    def test_returns_false_when_both_miss(self, manager, tmp_path):
        """Neither local nor remote ref exists → False."""
        with patch("agent_framework.workspace.worktree_manager.run_git_command") as mock_git:
            mock_git.return_value = MagicMock(returncode=128)
            assert manager._branch_exists(tmp_path, "nonexistent") is False

    def test_returns_false_on_timeout(self, manager, tmp_path):
        """Timeout → False immediately (no redundant remote probe)."""
        with patch("agent_framework.workspace.worktree_manager.run_git_command") as mock_git:
            mock_git.side_effect = subprocess.TimeoutExpired("git", 10)
            assert manager._branch_exists(tmp_path, "slow-branch") is False
            mock_git.assert_called_once()


class TestGetDefaultBranch:
    """Tests for _get_default_branch — symbolic-ref with fallback probing."""

    @pytest.fixture
    def manager(self, tmp_path):
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        return WorktreeManager(config=config)

    def test_returns_branch_from_symbolic_ref(self, manager, tmp_path):
        """symbolic-ref succeeds → parse branch name from output."""
        with patch("agent_framework.workspace.worktree_manager.run_git_command") as mock_git:
            mock_git.return_value = MagicMock(
                returncode=0, stdout="refs/remotes/origin/main\n"
            )
            assert manager._get_default_branch(tmp_path) == "main"
            mock_git.assert_called_once()

    def test_falls_back_to_probe_on_symbolic_ref_failure(self, manager, tmp_path):
        """symbolic-ref fails → probes origin/main, origin/master."""
        with patch("agent_framework.workspace.worktree_manager.run_git_command") as mock_git:
            mock_git.side_effect = [
                MagicMock(returncode=1),   # symbolic-ref fails
                MagicMock(returncode=0),   # origin/main exists
            ]
            assert manager._get_default_branch(tmp_path) == "main"

    def test_returns_master_when_main_missing(self, manager, tmp_path):
        """origin/main missing but origin/master exists → 'master'."""
        with patch("agent_framework.workspace.worktree_manager.run_git_command") as mock_git:
            mock_git.side_effect = [
                MagicMock(returncode=1),     # symbolic-ref fails
                MagicMock(returncode=128),   # origin/main missing
                MagicMock(returncode=0),     # origin/master exists
            ]
            assert manager._get_default_branch(tmp_path) == "master"

    def test_defaults_to_main_when_all_fail(self, manager, tmp_path):
        """All probes fail → falls back to 'main'."""
        with patch("agent_framework.workspace.worktree_manager.run_git_command") as mock_git:
            mock_git.return_value = MagicMock(returncode=128)
            assert manager._get_default_branch(tmp_path) == "main"

    def test_handles_timeout_gracefully(self, manager, tmp_path):
        """Timeout on symbolic-ref → still probes fallback branches."""
        with patch("agent_framework.workspace.worktree_manager.run_git_command") as mock_git:
            mock_git.side_effect = [
                subprocess.TimeoutExpired("git", 10),  # symbolic-ref times out
                MagicMock(returncode=0),                # origin/main exists
            ]
            assert manager._get_default_branch(tmp_path) == "main"


class TestStartPointUsesRemoteCheck:
    """Verify create_worktree uses _remote_branch_exists for start_point."""

    def test_start_point_calls_remote_branch_exists(self, tmp_path):
        """start_point verification goes through _remote_branch_exists, not _branch_exists."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        base_repo = tmp_path / "base"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()

        with patch.object(manager, '_enforce_capacity_limit'), \
             patch.object(manager, '_run_git'), \
             patch.object(manager, '_remote_branch_exists', return_value=True) as mock_remote, \
             patch.object(manager, '_branch_exists', return_value=False) as mock_local:
            manager.create_worktree(
                base_repo=base_repo,
                branch_name="agent/architect/ME-1-review",
                agent_id="architect",
                task_id="task-review",
                owner_repo="owner/repo",
                start_point="agent/engineer/ME-1-impl",
            )

        mock_remote.assert_called_once_with(base_repo, "agent/engineer/ME-1-impl")
        # _branch_exists should only be called for branch_name existence check, not start_point
        for call_args in mock_local.call_args_list:
            assert call_args[0][1] != "agent/engineer/ME-1-impl"


class TestFindWorktreeAgent:
    """Tests for _find_worktree_agent helper."""

    @pytest.fixture
    def manager(self, tmp_path):
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        return WorktreeManager(config=config)

    def test_finds_agent_for_registered_path(self, manager, tmp_path):
        """Returns agent_id when path matches a registry entry."""
        wt_path = tmp_path / "wt-engineer"
        manager._registry["eng-key"] = WorktreeInfo(
            path=str(wt_path), branch="b1", agent_id="engineer", task_id="t1",
            created_at="2025-01-01T00:00:00", last_accessed="2025-01-01T00:00:00",
            base_repo=str(tmp_path),
        )
        assert manager._find_worktree_agent(wt_path) == "engineer"

    def test_returns_none_for_unknown_path(self, manager, tmp_path):
        """Returns None when path not in registry."""
        assert manager._find_worktree_agent(tmp_path / "unknown") is None

    def test_resolves_symlinks(self, manager, tmp_path):
        """Resolves paths before comparison so symlinks don't break lookup."""
        real_path = tmp_path / "real-wt"
        real_path.mkdir()
        link_path = tmp_path / "linked-wt"
        link_path.symlink_to(real_path)
        manager._registry["key"] = WorktreeInfo(
            path=str(real_path), branch="b1", agent_id="architect", task_id="t1",
            created_at="2025-01-01T00:00:00", last_accessed="2025-01-01T00:00:00",
            base_repo=str(tmp_path),
        )
        assert manager._find_worktree_agent(link_path) == "architect"


class TestBranchConflictOwnershipValidation:
    """Tests for cross-agent ownership check in branch conflict handler."""

    def test_rejects_cross_agent_conflict(self, tmp_path):
        """Branch checked out by another agent raises RuntimeError."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        base_repo = tmp_path / "base"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()
        existing_wt = tmp_path / "existing-worktree"
        existing_wt.mkdir()

        # Register the existing worktree as owned by engineer
        manager._registry["eng-key"] = WorktreeInfo(
            path=str(existing_wt), branch="feature/shared", agent_id="engineer",
            task_id="t-eng", created_at="2025-01-01T00:00:00",
            last_accessed="2025-01-01T00:00:00", base_repo=str(base_repo),
        )

        conflict_msg = f"fatal: 'feature/shared' is already checked out at '{existing_wt}'"
        error = subprocess.CalledProcessError(128, "git")
        error.stderr = conflict_msg.encode()

        with patch.object(manager, '_enforce_capacity_limit'), \
             patch.object(manager, '_run_git', side_effect=error):
            with pytest.raises(RuntimeError, match="locked by agent 'engineer'"):
                manager.create_worktree(
                    base_repo=base_repo,
                    branch_name="feature/shared",
                    agent_id="architect",
                    task_id="task-conflict-cross",
                    owner_repo="owner/repo",
                )

    def test_allows_same_agent_conflict(self, tmp_path):
        """Branch checked out by the same agent is reused (existing behavior)."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        base_repo = tmp_path / "base"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()
        existing_wt = tmp_path / "existing-worktree"
        existing_wt.mkdir()

        # Register the existing worktree as owned by engineer
        manager._registry["eng-key"] = WorktreeInfo(
            path=str(existing_wt), branch="feature/mine", agent_id="engineer",
            task_id="t-eng", created_at="2025-01-01T00:00:00",
            last_accessed="2025-01-01T00:00:00", base_repo=str(base_repo),
        )

        conflict_msg = f"fatal: 'feature/mine' is already checked out at '{existing_wt}'"
        error = subprocess.CalledProcessError(128, "git")
        error.stderr = conflict_msg.encode()

        with patch.object(manager, '_enforce_capacity_limit'), \
             patch.object(manager, '_run_git', side_effect=error):
            result = manager.create_worktree(
                base_repo=base_repo,
                branch_name="feature/mine",
                agent_id="engineer",
                task_id="task-same-agent",
                owner_repo="owner/repo",
            )

        assert result == existing_wt  # same-agent reuse

    def test_allows_unknown_owner_conflict(self, tmp_path):
        """Branch conflict with no matching registry entry is still reused (graceful fallback)."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        base_repo = tmp_path / "base"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()
        existing_wt = tmp_path / "orphan-worktree"
        existing_wt.mkdir()

        conflict_msg = f"fatal: 'feature/orphan' is already checked out at '{existing_wt}'"
        error = subprocess.CalledProcessError(128, "git")
        error.stderr = conflict_msg.encode()

        with patch.object(manager, '_enforce_capacity_limit'), \
             patch.object(manager, '_run_git', side_effect=error):
            result = manager.create_worktree(
                base_repo=base_repo,
                branch_name="feature/orphan",
                agent_id="engineer",
                task_id="task-orphan",
                owner_repo="owner/repo",
            )

        assert result == existing_wt  # orphan reuse


class TestHasUnpushedCommitsFallback:
    """Tests for has_unpushed_commits() when no tracking branch is set."""

    @pytest.fixture
    def manager(self, tmp_path):
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        return WorktreeManager(config=config)

    @patch("subprocess.run")
    def test_detects_local_only_commits_via_git_log(self, mock_run, manager, tmp_path):
        """Branches with commits not on any remote are detected as unpushed."""
        rev_list = MagicMock(returncode=128, stdout="", stderr="no upstream")
        log_result = MagicMock(returncode=0, stdout="abc1234 Add feature\n")
        mock_run.side_effect = [rev_list, log_result]

        assert manager.has_unpushed_commits(tmp_path) is True
        log_call = mock_run.call_args_list[1]
        assert "--not" in log_call[0][0]
        assert "--remotes" in log_call[0][0]

    @patch("subprocess.run")
    def test_falls_through_to_porcelain_when_no_local_commits(self, mock_run, manager, tmp_path):
        """When git log finds nothing, falls through to status --porcelain."""
        rev_list = MagicMock(returncode=128, stdout="", stderr="no upstream")
        log_result = MagicMock(returncode=0, stdout="")
        porcelain = MagicMock(returncode=0, stdout=" M file.py\n")
        mock_run.side_effect = [rev_list, log_result, porcelain]

        assert manager.has_unpushed_commits(tmp_path) is True

    @patch("subprocess.run")
    def test_returns_false_when_clean_and_no_local_commits(self, mock_run, manager, tmp_path):
        """Clean worktree with no local-only commits returns False."""
        rev_list = MagicMock(returncode=128, stdout="", stderr="no upstream")
        log_result = MagicMock(returncode=0, stdout="")
        porcelain = MagicMock(returncode=0, stdout="")
        mock_run.side_effect = [rev_list, log_result, porcelain]

        assert manager.has_unpushed_commits(tmp_path) is False

    @patch("subprocess.run")
    def test_tracking_branch_path_still_works(self, mock_run, manager, tmp_path):
        """Normal path (tracking branch exists) is unchanged."""
        rev_list = MagicMock(returncode=0, stdout="3\n")
        mock_run.return_value = rev_list

        assert manager.has_unpushed_commits(tmp_path) is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_tracking_branch_zero_ahead(self, mock_run, manager, tmp_path):
        """Zero commits ahead with tracking branch returns False."""
        rev_list = MagicMock(returncode=0, stdout="0\n")
        mock_run.return_value = rev_list

        assert manager.has_unpushed_commits(tmp_path) is False


class TestCleanupSafetyChecks:
    """Tests for safety checks that prevent deleting worktrees with unsaved work."""

    def _make_unregistered_worktree(self, tmp_path):
        """Set up a worktree root with one unregistered worktree directory."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        # Create owner/repo/worktree structure with .git marker
        owner_dir = config.root / "owner"
        repo_dir = owner_dir / "repo"
        wt_dir = repo_dir / "orphan-wt"
        wt_dir.mkdir(parents=True)
        (wt_dir / ".git").write_text("gitdir: /some/base/.git/worktrees/orphan-wt")

        return manager, wt_dir

    def test_cleanup_unregistered_skips_worktree_with_uncommitted_changes(self, tmp_path):
        """Unregistered worktree with uncommitted changes is not deleted."""
        manager, wt_dir = self._make_unregistered_worktree(tmp_path)

        with patch.object(manager, 'has_uncommitted_changes', return_value=True), \
             patch.object(manager, 'has_unpushed_commits', return_value=False), \
             patch.object(manager, '_remove_worktree_directory') as mock_remove:
            removed = manager._cleanup_unregistered_worktrees()

        assert removed == 0
        mock_remove.assert_not_called()

    def test_cleanup_unregistered_skips_worktree_with_unpushed_commits(self, tmp_path):
        """Unregistered worktree with unpushed commits is not deleted."""
        manager, wt_dir = self._make_unregistered_worktree(tmp_path)

        with patch.object(manager, 'has_uncommitted_changes', return_value=False), \
             patch.object(manager, 'has_unpushed_commits', return_value=True), \
             patch.object(manager, '_remove_worktree_directory') as mock_remove:
            removed = manager._cleanup_unregistered_worktrees()

        assert removed == 0
        mock_remove.assert_not_called()

    def test_cleanup_unregistered_removes_clean_worktree(self, tmp_path):
        """Unregistered worktree with no unsaved work is removed (with force=False)."""
        manager, wt_dir = self._make_unregistered_worktree(tmp_path)

        with patch.object(manager, 'has_uncommitted_changes', return_value=False), \
             patch.object(manager, 'has_unpushed_commits', return_value=False), \
             patch.object(manager, '_remove_worktree_directory', return_value=True) as mock_remove:
            removed = manager._cleanup_unregistered_worktrees()

        assert removed == 1
        # Verify force=False — we confirmed no uncommitted changes
        mock_remove.assert_called_once()
        _, kwargs = mock_remove.call_args
        assert kwargs.get("force") is False

    def test_cleanup_orphaned_skips_stale_worktree_with_unsaved_work(self, tmp_path):
        """Stale registered worktree with uncommitted changes survives cleanup."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees", max_age_hours=0)
        manager = WorktreeManager(config=config)

        wt_path = tmp_path / "wt-stale"
        wt_path.mkdir()

        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        manager._registry["stale-key"] = WorktreeInfo(
            path=str(wt_path), branch="feature/stale", agent_id="engineer",
            task_id="task-stale", created_at=old_ts, last_accessed=old_ts,
            base_repo=str(tmp_path),
        )

        with patch.object(manager, 'has_uncommitted_changes', return_value=True), \
             patch.object(manager, 'has_unpushed_commits', return_value=False), \
             patch.object(manager, '_remove_worktree_directory') as mock_remove, \
             patch.object(manager, '_cleanup_unregistered_worktrees', return_value=0), \
             patch.object(manager, '_load_registry'):
            result = manager.cleanup_orphaned_worktrees()

        assert "stale-key" in manager._registry
        mock_remove.assert_not_called()
        assert result["registered"] == 0


class TestTouchWorktree:
    """Tests for touch_worktree() timestamp update."""

    def test_touch_worktree_updates_last_accessed(self, tmp_path):
        """Timestamp updated, active flag unchanged."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        wt_path = tmp_path / "wt"
        old_ts = "2025-01-01T00:00:00+00:00"
        manager._registry["key1"] = WorktreeInfo(
            path=str(wt_path), branch="b1", agent_id="a1", task_id="t1",
            created_at=old_ts, last_accessed=old_ts,
            base_repo="/base", active=True,
        )

        with patch.object(manager, '_save_registry') as mock_save:
            manager.touch_worktree(wt_path)

        assert manager._registry["key1"].last_accessed != old_ts
        assert manager._registry["key1"].active is True
        mock_save.assert_called_once()

    def test_touch_worktree_noop_for_unknown_path(self, tmp_path):
        """No crash for unregistered path, no save triggered."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        with patch.object(manager, '_save_registry') as mock_save:
            manager.touch_worktree(tmp_path / "nonexistent")

        mock_save.assert_not_called()


class TestCreateWorktreePrunesBeforeAdd:
    """Tests that create_worktree prunes stale refs before git worktree add."""

    def test_create_worktree_prunes_before_add(self, tmp_path):
        """git worktree prune is called during creation when path doesn't exist."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        base_repo = tmp_path / "base"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()

        git_calls = []

        def mock_run_git(args, cwd, timeout=30):
            git_calls.append(args)
            return MagicMock()

        with patch.object(manager, '_enforce_capacity_limit'), \
             patch.object(manager, '_run_git', side_effect=mock_run_git), \
             patch.object(manager, '_branch_exists', return_value=False), \
             patch.object(manager, '_get_default_branch', return_value="main"):
            manager.create_worktree(
                base_repo=base_repo,
                branch_name="agent/engineer/test-prune",
                agent_id="engineer",
                task_id="task-prune",
                owner_repo="owner/repo",
            )

        # Verify prune happens before worktree add
        prune_idx = next(i for i, c in enumerate(git_calls) if c == ["worktree", "prune"])
        add_idx = next(i for i, c in enumerate(git_calls) if c[:2] == ["worktree", "add"])
        assert prune_idx < add_idx


class TestForceRemovalPrunesAfterRmtree:
    """Tests that force-removal fallback prunes after shutil.rmtree."""

    def test_force_removal_prunes_after_rmtree(self, tmp_path):
        """git worktree prune called after shutil.rmtree fallback."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        wt_path = tmp_path / "wt-to-remove"
        wt_path.mkdir()
        base_repo = tmp_path / "base"
        base_repo.mkdir()

        git_calls = []

        def mock_run_git(args, cwd, timeout=30):
            git_calls.append(args)
            # The initial "git worktree remove" fails, triggering force fallback
            if args[0] == "worktree" and args[1] == "remove":
                raise subprocess.CalledProcessError(1, "git")
            return MagicMock()

        with patch.object(manager, '_run_git', side_effect=mock_run_git):
            result = manager._remove_worktree_directory(wt_path, base_repo, force=True)

        assert result is True
        # Verify prune was called after the failed remove attempt
        assert ["worktree", "prune"] in git_calls


class TestProtectedAgentIds:
    """Tests for protected_agent_ids guard across cleanup methods."""

    def test_cleanup_orphaned_reloads_registry(self, tmp_path):
        """cleanup_orphaned_worktrees calls _load_registry before processing."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        with patch.object(manager, '_load_registry') as mock_load, \
             patch.object(manager, '_cleanup_unregistered_worktrees', return_value=0):
            manager.cleanup_orphaned_worktrees()

        mock_load.assert_called_once()

    def test_cleanup_orphaned_skips_protected_agent(self, tmp_path):
        """Worktrees belonging to protected agents are never evicted."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees", max_age_hours=0)
        manager = WorktreeManager(config=config)

        wt_path = tmp_path / "wt-protected"
        wt_path.mkdir()

        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        manager._registry["eng-key"] = WorktreeInfo(
            path=str(wt_path), branch="b1", agent_id="engineer", task_id="t1",
            created_at=old_ts, last_accessed=old_ts, base_repo=str(tmp_path),
        )

        with patch.object(manager, '_load_registry'), \
             patch.object(manager, '_remove_worktree_directory') as mock_remove, \
             patch.object(manager, '_cleanup_unregistered_worktrees', return_value=0):
            result = manager.cleanup_orphaned_worktrees(
                protected_agent_ids={"engineer"},
            )

        assert "eng-key" in manager._registry
        mock_remove.assert_not_called()
        assert result["registered"] == 0

    def test_cleanup_orphaned_evicts_unprotected_agent(self, tmp_path):
        """Worktrees belonging to unprotected agents are still evicted normally."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees", max_age_hours=0)
        manager = WorktreeManager(config=config)

        wt_path = tmp_path / "wt-unprotected"
        wt_path.mkdir()

        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        manager._registry["qa-key"] = WorktreeInfo(
            path=str(wt_path), branch="b1", agent_id="qa", task_id="t1",
            created_at=old_ts, last_accessed=old_ts, base_repo=str(tmp_path),
        )

        with patch.object(manager, '_load_registry'), \
             patch.object(manager, 'has_uncommitted_changes', return_value=False), \
             patch.object(manager, 'has_unpushed_commits', return_value=False), \
             patch.object(manager, '_remove_worktree_directory', return_value=True), \
             patch.object(manager, '_cleanup_unregistered_worktrees', return_value=0):
            result = manager.cleanup_orphaned_worktrees(
                protected_agent_ids={"engineer"},
            )

        assert "qa-key" not in manager._registry
        assert result["registered"] == 1

    def test_cleanup_unregistered_skips_protected_agent_prefix(self, tmp_path):
        """Unregistered worktrees whose dir name starts with a protected agent ID are skipped."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        # Create owner/repo/engineer-ME-123 structure with .git marker
        wt_dir = config.root / "owner" / "repo" / "engineer-ME-123"
        wt_dir.mkdir(parents=True)
        (wt_dir / ".git").write_text("gitdir: /some/base/.git/worktrees/engineer-ME-123")

        with patch.object(manager, '_remove_worktree_directory') as mock_remove:
            removed = manager._cleanup_unregistered_worktrees(
                protected_agent_ids={"engineer"},
            )

        assert removed == 0
        mock_remove.assert_not_called()

    def test_cleanup_unregistered_protects_replica_worktrees(self, tmp_path):
        """Protecting 'engineer' also protects replica 'engineer-2' worktrees (same role)."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        # "engineer-2-ME-1" belongs to replica agent "engineer-2"
        wt_dir = config.root / "owner" / "repo" / "engineer-2-ME-1"
        wt_dir.mkdir(parents=True)
        (wt_dir / ".git").write_text("gitdir: /some/base/.git/worktrees/engineer-2-ME-1")

        with patch.object(manager, '_remove_worktree_directory') as mock_remove:
            removed = manager._cleanup_unregistered_worktrees(
                protected_agent_ids={"engineer"},
            )

        # Replica worktrees share the base agent ID prefix, so they're protected
        assert removed == 0
        mock_remove.assert_not_called()

    def test_cleanup_unregistered_does_not_match_different_agent(self, tmp_path):
        """Protecting 'qa' does not protect 'architect' worktrees."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        wt_dir = config.root / "owner" / "repo" / "architect-ME-1"
        wt_dir.mkdir(parents=True)
        (wt_dir / ".git").write_text("gitdir: /some/base/.git/worktrees/architect-ME-1")

        with patch.object(manager, 'has_uncommitted_changes', return_value=False), \
             patch.object(manager, 'has_unpushed_commits', return_value=False), \
             patch.object(manager, '_remove_worktree_directory', return_value=True) as mock_remove:
            removed = manager._cleanup_unregistered_worktrees(
                protected_agent_ids={"qa"},
            )

        assert removed == 1
        mock_remove.assert_called_once()

    def test_enforce_capacity_skips_protected_agent(self, tmp_path):
        """Protected agents' worktrees are not evicted even when over capacity."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees", max_worktrees=1)
        manager = WorktreeManager(config=config)

        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(minutes=5)).isoformat()

        # Two worktrees — over the limit of 1
        manager._registry["eng-key"] = WorktreeInfo(
            path=str(tmp_path / "wt1"), branch="b1", agent_id="engineer", task_id="t1",
            created_at=old_ts, last_accessed=old_ts, base_repo=str(tmp_path),
        )
        manager._registry["qa-key"] = WorktreeInfo(
            path=str(tmp_path / "wt2"), branch="b2", agent_id="qa", task_id="t2",
            created_at=old_ts, last_accessed=old_ts, base_repo=str(tmp_path),
        )

        with patch.object(manager, '_load_registry'), \
             patch.object(manager, '_remove_worktree_directory', return_value=True):
            manager._enforce_capacity_limit(protected_agent_ids={"engineer"})

        # Engineer's worktree survives, QA's gets evicted
        assert "eng-key" in manager._registry
        assert "qa-key" not in manager._registry


class TestSwitchWorktreeBranch:
    """Tests for _switch_worktree_branch()."""

    @pytest.fixture
    def manager(self, tmp_path):
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        return WorktreeManager(config=config)

    def test_switches_to_existing_branch(self, manager, tmp_path):
        """Checks out an existing branch via git checkout."""
        base_repo = tmp_path / "base"
        base_repo.mkdir()
        wt_path = tmp_path / "wt"
        wt_path.mkdir()

        git_calls = []

        def mock_run_git(args, cwd, timeout=30):
            git_calls.append(args)
            return MagicMock()

        with patch.object(manager, '_run_git', side_effect=mock_run_git), \
             patch.object(manager, '_branch_exists', return_value=True):
            result = manager._switch_worktree_branch(
                wt_path, "agent/engineer/ME-1", base_repo
            )

        assert result is True
        checkout_calls = [c for c in git_calls if c[0] == "checkout"]
        assert checkout_calls == [["checkout", "agent/engineer/ME-1"]]

    def test_creates_branch_from_remote_start_point(self, manager, tmp_path):
        """Creates new branch from origin/<start_point> when branch doesn't exist."""
        base_repo = tmp_path / "base"
        base_repo.mkdir()
        wt_path = tmp_path / "wt"
        wt_path.mkdir()

        git_calls = []

        def mock_run_git(args, cwd, timeout=30):
            git_calls.append(args)
            return MagicMock()

        with patch.object(manager, '_run_git', side_effect=mock_run_git), \
             patch.object(manager, '_branch_exists', return_value=False), \
             patch.object(manager, '_remote_branch_exists', return_value=True):
            result = manager._switch_worktree_branch(
                wt_path, "agent/architect/ME-1-review", base_repo,
                start_point="agent/engineer/ME-1-impl",
            )

        assert result is True
        checkout_calls = [c for c in git_calls if c[0] == "checkout"]
        assert checkout_calls == [
            ["checkout", "-b", "agent/architect/ME-1-review", "origin/agent/engineer/ME-1-impl"]
        ]

    def test_falls_back_to_default_when_start_point_not_on_remote(self, manager, tmp_path):
        """Falls back to default branch when start_point isn't pushed."""
        base_repo = tmp_path / "base"
        base_repo.mkdir()
        wt_path = tmp_path / "wt"
        wt_path.mkdir()

        git_calls = []

        def mock_run_git(args, cwd, timeout=30):
            git_calls.append(args)
            return MagicMock()

        with patch.object(manager, '_run_git', side_effect=mock_run_git), \
             patch.object(manager, '_branch_exists', return_value=False), \
             patch.object(manager, '_remote_branch_exists', return_value=False), \
             patch.object(manager, '_get_default_branch', return_value="main"):
            result = manager._switch_worktree_branch(
                wt_path, "agent/architect/ME-1-review", base_repo,
                start_point="agent/engineer/ME-1-impl",
            )

        assert result is True
        checkout_calls = [c for c in git_calls if c[0] == "checkout"]
        assert checkout_calls == [
            ["checkout", "-b", "agent/architect/ME-1-review", "origin/main"]
        ]

    def test_falls_back_to_default_when_no_start_point(self, manager, tmp_path):
        """Uses default branch when no start_point given."""
        base_repo = tmp_path / "base"
        base_repo.mkdir()
        wt_path = tmp_path / "wt"
        wt_path.mkdir()

        git_calls = []

        def mock_run_git(args, cwd, timeout=30):
            git_calls.append(args)
            return MagicMock()

        with patch.object(manager, '_run_git', side_effect=mock_run_git), \
             patch.object(manager, '_branch_exists', return_value=False), \
             patch.object(manager, '_get_default_branch', return_value="main"):
            result = manager._switch_worktree_branch(
                wt_path, "new-branch", base_repo
            )

        assert result is True
        checkout_calls = [c for c in git_calls if c[0] == "checkout"]
        assert checkout_calls == [["checkout", "-b", "new-branch", "origin/main"]]

    def test_fetch_failure_is_nonfatal(self, manager, tmp_path):
        """Fetch failure doesn't prevent the branch switch."""
        base_repo = tmp_path / "base"
        base_repo.mkdir()
        wt_path = tmp_path / "wt"
        wt_path.mkdir()

        call_count = {"n": 0}

        def mock_run_git(args, cwd, timeout=30):
            call_count["n"] += 1
            if args == ["fetch", "origin"]:
                raise subprocess.CalledProcessError(1, "git")
            return MagicMock()

        with patch.object(manager, '_run_git', side_effect=mock_run_git), \
             patch.object(manager, '_branch_exists', return_value=True):
            result = manager._switch_worktree_branch(
                wt_path, "existing-branch", base_repo
            )

        assert result is True
        assert call_count["n"] == 2  # fetch (failed) + checkout (success)

    def test_returns_false_on_checkout_failure(self, manager, tmp_path):
        """Checkout failure returns False (not raised to caller)."""
        base_repo = tmp_path / "base"
        base_repo.mkdir()
        wt_path = tmp_path / "wt"
        wt_path.mkdir()

        def mock_run_git(args, cwd, timeout=30):
            if args[0] == "checkout":
                raise subprocess.CalledProcessError(1, "git")
            return MagicMock()

        with patch.object(manager, '_run_git', side_effect=mock_run_git), \
             patch.object(manager, '_branch_exists', return_value=True):
            result = manager._switch_worktree_branch(
                wt_path, "bad-branch", base_repo
            )

        assert result is False


class TestCreateWorktreeBranchMismatch:
    """Tests for branch mismatch handling in create_worktree() reuse block."""

    def test_same_branch_reuses_without_switch(self, tmp_path):
        """Matching branch reuses worktree without calling _switch_worktree_branch."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        base_repo = tmp_path / "base"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()

        # Pre-create the worktree path and registry entry
        wt_path = manager._get_worktree_path("owner/repo", "architect", "jira-ME-1-123")
        wt_path.mkdir(parents=True)
        key = manager._get_worktree_key("architect", "jira-ME-1-123")
        manager._registry[key] = WorktreeInfo(
            path=str(wt_path), branch="agent/architect/ME-1-plan",
            agent_id="architect", task_id="jira-ME-1-123",
            created_at="2025-01-01T00:00:00+00:00",
            last_accessed="2025-01-01T00:00:00+00:00",
            base_repo=str(base_repo),
        )

        with patch.object(manager, '_enforce_capacity_limit'), \
             patch.object(manager, '_switch_worktree_branch') as mock_switch:
            result = manager.create_worktree(
                base_repo=base_repo,
                branch_name="agent/architect/ME-1-plan",
                agent_id="architect",
                task_id="jira-ME-1-123",
                owner_repo="owner/repo",
            )

        assert result == wt_path
        mock_switch.assert_not_called()

    def test_different_branch_triggers_switch_and_updates_registry(self, tmp_path):
        """Branch mismatch calls _switch_worktree_branch and updates registry on success."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        base_repo = tmp_path / "base"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()

        wt_path = manager._get_worktree_path("owner/repo", "architect", "jira-ME-1-123")
        wt_path.mkdir(parents=True)
        key = manager._get_worktree_key("architect", "jira-ME-1-123")
        manager._registry[key] = WorktreeInfo(
            path=str(wt_path), branch="agent/architect/ME-1-plan",
            agent_id="architect", task_id="jira-ME-1-123",
            created_at="2025-01-01T00:00:00+00:00",
            last_accessed="2025-01-01T00:00:00+00:00",
            base_repo=str(base_repo),
        )

        with patch.object(manager, '_enforce_capacity_limit'), \
             patch.object(manager, '_switch_worktree_branch', return_value=True) as mock_switch:
            result = manager.create_worktree(
                base_repo=base_repo,
                branch_name="agent/architect/ME-1-review",
                agent_id="architect",
                task_id="jira-ME-1-123",
                owner_repo="owner/repo",
                start_point="agent/engineer/ME-1-impl",
            )

        assert result == wt_path
        mock_switch.assert_called_once_with(
            wt_path, "agent/architect/ME-1-review", base_repo, "agent/engineer/ME-1-impl"
        )
        assert manager._registry[key].branch == "agent/architect/ME-1-review"
        assert manager._registry[key].active is True

    def test_switch_failure_removes_worktree_and_recreates(self, tmp_path):
        """Failed branch switch removes worktree and falls through to fresh creation."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        base_repo = tmp_path / "base"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()

        wt_path = manager._get_worktree_path("owner/repo", "architect", "jira-ME-1-123")
        wt_path.mkdir(parents=True)
        key = manager._get_worktree_key("architect", "jira-ME-1-123")
        manager._registry[key] = WorktreeInfo(
            path=str(wt_path), branch="agent/architect/ME-1-plan",
            agent_id="architect", task_id="jira-ME-1-123",
            created_at="2025-01-01T00:00:00+00:00",
            last_accessed="2025-01-01T00:00:00+00:00",
            base_repo=str(base_repo),
        )

        git_calls = []

        def mock_run_git(args, cwd, timeout=30):
            git_calls.append(args)
            return MagicMock()

        with patch.object(manager, '_enforce_capacity_limit'), \
             patch.object(manager, '_switch_worktree_branch', return_value=False), \
             patch.object(manager, '_remove_worktree_directory', return_value=True), \
             patch.object(manager, '_run_git', side_effect=mock_run_git), \
             patch.object(manager, '_branch_exists', return_value=False), \
             patch.object(manager, '_get_default_branch', return_value="main"):
            result = manager.create_worktree(
                base_repo=base_repo,
                branch_name="agent/architect/ME-1-review",
                agent_id="architect",
                task_id="jira-ME-1-123",
                owner_repo="owner/repo",
            )

        assert result == wt_path
        # Registry should have the new branch after fresh creation
        assert manager._registry[key].branch == "agent/architect/ME-1-review"

    def test_switch_and_removal_failure_raises(self, tmp_path):
        """When both switch and removal fail, raises RuntimeError."""
        config = WorktreeConfig(enabled=True, root=tmp_path / "worktrees")
        manager = WorktreeManager(config=config)

        base_repo = tmp_path / "base"
        base_repo.mkdir()
        (base_repo / ".git").mkdir()

        wt_path = manager._get_worktree_path("owner/repo", "architect", "jira-ME-1-123")
        wt_path.mkdir(parents=True)
        key = manager._get_worktree_key("architect", "jira-ME-1-123")
        manager._registry[key] = WorktreeInfo(
            path=str(wt_path), branch="agent/architect/ME-1-plan",
            agent_id="architect", task_id="jira-ME-1-123",
            created_at="2025-01-01T00:00:00+00:00",
            last_accessed="2025-01-01T00:00:00+00:00",
            base_repo=str(base_repo),
        )

        with patch.object(manager, '_enforce_capacity_limit'), \
             patch.object(manager, '_switch_worktree_branch', return_value=False), \
             patch.object(manager, '_remove_worktree_directory', return_value=False):
            with pytest.raises(RuntimeError, match="Cannot switch branch and cannot remove"):
                manager.create_worktree(
                    base_repo=base_repo,
                    branch_name="agent/architect/ME-1-review",
                    agent_id="architect",
                    task_id="jira-ME-1-123",
                    owner_repo="owner/repo",
                )
