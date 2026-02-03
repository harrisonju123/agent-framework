"""Unit tests for MultiRepoManager."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from agent_framework.workspace.multi_repo_manager import MultiRepoManager


@pytest.fixture
def mock_token():
    """Provide a mock GitHub token."""
    return "ghp_mock_token_1234567890"


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory."""
    workspace = tmp_path / "workspaces"
    workspace.mkdir()
    return workspace


@pytest.fixture
def manager(temp_workspace, mock_token):
    """Create a MultiRepoManager instance with mocked authentication."""
    with patch('github.Github') as mock_github:
        mock_gh = MagicMock()
        mock_user = MagicMock()
        mock_user.login = "test-user"
        mock_gh.get_user.return_value = mock_user
        mock_github.return_value = mock_gh

        mgr = MultiRepoManager(temp_workspace, mock_token)
        return mgr


class TestInputValidation:
    """Test input validation methods."""

    def test_validate_owner_repo_valid(self, manager):
        """Test valid repository name."""
        result = manager._validate_owner_repo("my-org/my-repo")
        assert result == "my-org/my-repo"

    def test_validate_owner_repo_invalid_format(self, manager):
        """Test invalid repository format."""
        with pytest.raises(ValueError, match="Invalid repository format"):
            manager._validate_owner_repo("not-a-valid-repo")

    def test_validate_owner_repo_path_traversal(self, manager):
        """Test path traversal attempt in repo name."""
        with pytest.raises(ValueError, match="Invalid repository"):
            manager._validate_owner_repo("my-org/../etc/passwd")

    def test_validate_owner_repo_multiple_slashes(self, manager):
        """Test multiple slashes in repo name."""
        with pytest.raises(ValueError, match="Invalid repository"):
            manager._validate_owner_repo("org/repo/extra")

    def test_validate_branch_name_valid(self, manager):
        """Test valid branch name."""
        result = manager._validate_branch_name("feature/my-feature")
        assert result == "feature/my-feature"

    def test_validate_branch_name_empty(self, manager):
        """Test empty branch name."""
        with pytest.raises(ValueError, match="cannot be empty"):
            manager._validate_branch_name("")

    def test_validate_branch_name_invalid_chars(self, manager):
        """Test branch name with invalid characters."""
        with pytest.raises(ValueError, match="Invalid branch name"):
            manager._validate_branch_name("feature/bad name")

    def test_validate_branch_name_path_traversal(self, manager):
        """Test path traversal in branch name."""
        with pytest.raises(ValueError):
            manager._validate_branch_name("feature/../master")

    def test_validate_branch_name_too_long(self, manager):
        """Test branch name exceeding length limit."""
        long_name = "a" * 300
        with pytest.raises(ValueError):
            manager._validate_branch_name(long_name)

    def test_validate_file_path_valid(self, manager):
        """Test valid file path."""
        result = manager._validate_file_path("src/main.py")
        assert result == "src/main.py"

    def test_validate_file_path_absolute(self, manager):
        """Test absolute path rejection."""
        with pytest.raises(ValueError):
            manager._validate_file_path("/etc/passwd")

    def test_validate_file_path_traversal(self, manager):
        """Test path traversal attempt."""
        with pytest.raises(ValueError):
            manager._validate_file_path("../../../etc/passwd")

    def test_validate_commit_message_valid(self, manager):
        """Test valid commit message."""
        result = manager._validate_commit_message("Fix: Update authentication")
        assert result == "Fix: Update authentication"

    def test_validate_commit_message_empty(self, manager):
        """Test empty commit message."""
        with pytest.raises(ValueError):
            manager._validate_commit_message("")

    def test_validate_commit_message_too_long(self, manager):
        """Test commit message exceeding length limit."""
        long_msg = "a" * 15000
        with pytest.raises(ValueError):
            manager._validate_commit_message(long_msg)

    def test_validate_commit_message_removes_control_chars(self, manager):
        """Test control character removal."""
        result = manager._validate_commit_message("Fix\x00test\x01")
        assert "\x00" not in result
        assert "\x01" not in result


class TestRepositoryOperations:
    """Test repository operation methods."""

    @patch('subprocess.run')
    def test_clone_success(self, mock_run, manager):
        """Test successful repository clone."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        repo_path = manager.workspace_root / "test-org" / "test-repo"
        manager._clone("test-org/test-repo", repo_path)

        # Verify git clone was called
        assert mock_run.called
        call_args = mock_run.call_args
        assert "git" in call_args[0][0]
        assert "clone" in call_args[0][0]

        # Verify token not in URL
        assert manager.token not in str(call_args[0][0])

    @patch('subprocess.run')
    def test_clone_timeout(self, mock_run, manager):
        """Test clone timeout handling."""
        mock_run.side_effect = subprocess.TimeoutExpired("git", 300)

        repo_path = manager.workspace_root / "test-org" / "test-repo"
        with pytest.raises(subprocess.TimeoutExpired):
            manager._clone("test-org/test-repo", repo_path)

    @patch('subprocess.run')
    def test_pull_success(self, mock_run, manager):
        """Test successful pull."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        repo_path = manager.workspace_root / "test-org" / "test-repo"
        repo_path.mkdir(parents=True)

        manager._pull(repo_path)

        # Verify git pull was called
        assert mock_run.called
        call_args = mock_run.call_args
        assert "git" in call_args[0][0]
        assert "pull" in call_args[0][0]

    @patch('subprocess.run')
    def test_get_current_branch(self, mock_run, manager):
        """Test getting current branch."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="feature/test-branch\n",
            stderr=""
        )

        result = manager.get_current_branch("test-org/test-repo")
        assert result == "feature/test-branch"

    @patch('subprocess.run')
    def test_get_default_branch(self, mock_run, manager):
        """Test getting default branch."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="refs/heads/main\n",
            stderr=""
        )

        result = manager.get_default_branch("test-org/test-repo")
        assert result == "main"


class TestFileOperations:
    """Test file reading operations."""

    def test_read_files_success(self, manager, tmp_path):
        """Test successful file reading."""
        # Create a mock repository
        repo_path = manager.workspace_root / "test-org" / "test-repo"
        repo_path.mkdir(parents=True)

        # Create test files
        file1 = repo_path / "src" / "main.py"
        file1.parent.mkdir(parents=True)
        file1.write_text("print('hello')")

        file2 = repo_path / "README.md"
        file2.write_text("# Test Repo")

        # Mock ensure_repo to return our test path
        with patch.object(manager, 'ensure_repo', return_value=repo_path):
            result = manager.read_files("test-org/test-repo", ["src/main.py", "README.md"])

        assert "src/main.py" in result
        assert result["src/main.py"] == "print('hello')"
        assert "README.md" in result
        assert result["README.md"] == "# Test Repo"

    def test_read_files_path_traversal(self, manager, tmp_path):
        """Test path traversal prevention in file reading."""
        repo_path = manager.workspace_root / "test-org" / "test-repo"
        repo_path.mkdir(parents=True)

        with patch.object(manager, 'ensure_repo', return_value=repo_path):
            result = manager.read_files("test-org/test-repo", ["../../../etc/passwd"])

        # Should skip invalid path
        assert len(result) == 0

    def test_read_files_too_large(self, manager, tmp_path):
        """Test large file handling."""
        repo_path = manager.workspace_root / "test-org" / "test-repo"
        repo_path.mkdir(parents=True)

        # Create a file larger than 10MB
        large_file = repo_path / "large.bin"
        large_file.write_bytes(b"x" * (11 * 1024 * 1024))

        with patch.object(manager, 'ensure_repo', return_value=repo_path):
            result = manager.read_files("test-org/test-repo", ["large.bin"])

        # Should skip large file
        assert len(result) == 0


class TestBranchOperations:
    """Test branch creation and management."""

    def test_branch_validation(self, manager):
        """Test branch name validation during creation."""
        # Invalid branch name should raise
        with pytest.raises(ValueError):
            manager.create_branch("test-org/test-repo", "invalid branch name!")


class TestCommitAndPush:
    """Test commit and push operations."""

    def test_commit_message_validation(self, manager):
        """Test commit message validation."""
        # Empty message should fail
        with pytest.raises(ValueError):
            manager._validate_commit_message("")

        # Too long message should fail
        with pytest.raises(ValueError):
            manager._validate_commit_message("x" * 15000)


class TestAuthenticationVerification:
    """Test authentication and authorization checks."""

    @patch('github.Github')
    def test_verify_authentication_success(self, mock_github_class, temp_workspace, mock_token):
        """Test successful authentication verification."""
        mock_gh = MagicMock()
        mock_user = MagicMock()
        mock_user.login = "test-user"
        mock_gh.get_user.return_value = mock_user
        mock_github_class.return_value = mock_gh

        manager = MultiRepoManager(temp_workspace, mock_token)
        assert manager.gh_user == "test-user"

    @patch('github.Github')
    def test_verify_authentication_failure(self, mock_github_class, temp_workspace, mock_token):
        """Test authentication failure."""
        mock_gh = MagicMock()
        mock_gh.get_user.side_effect = Exception("Invalid credentials")
        mock_github_class.return_value = mock_gh

        with pytest.raises(ValueError, match="authentication failed"):
            MultiRepoManager(temp_workspace, mock_token)

    def test_verify_repo_access_validation(self):
        """Test repository access requires valid repo format."""
        # This is tested indirectly through _validate_owner_repo
        pass  # Covered by existing validation tests
