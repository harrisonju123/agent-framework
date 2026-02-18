"""Tests for `agent purge` CLI command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from agent_framework.cli.main import cli


# ── fixtures ──────────────────────────────────────────────────────────────────


def _make_comm_dir(base: Path) -> Path:
    """Create a minimal .agent-communication directory so the purge command proceeds."""
    comm = base / ".agent-communication"
    comm.mkdir(parents=True)
    return comm


def _populate_workspace(base: Path) -> dict[str, list[Path]]:
    """Create representative files in every purge category. Returns created paths by category."""
    comm = _make_comm_dir(base)
    created: dict[str, list[Path]] = {}

    # Task queues
    q = comm / "queues" / "engineer"
    q.mkdir(parents=True)
    queue_file = q / "task-001.json"
    queue_file.write_text("{}")
    created["queues"] = [queue_file]

    # Completed tasks
    c = comm / "completed"
    c.mkdir()
    f = c / "done-001.json"
    f.write_text("{}")
    created["completed"] = [f]

    # Locks — these are directories, not plain files
    lk = comm / "locks"
    lk.mkdir()
    lock_dir = lk / "task-001.lock"
    lock_dir.mkdir()
    (lock_dir / "pid").write_text("1234")
    created["locks"] = [lock_dir]

    # Heartbeats
    hb = comm / "heartbeats"
    hb.mkdir()
    f = hb / "engineer.json"
    f.write_text("{}")
    created["heartbeats"] = [f]

    # Activity
    act = comm / "activity"
    act.mkdir()
    f = act / "event.json"
    f.write_text("{}")
    stream = comm / "activity-stream.jsonl"
    stream.write_text("{}\n")
    created["activity"] = [f, stream]

    # Memory
    mem = comm / "memory"
    mem.mkdir()
    f = mem / "patterns.json"
    f.write_text("{}")
    created["memory"] = [f]

    # Indexes
    idx = comm / "indexes"
    idx.mkdir()
    f = idx / "codebase.json"
    f.write_text("{}")
    created["indexes"] = [f]

    # Context artifacts
    ctx_dir = base / ".agent-context"
    ctx_dir.mkdir()
    f = ctx_dir / "summary.md"
    f.write_text("# summary")
    created["context"] = [f]

    # Logs
    logs = base / "logs"
    logs.mkdir()
    f = logs / "agent.log"
    f.write_text("log line\n")
    sessions = logs / "sessions"
    sessions.mkdir()
    sf = sessions / "session-001.jsonl"
    sf.write_text("{}\n")
    created["logs"] = [f, sf]

    return created


def _mock_orchestrator():
    """Return a patched Orchestrator that does nothing."""
    orch = MagicMock()
    orch.get_dashboard_info.return_value = None
    return orch


# ── tests ─────────────────────────────────────────────────────────────────────


def test_purge_empty_workspace(tmp_path):
    """`agent purge` exits cleanly when .agent-communication/ doesn't exist."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "purge", "--yes"])

    assert result.exit_code == 0
    assert "Nothing to purge" in result.output


@patch("agent_framework.cli.main.Orchestrator")
def test_purge_removes_all_categories(mock_orch_cls, tmp_path, monkeypatch):
    """All file categories are deleted when no --keep flags are given."""
    mock_orch_cls.return_value = _mock_orchestrator()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    created = _populate_workspace(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "purge", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Purged" in result.output

    # Every created file/dir should be gone
    for paths in created.values():
        for p in paths:
            assert not p.exists(), f"Expected {p} to be removed"

    # Glob targets delete file contents but leave the parent dirs intact so the
    # framework can recreate them on next startup without mkdir errors.
    assert (tmp_path / ".agent-communication" / "queues").exists(), "queues/ parent dir must survive"
    assert (tmp_path / ".agent-communication" / "completed").exists(), "completed/ parent dir must survive"


@patch("agent_framework.cli.main.Orchestrator")
def test_purge_keep_memory(mock_orch_cls, tmp_path, monkeypatch):
    """--keep-memory preserves memory dir while other categories are deleted."""
    mock_orch_cls.return_value = _mock_orchestrator()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    _populate_workspace(tmp_path)

    memory_dir = tmp_path / ".agent-communication" / "memory"
    queue_file = tmp_path / ".agent-communication" / "queues" / "engineer" / "task-001.json"

    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "purge", "--keep-memory", "--yes"])

    assert result.exit_code == 0, result.output
    assert memory_dir.exists(), "Memory dir should be preserved"
    assert not queue_file.exists(), "Queue file should be removed"
    assert "Preserved: memory" in result.output


@patch("agent_framework.cli.main.Orchestrator")
def test_purge_keep_indexes(mock_orch_cls, tmp_path, monkeypatch):
    """--keep-indexes preserves the indexes directory."""
    mock_orch_cls.return_value = _mock_orchestrator()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    _populate_workspace(tmp_path)

    indexes_dir = tmp_path / ".agent-communication" / "indexes"

    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "purge", "--keep-indexes", "--yes"])

    assert result.exit_code == 0, result.output
    assert indexes_dir.exists(), "Indexes dir should be preserved"
    assert "Preserved: indexes" in result.output


@patch("agent_framework.cli.main.Orchestrator")
def test_purge_keep_worktrees(mock_orch_cls, tmp_path, monkeypatch):
    """--keep-worktrees preserves ~/.agent-workspaces/."""
    mock_orch_cls.return_value = _mock_orchestrator()
    _populate_workspace(tmp_path)

    # Point home to tmp_path so we don't touch the real ~/.agent-workspaces
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    workspaces = fake_home / ".agent-workspaces"
    workspaces.mkdir()
    (workspaces / "some-repo").mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "purge", "--keep-worktrees", "--yes"])

    assert result.exit_code == 0, result.output
    assert workspaces.exists(), "Workspaces dir should be preserved"
    assert "Preserved: worktrees" in result.output


@patch("agent_framework.cli.main.Orchestrator")
def test_purge_dry_run(mock_orch_cls, tmp_path):
    """--dry-run outputs the summary table but deletes nothing."""
    mock_orch_cls.return_value = _mock_orchestrator()
    created = _populate_workspace(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "purge", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert "Purged" not in result.output

    # Nothing should be deleted
    for paths in created.values():
        for p in paths:
            assert p.exists(), f"Expected {p} to survive dry run"


@patch("agent_framework.cli.main.Orchestrator")
def test_purge_stops_agents_before_deleting(mock_orch_cls, tmp_path, monkeypatch):
    """Purge stops agents before deleting files (but only when there is something to delete)."""
    mock_orch = _mock_orchestrator()
    mock_orch_cls.return_value = mock_orch
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    _populate_workspace(tmp_path)

    runner = CliRunner()
    runner.invoke(cli, ["--workspace", str(tmp_path), "purge", "--yes"])

    mock_orch.stop_all_agents.assert_called_once_with(graceful=True)


def test_purge_confirmation_cancelled(tmp_path):
    """Answering 'n' at the confirmation prompt leaves files intact."""
    _populate_workspace(tmp_path)
    queue_file = tmp_path / ".agent-communication" / "queues" / "engineer" / "task-001.json"

    with patch("agent_framework.cli.main.Orchestrator") as mock_orch_cls:
        mock_orch_cls.return_value = _mock_orchestrator()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--workspace", str(tmp_path), "purge"],
            input="n\n",
        )

    assert result.exit_code == 0
    assert "Cancelled" in result.output
    assert queue_file.exists(), "File must not be deleted when user cancels"


@patch("agent_framework.cli.main.Orchestrator")
def test_purge_handles_missing_directories(mock_orch_cls, tmp_path):
    """Purge handles a workspace where most subdirectories don't exist yet."""
    mock_orch_cls.return_value = _mock_orchestrator()
    # Only create the bare minimum so the command proceeds past the early-exit check
    _make_comm_dir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "purge", "--yes"])

    assert result.exit_code == 0, result.output
    # Nothing to remove, but command should complete without error
    assert "Nothing to purge" in result.output


@patch("agent_framework.cli.main.Orchestrator")
def test_purge_handles_permission_errors(mock_orch_cls, tmp_path, monkeypatch):
    """Purge continues past files it cannot delete and reports success for others."""
    mock_orch_cls.return_value = _mock_orchestrator()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    _populate_workspace(tmp_path)

    original_unlink = Path.unlink

    def flaky_unlink(self, missing_ok=False):
        # Simulate a permission error on a specific file
        if self.name == "engineer.json":
            raise PermissionError("Permission denied")
        original_unlink(self, missing_ok=missing_ok)

    with patch.object(Path, "unlink", flaky_unlink):
        runner = CliRunner()
        result = runner.invoke(cli, ["--workspace", str(tmp_path), "purge", "--yes"])

    assert result.exit_code == 0, result.output
    # Warning emitted for the failing file
    assert "Warning" in result.output
    # Command still completes with a purged count
    assert "Purged" in result.output


@patch("agent_framework.cli.main.Orchestrator")
def test_purge_dry_run_does_not_stop_agents(mock_orch_cls, tmp_path):
    """--dry-run must not stop agents — it should have zero side effects."""
    mock_orch = _mock_orchestrator()
    mock_orch_cls.return_value = mock_orch
    _populate_workspace(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "purge", "--dry-run"])

    assert result.exit_code == 0, result.output
    mock_orch.stop_all_agents.assert_not_called()
    mock_orch.stop_dashboard.assert_not_called()


def test_purge_whole_dir_count_reflects_file_content(tmp_path):
    """_count_purge_target returns the file count inside a whole-dir target, not always 1."""
    from agent_framework.cli.main import _PurgeTarget, _count_purge_target

    d = tmp_path / "memory"
    d.mkdir()
    for i in range(5):
        (d / f"pattern-{i}.json").write_text("{}")

    target = _PurgeTarget(d, whole_dir=True)
    assert _count_purge_target(target) == 5
