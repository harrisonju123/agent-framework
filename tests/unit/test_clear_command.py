"""Tests for `agent clear` CLI command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from agent_framework.cli.main import cli


# ── fixtures ──────────────────────────────────────────────────────────────────


def _make_comm_dir(base: Path) -> Path:
    """Create a minimal .agent-communication directory so the clear command proceeds."""
    comm = base / ".agent-communication"
    comm.mkdir(parents=True)
    return comm


def _populate_workspace(base: Path) -> dict[str, list[Path]]:
    """Create representative files in every runtime category that `clear` removes."""
    comm = _make_comm_dir(base)
    created: dict[str, list[Path]] = {}

    # Task queues
    q = comm / "queues" / "engineer"
    q.mkdir(parents=True)
    f = q / "task-001.json"
    f.write_text("{}")
    created["queues"] = [f]

    # Completed tasks
    c = comm / "completed"
    c.mkdir()
    f = c / "done-001.json"
    f.write_text("{}")
    created["completed"] = [f]

    # Locks (directory-based)
    lk = comm / "locks"
    lk.mkdir()
    lock_dir = lk / "task-001.lock"
    lock_dir.mkdir()
    (lock_dir / "pid").write_text("1234")
    created["locks"] = [lock_dir]

    # Heartbeats
    hb = comm / "heartbeats"
    hb.mkdir()
    f = hb / "engineer"
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

    # Chain state
    cs = comm / "chain-state"
    cs.mkdir()
    f = cs / "root-001.json"
    f.write_text("{}")
    created["chain_state"] = [f]

    # Read cache
    rc = comm / "read-cache"
    rc.mkdir()
    f = rc / "root-001.json"
    f.write_text("{}")
    created["read_cache"] = [f]

    # Pre-scans
    ps = comm / "pre-scans"
    ps.mkdir()
    f = ps / "scan.json"
    f.write_text("{}")
    created["pre_scans"] = [f]

    # Metrics
    met = comm / "metrics"
    met.mkdir()
    f = met / "usage.json"
    f.write_text("{}")
    created["metrics"] = [f]

    # Routing signals
    rs = comm / "routing-signals"
    rs.mkdir()
    f = rs / "signal.json"
    f.write_text("{}")
    created["routing_signals"] = [f]

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

    # Memory and indexes — should be PRESERVED by clear
    mem = comm / "memory"
    mem.mkdir()
    f = mem / "patterns.json"
    f.write_text("{}")
    created["memory"] = [f]

    idx = comm / "indexes"
    idx.mkdir()
    f = idx / "codebase.json"
    f.write_text("{}")
    created["indexes"] = [f]

    return created


def _mock_orchestrator():
    """Return a patched Orchestrator that does nothing."""
    orch = MagicMock()
    orch.get_dashboard_info.return_value = None
    return orch


# ── tests ─────────────────────────────────────────────────────────────────────


def test_clear_empty_workspace(tmp_path):
    """`agent clear` exits cleanly when .agent-communication/ doesn't exist."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "clear", "--yes"])

    assert result.exit_code == 0
    assert "Nothing to clear" in result.output


def test_clear_nothing_to_remove(tmp_path):
    """`agent clear` exits cleanly when comm dir exists but is empty."""
    _make_comm_dir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "clear", "--yes"])

    assert result.exit_code == 0
    assert "Nothing to clear" in result.output


@patch("agent_framework.cli.main.Orchestrator")
def test_clear_removes_all_runtime_categories(mock_orch_cls, tmp_path):
    """All runtime categories are removed, memory and indexes are preserved."""
    mock_orch_cls.return_value = _mock_orchestrator()
    created = _populate_workspace(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "clear", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Cleared" in result.output

    # Runtime files should be gone
    preserved_keys = {"memory", "indexes"}
    for key, paths in created.items():
        if key in preserved_keys:
            continue
        for p in paths:
            assert not p.exists(), f"Expected {p} to be removed (category: {key})"

    # Memory and indexes must survive
    for key in preserved_keys:
        for p in created[key]:
            assert p.exists(), f"Expected {p} to be preserved (category: {key})"


@patch("agent_framework.cli.main.Orchestrator")
def test_clear_preserves_message(mock_orch_cls, tmp_path):
    """Output tells the user what was preserved."""
    mock_orch_cls.return_value = _mock_orchestrator()
    _populate_workspace(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "clear", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Preserved: memory, indexes, worktrees" in result.output


@patch("agent_framework.cli.main.Orchestrator")
def test_clear_stops_agents_before_deleting(mock_orch_cls, tmp_path):
    """Clear stops agents before removing files."""
    mock_orch = _mock_orchestrator()
    mock_orch_cls.return_value = mock_orch
    _populate_workspace(tmp_path)

    runner = CliRunner()
    runner.invoke(cli, ["--workspace", str(tmp_path), "clear", "--yes"])

    mock_orch.stop_all_agents.assert_called_once_with(graceful=True)


def test_clear_confirmation_cancelled(tmp_path):
    """Answering 'n' at the confirmation prompt leaves files intact."""
    _populate_workspace(tmp_path)
    queue_file = tmp_path / ".agent-communication" / "queues" / "engineer" / "task-001.json"

    with patch("agent_framework.cli.main.Orchestrator") as mock_orch_cls:
        mock_orch_cls.return_value = _mock_orchestrator()

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--workspace", str(tmp_path), "clear"],
            input="n\n",
        )

    assert result.exit_code == 0
    assert "Cancelled" in result.output
    assert queue_file.exists(), "File must not be deleted when user cancels"


@patch("agent_framework.cli.main.Orchestrator")
def test_clear_handles_missing_directories(mock_orch_cls, tmp_path):
    """Clear handles a workspace where most subdirectories don't exist."""
    mock_orch_cls.return_value = _mock_orchestrator()
    _make_comm_dir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "clear", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Nothing to clear" in result.output


@patch("agent_framework.cli.main.Orchestrator")
def test_clear_parent_dirs_survive(mock_orch_cls, tmp_path):
    """Glob-based targets remove contents but leave parent directories intact."""
    mock_orch_cls.return_value = _mock_orchestrator()
    _populate_workspace(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "clear", "--yes"])

    assert result.exit_code == 0, result.output
    # Glob targets preserve parent dirs so the framework can recreate files on next startup
    assert (tmp_path / ".agent-communication" / "queues").exists()
    assert (tmp_path / ".agent-communication" / "completed").exists()


@patch("agent_framework.cli.main.Orchestrator")
def test_clear_does_not_stop_agents_when_empty(mock_orch_cls, tmp_path):
    """When there's nothing to clear, don't bother stopping agents."""
    mock_orch = _mock_orchestrator()
    mock_orch_cls.return_value = mock_orch
    _make_comm_dir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "clear", "--yes"])

    assert result.exit_code == 0
    mock_orch.stop_all_agents.assert_not_called()


@patch("agent_framework.cli.main.Orchestrator")
def test_clear_shows_summary_table(mock_orch_cls, tmp_path):
    """Clear displays a Rich table with category names before proceeding."""
    mock_orch_cls.return_value = _mock_orchestrator()
    _populate_workspace(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["--workspace", str(tmp_path), "clear", "--yes"])

    assert result.exit_code == 0, result.output
    # Table header and some category names should appear
    assert "Clear Summary" in result.output
    assert "Task queues" in result.output
    assert "Chain state" in result.output


def test_clear_old_flags_rejected(tmp_path):
    """Old --agent/--completed/--locks flags are no longer accepted."""
    runner = CliRunner()

    for flag in ["--agent", "--completed", "--locks"]:
        result = runner.invoke(cli, ["--workspace", str(tmp_path), "clear", flag, "test"])
        assert result.exit_code != 0, f"{flag} should be rejected"
