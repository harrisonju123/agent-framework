"""Tests for DashboardDataProvider log file filtering.

Verifies that get_all_log_positions() and get_available_log_files() only
return entries for agents defined in agents.yaml — not CLI, dashboard,
or other artifact log files.
"""

import tempfile
from pathlib import Path

from agent_framework.web.data_provider import DashboardDataProvider


def _make_workspace(tmpdir: str) -> Path:
    """Create a workspace with agents.yaml defining architect/engineer/qa."""
    workspace = Path(tmpdir)
    config_dir = workspace / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "agents.yaml").write_text(
        """
agents:
  - id: architect
    name: Architect
    queue: architect
    enabled: true
    prompt: "test"
  - id: engineer
    name: Engineer
    queue: engineer
    enabled: true
    prompt: "test"
  - id: qa
    name: QA
    queue: qa
    enabled: true
    prompt: "test"
"""
    )
    return workspace


def _populate_logs(logs_dir: Path, filenames: list[str]) -> None:
    """Create log files with some content."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        (logs_dir / name).write_text("2026-01-01 INFO test log line\n")


class TestGetAvailableLogFiles:
    def test_returns_only_known_agent_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            _populate_logs(
                workspace / "logs",
                [
                    "architect.log",
                    "engineer.log",
                    "qa.log",
                    "dashboard.log",
                    "--help.log",
                    "claude-cli-task-abc.log",
                    "claude-cli-task-xyz.log",
                ],
            )
            provider = DashboardDataProvider(workspace)
            result = sorted(provider.get_available_log_files())
            assert result == ["architect", "engineer", "qa"]

    def test_excludes_agents_without_log_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            _populate_logs(workspace / "logs", ["engineer.log"])
            provider = DashboardDataProvider(workspace)
            assert provider.get_available_log_files() == ["engineer"]

    def test_empty_when_no_logs_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)
            assert provider.get_available_log_files() == []


class TestGetAllLogPositions:
    def test_returns_only_known_agent_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            _populate_logs(
                workspace / "logs",
                [
                    "architect.log",
                    "engineer.log",
                    "dashboard.log",
                    "claude-cli-task-1.log",
                ],
            )
            provider = DashboardDataProvider(workspace)
            positions = provider.get_all_log_positions()

            assert set(positions.keys()) == {"architect", "engineer"}
            # Each file has content so position > 0
            for pos in positions.values():
                assert pos > 0

    def test_excludes_agents_without_log_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            _populate_logs(workspace / "logs", ["qa.log"])
            provider = DashboardDataProvider(workspace)
            positions = provider.get_all_log_positions()
            assert set(positions.keys()) == {"qa"}

    def test_empty_when_no_logs_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)
            assert provider.get_all_log_positions() == {}


class TestGetKnownAgentIds:
    def test_returns_all_configured_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = _make_workspace(tmpdir)
            provider = DashboardDataProvider(workspace)
            assert provider._get_known_agent_ids() == {"architect", "engineer", "qa"}

    def test_empty_when_no_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            provider = DashboardDataProvider(workspace)
            assert provider._get_known_agent_ids() == set()
