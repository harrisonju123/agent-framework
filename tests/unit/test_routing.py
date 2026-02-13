"""Tests for routing signal handling."""

import json
from types import SimpleNamespace

import pytest

from agent_framework.core.routing import (
    RoutingSignal,
    read_routing_signal,
    validate_routing_signal,
    log_routing_decision,
    WORKFLOW_COMPLETE,
)


def _make_signal_file(tmp_path, task_id, target="qa", reason="PR ready", source="engineer"):
    signals_dir = tmp_path / ".agent-communication" / "routing-signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    signal = {
        "target_agent": target,
        "reason": reason,
        "timestamp": "2026-02-13T16:30:00Z",
        "source_agent": source,
    }
    path = signals_dir / f"{task_id}.json"
    path.write_text(json.dumps(signal))
    return path


def _make_agent_def(agent_id):
    return SimpleNamespace(id=agent_id)


KNOWN_AGENTS = [
    _make_agent_def("architect"),
    _make_agent_def("engineer"),
    _make_agent_def("qa"),
]


class TestReadRoutingSignal:
    def test_reads_and_deletes_signal(self, tmp_path):
        path = _make_signal_file(tmp_path, "task-001")
        assert path.exists()

        signal = read_routing_signal(tmp_path, "task-001")

        assert signal is not None
        assert signal.target_agent == "qa"
        assert signal.reason == "PR ready"
        assert signal.source_agent == "engineer"
        assert not path.exists()

    def test_returns_none_when_no_file(self, tmp_path):
        signal = read_routing_signal(tmp_path, "nonexistent-task")
        assert signal is None

    def test_handles_corrupt_json(self, tmp_path):
        signals_dir = tmp_path / ".agent-communication" / "routing-signals"
        signals_dir.mkdir(parents=True, exist_ok=True)
        (signals_dir / "task-corrupt.json").write_text("{bad json")

        signal = read_routing_signal(tmp_path, "task-corrupt")

        assert signal is None
        assert not (signals_dir / "task-corrupt.json").exists()

    def test_handles_missing_fields(self, tmp_path):
        signals_dir = tmp_path / ".agent-communication" / "routing-signals"
        signals_dir.mkdir(parents=True, exist_ok=True)
        (signals_dir / "task-partial.json").write_text(json.dumps({"target_agent": "qa"}))

        signal = read_routing_signal(tmp_path, "task-partial")

        assert signal is None


class TestValidateRoutingSignal:
    def _signal(self, target="qa", source="engineer"):
        return RoutingSignal(
            target_agent=target,
            reason="test reason",
            timestamp="2026-02-13T16:30:00Z",
            source_agent=source,
        )

    def test_valid_signal_returns_target(self):
        result = validate_routing_signal(
            self._signal(target="qa"),
            current_agent="engineer",
            task_type="implementation",
            agents_config=KNOWN_AGENTS,
        )
        assert result == "qa"

    def test_rejects_route_to_self(self):
        result = validate_routing_signal(
            self._signal(target="engineer"),
            current_agent="engineer",
            task_type="implementation",
            agents_config=KNOWN_AGENTS,
        )
        assert result is None

    def test_allows_workflow_complete(self):
        result = validate_routing_signal(
            self._signal(target=WORKFLOW_COMPLETE),
            current_agent="engineer",
            task_type="implementation",
            agents_config=KNOWN_AGENTS,
        )
        assert result == WORKFLOW_COMPLETE

    def test_rejects_unknown_agent(self):
        result = validate_routing_signal(
            self._signal(target="unknown-agent"),
            current_agent="engineer",
            task_type="implementation",
            agents_config=KNOWN_AGENTS,
        )
        assert result is None

    def test_rejects_escalation_reroute(self):
        result = validate_routing_signal(
            self._signal(target="qa"),
            current_agent="engineer",
            task_type="escalation",
            agents_config=KNOWN_AGENTS,
        )
        assert result is None

    def test_skips_agent_check_with_empty_config(self):
        result = validate_routing_signal(
            self._signal(target="qa"),
            current_agent="engineer",
            task_type="implementation",
            agents_config=[],
        )
        assert result == "qa"


class TestLogRoutingDecision:
    def test_writes_jsonl_entry(self, tmp_path):
        signal = RoutingSignal(
            target_agent="qa", reason="PR ready",
            timestamp="2026-02-13T16:30:00Z", source_agent="engineer",
        )
        log_routing_decision(
            tmp_path, "task-001", "engineer",
            signal, "qa", used_fallback=False,
        )

        metrics_file = tmp_path / "metrics" / "routing.jsonl"
        assert metrics_file.exists()
        entry = json.loads(metrics_file.read_text().strip())
        assert entry["task_id"] == "task-001"
        assert entry["signal_target"] == "qa"
        assert entry["validated_target"] == "qa"
        assert entry["used_fallback"] is False

    def test_handles_none_signal(self, tmp_path):
        log_routing_decision(
            tmp_path, "task-002", "engineer",
            None, "qa", used_fallback=True,
        )

        metrics_file = tmp_path / "metrics" / "routing.jsonl"
        entry = json.loads(metrics_file.read_text().strip())
        assert entry["signal_target"] is None
        assert entry["signal_reason"] is None
        assert entry["used_fallback"] is True

    def test_creates_metrics_directory(self, tmp_path):
        log_routing_decision(
            tmp_path, "task-003", "engineer",
            None, None, used_fallback=True,
        )
        assert (tmp_path / "metrics" / "routing.jsonl").exists()
