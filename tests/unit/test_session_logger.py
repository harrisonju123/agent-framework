"""Tests for the SessionLogger structured logging system."""

import json
import os
import time
from pathlib import Path

import pytest

from agent_framework.core.session_logger import (
    SessionLogger,
    _redact_sensitive_values,
    noop_logger,
)


@pytest.fixture
def logs_dir(tmp_path):
    return tmp_path / "logs"


@pytest.fixture
def session_log_path(logs_dir):
    return logs_dir / "sessions" / "test-task-123.jsonl"


@pytest.fixture
def logger(logs_dir):
    sl = SessionLogger(logs_dir, "test-task-123", enabled=True)
    yield sl
    sl.close()


# -- Enabled / Disabled --


class TestEnabledDisabled:
    def test_enabled_writes_events(self, logger, session_log_path):
        logger.log("task_start", agent="engineer")
        assert session_log_path.exists()
        lines = session_log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event"] == "task_start"
        assert event["agent"] == "engineer"
        assert event["task_id"] == "test-task-123"
        assert "ts" in event

    def test_disabled_writes_nothing(self, logs_dir, session_log_path):
        sl = SessionLogger(logs_dir, "test-task-123", enabled=False)
        sl.log("task_start", agent="engineer")
        sl.log_tool_call("Read", {"file_path": "/foo"})
        sl.log_prompt("You are engineer...")
        sl.close()
        assert not session_log_path.exists()

    def test_enabled_property(self, logs_dir):
        assert SessionLogger(logs_dir, "t1", enabled=True).enabled is True
        assert SessionLogger(logs_dir, "t2", enabled=False).enabled is False


# -- log_prompts flag --


class TestLogPrompts:
    def test_log_prompts_true_writes_prompt_content(self, logs_dir, session_log_path):
        sl = SessionLogger(logs_dir, "test-task-123", log_prompts=True)
        sl.log_prompt("You are engineer working on JIRA-123")
        sl.close()
        lines = session_log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["event"] == "prompt_content"
        assert "You are engineer" in event["prompt"]

    def test_log_prompts_false_suppresses_prompt_content(self, logs_dir, session_log_path):
        sl = SessionLogger(logs_dir, "test-task-123", log_prompts=False)
        sl.log_prompt("You are engineer working on JIRA-123")
        sl.close()
        assert not session_log_path.exists()

    def test_log_prompts_false_does_not_suppress_other_events(self, logs_dir, session_log_path):
        sl = SessionLogger(logs_dir, "test-task-123", log_prompts=False)
        sl.log("task_start", agent="engineer")
        sl.close()
        lines = session_log_path.read_text().strip().splitlines()
        assert len(lines) == 1


# -- log_tool_inputs flag --


class TestLogToolInputs:
    def test_tool_inputs_true_includes_input(self, logs_dir, session_log_path):
        sl = SessionLogger(logs_dir, "test-task-123", log_tool_inputs=True)
        sl.log_tool_call("Read", {"file_path": "/src/foo.py"})
        sl.close()
        event = json.loads(session_log_path.read_text().strip())
        assert event["input"] == {"file_path": "/src/foo.py"}

    def test_tool_inputs_false_excludes_input(self, logs_dir, session_log_path):
        sl = SessionLogger(logs_dir, "test-task-123", log_tool_inputs=False)
        sl.log_tool_call("Read", {"file_path": "/src/foo.py"})
        sl.close()
        event = json.loads(session_log_path.read_text().strip())
        assert "input" not in event
        assert event["tool"] == "Read"

    def test_tool_call_sequence_increments(self, logs_dir, session_log_path):
        sl = SessionLogger(logs_dir, "test-task-123")
        sl.log_tool_call("Read", {"file_path": "/a.py"})
        sl.log_tool_call("Edit", {"file_path": "/a.py"})
        sl.log_tool_call("Bash", {"command": "pytest"})
        sl.close()
        lines = session_log_path.read_text().strip().splitlines()
        sequences = [json.loads(l)["sequence"] for l in lines]
        assert sequences == [1, 2, 3]


# -- JSONL validity --


class TestJSONLValidity:
    def test_each_line_is_valid_json(self, logger, session_log_path):
        logger.log("task_start", agent="engineer", title="Implement feature")
        logger.log_tool_call("Read", {"file_path": "/src/main.py"})
        logger.log_prompt("You are engineer...")
        logger.log("llm_complete", success=True, tokens_in=1000, tokens_out=500)
        logger.log("task_complete", status="completed", duration_ms=45000)
        logger.close()

        lines = session_log_path.read_text().strip().splitlines()
        assert len(lines) == 5
        for line in lines:
            event = json.loads(line)
            assert "ts" in event
            assert "event" in event
            assert "task_id" in event

    def test_non_serializable_values_use_default_str(self, logger, session_log_path):
        """json.dumps(default=str) should handle Path objects etc."""
        logger.log("test_event", path=Path("/some/path"))
        logger.close()
        event = json.loads(session_log_path.read_text().strip())
        assert event["path"] == "/some/path"


# -- close() idempotency --


class TestClose:
    def test_close_twice_is_safe(self, logger):
        logger.log("task_start", agent="engineer")
        logger.close()
        logger.close()  # Should not raise

    def test_context_manager(self, logs_dir, session_log_path):
        with SessionLogger(logs_dir, "test-task-123") as sl:
            sl.log("task_start", agent="engineer")
        # File should be closed, content written
        assert session_log_path.exists()
        event = json.loads(session_log_path.read_text().strip())
        assert event["event"] == "task_start"


# -- noop_logger --


class TestNoopLogger:
    def test_noop_returns_same_instance(self):
        a = noop_logger()
        b = noop_logger()
        assert a is b

    def test_noop_is_disabled(self):
        assert noop_logger().enabled is False

    def test_noop_does_not_increment_sequence(self):
        nl = noop_logger()
        before = nl._sequence
        nl.log_tool_call("Read", {"file_path": "/foo"})
        nl.log_tool_call("Edit", {"file_path": "/foo"})
        assert nl._sequence == before


# -- Append mode (retries accumulate) --


class TestAppendMode:
    def test_multiple_sessions_append_to_same_file(self, logs_dir, session_log_path):
        sl1 = SessionLogger(logs_dir, "test-task-123")
        sl1.log("task_start", retry=0)
        sl1.close()

        sl2 = SessionLogger(logs_dir, "test-task-123")
        sl2.log("task_start", retry=1)
        sl2.close()

        lines = session_log_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["retry"] == 0
        assert json.loads(lines[1])["retry"] == 1


# -- cleanup_old_sessions --


class TestCleanupOldSessions:
    def test_removes_old_files(self, logs_dir):
        sessions_dir = logs_dir / "sessions"
        sessions_dir.mkdir(parents=True)

        old_file = sessions_dir / "old-task.jsonl"
        old_file.write_text('{"event": "task_start"}\n')
        # Backdate mtime to 60 days ago
        old_mtime = time.time() - (60 * 86400)
        os.utime(old_file, (old_mtime, old_mtime))

        new_file = sessions_dir / "new-task.jsonl"
        new_file.write_text('{"event": "task_start"}\n')

        removed = SessionLogger.cleanup_old_sessions(logs_dir, retention_days=30)
        assert removed == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_no_sessions_dir_returns_zero(self, logs_dir):
        assert SessionLogger.cleanup_old_sessions(logs_dir, retention_days=30) == 0

    def test_no_old_files_returns_zero(self, logs_dir):
        sessions_dir = logs_dir / "sessions"
        sessions_dir.mkdir(parents=True)
        recent = sessions_dir / "recent-task.jsonl"
        recent.write_text('{"event": "task_start"}\n')

        assert SessionLogger.cleanup_old_sessions(logs_dir, retention_days=30) == 0
        assert recent.exists()


# -- Redaction --


class TestRedaction:
    def test_curl_password_redacted(self):
        result = _redact_sensitive_values({"command": "curl -u admin:s3cret https://jira.example.com/api"})
        assert "s3cret" not in result["command"]
        assert "-u admin:***" in result["command"]

    def test_authorization_bearer_redacted(self):
        result = _redact_sensitive_values({"command": 'curl -H "Authorization: Bearer tok_abc123" https://api.example.com'})
        assert "tok_abc123" not in result["command"]
        assert "Authorization: Bearer ***" in result["command"]

    def test_authorization_basic_redacted(self):
        result = _redact_sensitive_values({"command": 'curl -H "Authorization: Basic dXNlcjpwYXNz" https://api.example.com'})
        assert "dXNlcjpwYXNz" not in result["command"]
        assert "Authorization: Basic ***" in result["command"]

    def test_env_var_assignment_redacted(self):
        result = _redact_sensitive_values({"command": "JIRA_API_TOKEN=abc123 python script.py"})
        assert "abc123" not in result["command"]
        assert "JIRA_API_TOKEN=***" in result["command"]

    def test_safe_command_unchanged(self):
        cmd = "git status && pytest tests/ -v"
        result = _redact_sensitive_values({"command": cmd})
        assert result["command"] == cmd

    def test_none_input_returns_none(self):
        assert _redact_sensitive_values(None) is None

    def test_empty_dict_returns_empty(self):
        assert _redact_sensitive_values({}) == {}

    def test_non_string_values_pass_through(self):
        result = _redact_sensitive_values({"timeout": 30, "verbose": True})
        assert result == {"timeout": 30, "verbose": True}

    def test_redaction_applied_in_log_tool_call(self, logs_dir, session_log_path):
        """End-to-end: redacted values don't appear in JSONL output."""
        sl = SessionLogger(logs_dir, "test-task-123", log_tool_inputs=True)
        sl.log_tool_call("Bash", {"command": "curl -u user:hunter2 https://jira.example.com"})
        sl.close()
        event = json.loads(session_log_path.read_text().strip())
        assert "hunter2" not in json.dumps(event)
        assert "-u user:***" in event["input"]["command"]
