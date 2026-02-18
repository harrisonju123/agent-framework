"""Tests for stream-json parser used by ClaudeCLIBackend."""

import asyncio
import json
from io import StringIO

import pytest

from agent_framework.llm.claude_cli_backend import _process_stream_line


class TestProcessStreamLine:
    """Test _process_stream_line() for each event type."""

    def test_assistant_text_extraction(self):
        """Extract text from assistant message content blocks."""
        event = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello, world!"},
                    {"type": "text", "text": " More text."},
                ]
            }
        }
        text_chunks = []
        usage_result = {}

        _process_stream_line(json.dumps(event), text_chunks, usage_result)

        assert text_chunks == ["Hello, world!", " More text."]

    def test_result_event_captures_usage(self):
        """Capture token counts and cost from result event."""
        event = {
            "type": "result",
            "usage": {
                "input_tokens": 1500,
                "output_tokens": 800,
            },
            "total_cost_usd": 0.0234,
            "result": "Final output text",
        }
        text_chunks = []
        usage_result = {}

        _process_stream_line(json.dumps(event), text_chunks, usage_result)

        assert usage_result["input_tokens"] == 1500
        assert usage_result["output_tokens"] == 800
        assert usage_result["total_cost_usd"] == 0.0234
        assert usage_result["result_text"] == "Final output text"
        assert text_chunks == []

    def test_non_json_fallback(self):
        """Non-JSON lines treated as raw text."""
        text_chunks = []
        usage_result = {}

        _process_stream_line("This is plain text output", text_chunks, usage_result)

        assert text_chunks == ["This is plain text output\n"]
        assert usage_result == {}

    def test_tool_use_logged_as_marker(self):
        """Tool use blocks produce readable markers."""
        event = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "github_create_pr"},
                ]
            }
        }
        text_chunks = []
        usage_result = {}

        _process_stream_line(json.dumps(event), text_chunks, usage_result)

        assert len(text_chunks) == 1
        assert "[Tool Call: github_create_pr]" in text_chunks[0]

    def test_missing_usage_fields_default_to_zero(self):
        """Result event with empty usage defaults gracefully."""
        event = {
            "type": "result",
            "usage": {},
        }
        text_chunks = []
        usage_result = {}

        _process_stream_line(json.dumps(event), text_chunks, usage_result)

        assert usage_result["input_tokens"] == 0
        assert usage_result["output_tokens"] == 0
        assert usage_result.get("total_cost_usd") is None
        assert "result_text" not in usage_result

    def test_multi_turn_text_accumulation(self):
        """Multiple assistant events accumulate text in order."""
        events = [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Part 1. "}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Part 2. "}]}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "Part 3."}]}},
        ]
        text_chunks = []
        usage_result = {}

        for event in events:
            _process_stream_line(json.dumps(event), text_chunks, usage_result)

        assert "".join(text_chunks) == "Part 1. Part 2. Part 3."

    def test_system_event_ignored_for_content(self):
        """System events don't contribute to text_chunks."""
        event = {
            "type": "system",
            "subtype": "init",
            "session_id": "abc-123",
        }
        text_chunks = []
        usage_result = {}

        _process_stream_line(json.dumps(event), text_chunks, usage_result)

        assert text_chunks == []
        assert usage_result == {}

    def test_empty_line_ignored(self):
        """Empty lines are silently skipped."""
        text_chunks = []
        usage_result = {}

        _process_stream_line("", text_chunks, usage_result)
        _process_stream_line("   ", text_chunks, usage_result)

        assert text_chunks == []
        assert usage_result == {}

    def test_log_file_receives_text(self):
        """Text content is written to log file in real-time."""
        event = {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "Logged text"}]
            }
        }
        log_file = StringIO()
        text_chunks = []
        usage_result = {}

        _process_stream_line(json.dumps(event), text_chunks, usage_result, log_file=log_file)

        assert log_file.getvalue() == "Logged text"

    def test_system_init_logs_session_id(self):
        """System init events write session info to log file."""
        event = {
            "type": "system",
            "subtype": "init",
            "session_id": "sess-xyz",
        }
        log_file = StringIO()
        text_chunks = []
        usage_result = {}

        _process_stream_line(json.dumps(event), text_chunks, usage_result, log_file=log_file)

        assert "[Session: sess-xyz]" in log_file.getvalue()

    def test_system_compact_boundary_does_not_log_session(self):
        """Non-init system subtypes don't write session header."""
        event = {
            "type": "system",
            "subtype": "compact_boundary",
            "session_id": "sess-xyz",
        }
        log_file = StringIO()
        text_chunks = []
        usage_result = {}

        _process_stream_line(json.dumps(event), text_chunks, usage_result, log_file=log_file)

        assert log_file.getvalue() == ""

    def test_result_without_result_text(self):
        """Result event without result field doesn't set result_text."""
        event = {
            "type": "result",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "total_cost_usd": 0.001,
        }
        text_chunks = []
        usage_result = {}

        _process_stream_line(json.dumps(event), text_chunks, usage_result)

        assert usage_result["input_tokens"] == 100
        assert "result_text" not in usage_result

    def test_unknown_event_type_ignored(self):
        """Unknown event types don't affect text or usage."""
        event = {"type": "ping", "data": "keepalive"}
        text_chunks = []
        usage_result = {}

        _process_stream_line(json.dumps(event), text_chunks, usage_result)

        assert text_chunks == []
        assert usage_result == {}

    def test_per_turn_usage_accumulates(self):
        """Per-turn message.usage accumulates as fallback for missing result event."""
        events = [
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Turn 1"}],
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                }
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Turn 2"}],
                    "usage": {"input_tokens": 200, "output_tokens": 80},
                }
            },
        ]
        text_chunks = []
        usage_result = {}

        for event in events:
            _process_stream_line(json.dumps(event), text_chunks, usage_result)

        assert usage_result["input_tokens"] == 300
        assert usage_result["output_tokens"] == 130

    def test_result_event_uses_larger_of_accumulated_or_reported(self):
        """When result event has larger totals, those win."""
        events = [
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Turn 1"}],
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                }
            },
            {
                "type": "result",
                "usage": {"input_tokens": 500, "output_tokens": 200},
                "total_cost_usd": 0.01,
                "result": "Final",
            },
        ]
        text_chunks = []
        usage_result = {}

        for event in events:
            _process_stream_line(json.dumps(event), text_chunks, usage_result)

        assert usage_result["input_tokens"] == 500
        assert usage_result["output_tokens"] == 200

    def test_result_event_does_not_undercount_accumulated_tokens(self):
        """Result event with only final-turn usage doesn't erase accumulated totals."""
        events = [
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Turn 1"}],
                    "usage": {"input_tokens": 5000, "output_tokens": 1200},
                }
            },
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Turn 2"}],
                    "usage": {"input_tokens": 8000, "output_tokens": 800},
                }
            },
            {
                "type": "result",
                "usage": {"input_tokens": 3, "output_tokens": 77},
                "total_cost_usd": 3.38,
                "result": "Final",
            },
        ]
        text_chunks = []
        usage_result = {}

        for event in events:
            _process_stream_line(json.dumps(event), text_chunks, usage_result)

        # Accumulated per-turn totals (13000 in, 2000 out) should be preserved
        # over the result event's final-turn-only values (3 in, 77 out)
        assert usage_result["input_tokens"] == 13000
        assert usage_result["output_tokens"] == 2000
        assert usage_result["total_cost_usd"] == 3.38

    def test_assistant_without_usage_field(self):
        """Assistant events without message.usage don't affect token counts."""
        event = {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "No usage here"}],
            }
        }
        text_chunks = []
        usage_result = {}

        _process_stream_line(json.dumps(event), text_chunks, usage_result)

        assert text_chunks == ["No usage here"]
        assert "input_tokens" not in usage_result


class TestBufferSplitting:
    """Test that partial JSON lines split across read chunks are handled correctly."""

    @pytest.mark.asyncio
    async def test_json_line_split_across_chunks(self):
        """Verify line-buffering handles a JSON line split across two reads."""
        from agent_framework.llm.claude_cli_backend import _process_stream_line

        # Simulate a stream where a JSON line is split across two chunks
        line = json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "split test"}]}})
        # Split the line roughly in the middle
        mid = len(line) // 2
        chunk1 = (line[:mid]).encode()
        chunk2 = (line[mid:] + "\n").encode()

        text_chunks = []
        usage_result = {}

        # Reproduce the buffer logic from read_stdout_stream_json
        buffer = b""
        for chunk in [chunk1, chunk2, b""]:  # b"" signals EOF
            if not chunk:
                if buffer:
                    _process_stream_line(buffer.decode(errors='replace'), text_chunks, usage_result)
                break
            buffer += chunk
            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                _process_stream_line(line_bytes.decode(errors='replace'), text_chunks, usage_result)

        assert text_chunks == ["split test"]

    @pytest.mark.asyncio
    async def test_multiple_lines_in_single_chunk(self):
        """Verify multiple JSON lines delivered in one chunk are all parsed."""
        line1 = json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "A"}]}})
        line2 = json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": "B"}]}})
        chunk = (line1 + "\n" + line2 + "\n").encode()

        text_chunks = []
        usage_result = {}

        buffer = b""
        for c in [chunk, b""]:
            if not c:
                if buffer:
                    _process_stream_line(buffer.decode(errors='replace'), text_chunks, usage_result)
                break
            buffer += c
            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                _process_stream_line(line_bytes.decode(errors='replace'), text_chunks, usage_result)
