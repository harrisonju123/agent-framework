"""Detect inefficient tool usage patterns from session JSONL logs.

Parses session logs and applies sliding-window heuristics to find
anti-patterns like sequential reads (instead of grep), repeated globs,
and using bash for file operations. Recommendations feed back into
agent prompts to reduce token waste on subsequent tasks.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Tip text for each anti-pattern
_TIPS = {
    "sequential-reads": (
        "Use Grep to find relevant files before reading. "
        "Avoid reading multiple files sequentially — search first, then read matches."
    ),
    "grep-then-read-same": (
        "When you Grep a specific file, the matching lines are already in the result. "
        "Only Read the file if you need surrounding context beyond what Grep returned."
    ),
    "repeated-glob": (
        "Cache Glob results mentally — don't re-run the same pattern. "
        "If you need the file list again, refer to your earlier result."
    ),
    "bash-for-search": (
        "Use the dedicated Grep/Glob/Read tools instead of bash grep/find/cat/head/tail. "
        "Dedicated tools are faster and produce structured output."
    ),
    "read-without-limit": (
        "When you already know which lines are relevant from Grep, "
        "use Read with offset/limit to fetch only those lines instead of the whole file."
    ),
    "chunked-reread": (
        "Read each file once in full. Calling Read on the same file with "
        "different offset/limit values wastes tokens — the whole file fits in "
        "context after one read."
    ),
}

# Bash commands that indicate file-op anti-pattern
_BASH_SEARCH_COMMANDS = {"grep", "find", "cat", "head", "tail", "rg", "ack"}


@dataclass
class ToolPatternRecommendation:
    """A single tool-usage recommendation derived from session analysis."""
    pattern_id: str
    tip: str
    hit_count: int = 1
    last_seen: float = field(default_factory=time.time)


class ToolPatternAnalyzer:
    """Analyze session JSONL logs for inefficient tool usage patterns.

    Extracts tool_call events and applies a sliding window of 5 events
    to detect common anti-patterns.
    """

    WINDOW_SIZE = 5

    def analyze_session(self, session_path: Path) -> List[ToolPatternRecommendation]:
        """Parse a session JSONL file and return detected anti-patterns."""
        tool_calls = self._extract_tool_calls(session_path)
        if len(tool_calls) < 2:
            return []

        seen: Dict[str, ToolPatternRecommendation] = {}

        for i in range(len(tool_calls)):
            window = tool_calls[i:i + self.WINDOW_SIZE]
            for detector in (
                self._detect_sequential_reads,
                self._detect_grep_then_read_same,
                self._detect_repeated_glob,
                self._detect_bash_for_search,
                self._detect_read_without_limit,
            ):
                rec = detector(window, tool_calls[:i])
                if rec and rec.pattern_id not in seen:
                    seen[rec.pattern_id] = rec

        # Full-history detectors — patterns that span the entire session
        for detector in (self._detect_chunked_reread,):
            rec = detector(tool_calls, seen)
            if rec and rec.pattern_id not in seen:
                seen[rec.pattern_id] = rec

        return list(seen.values())

    def _extract_tool_calls(self, session_path: Path) -> List[Dict[str, Any]]:
        """Parse JSONL and return ordered list of tool_call events."""
        calls = []
        try:
            text = session_path.read_text()
        except OSError as e:
            logger.debug(f"Could not read session log {session_path}: {e}")
            return calls

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("event") == "tool_call":
                calls.append(entry)
        return calls

    def _detect_sequential_reads(
        self, window: List[Dict], history: List[Dict]
    ) -> Optional[ToolPatternRecommendation]:
        """3+ Read calls on different files within window, no Grep nearby."""
        read_files = set()
        has_grep = False

        # Check recent history (last WINDOW_SIZE calls) + current window for Grep
        lookback = history[-self.WINDOW_SIZE:] if history else []
        for call in lookback + window:
            if call.get("tool") == "Grep":
                has_grep = True
                break

        for call in window:
            if call.get("tool") == "Read":
                file_path = (call.get("input") or {}).get("file_path", "")
                if file_path:
                    read_files.add(file_path)

        if len(read_files) >= 3 and not has_grep:
            return ToolPatternRecommendation(
                pattern_id="sequential-reads",
                tip=_TIPS["sequential-reads"],
            )
        return None

    def _detect_grep_then_read_same(
        self, window: List[Dict], _history: List[Dict]
    ) -> Optional[ToolPatternRecommendation]:
        """Grep on file X followed by Read on same file X."""
        grepped_files: set = set()
        for call in window:
            tool = call.get("tool", "")
            inp = call.get("input") or {}
            if tool == "Grep":
                path = inp.get("path", "")
                if path and not path.endswith("/"):
                    grepped_files.add(path)
            elif tool == "Read" and grepped_files:
                file_path = inp.get("file_path", "")
                if file_path in grepped_files:
                    return ToolPatternRecommendation(
                        pattern_id="grep-then-read-same",
                        tip=_TIPS["grep-then-read-same"],
                    )
        return None

    def _detect_repeated_glob(
        self, window: List[Dict], _history: List[Dict]
    ) -> Optional[ToolPatternRecommendation]:
        """Same Glob pattern string appears 2+ times in window."""
        patterns_seen: Dict[str, int] = {}
        for call in window:
            if call.get("tool") == "Glob":
                pattern = (call.get("input") or {}).get("pattern", "")
                if pattern:
                    patterns_seen[pattern] = patterns_seen.get(pattern, 0) + 1
                    if patterns_seen[pattern] >= 2:
                        return ToolPatternRecommendation(
                            pattern_id="repeated-glob",
                            tip=_TIPS["repeated-glob"],
                        )
        return None

    def _detect_bash_for_search(
        self, window: List[Dict], _history: List[Dict]
    ) -> Optional[ToolPatternRecommendation]:
        """Bash command starts with grep, find, cat, head, or tail."""
        for call in window:
            if call.get("tool") == "Bash":
                command = (call.get("input") or {}).get("command", "")
                # Only check the leading command — pipe tails like `| cat` are
                # often used to disable pagers, not as file-search operations
                first_token = command.strip().split()[0] if command.strip() else ""
                if first_token in _BASH_SEARCH_COMMANDS:
                    return ToolPatternRecommendation(
                        pattern_id="bash-for-search",
                        tip=_TIPS["bash-for-search"],
                    )
        return None

    def _detect_chunked_reread(
        self, all_calls: List[Dict], _seen: Dict[str, ToolPatternRecommendation]
    ) -> Optional[ToolPatternRecommendation]:
        """Same file Read 3+ times across the session — chunked re-read anti-pattern."""
        read_counts: Dict[str, int] = {}
        for call in all_calls:
            if call.get("tool") == "Read":
                file_path = (call.get("input") or {}).get("file_path", "")
                if file_path:
                    read_counts[file_path] = read_counts.get(file_path, 0) + 1
        for count in read_counts.values():
            if count >= 3:
                return ToolPatternRecommendation(
                    pattern_id="chunked-reread",
                    tip=_TIPS["chunked-reread"],
                )
        return None

    def _detect_read_without_limit(
        self, window: List[Dict], history: List[Dict]
    ) -> Optional[ToolPatternRecommendation]:
        """Read with no offset/limit on a file that was previously Grep'd."""
        # Check recent history (not full history) to avoid O(n^2)
        grepped_files: set = set()
        lookback = history[-(self.WINDOW_SIZE * 4):] if history else []
        for call in lookback:
            if call.get("tool") == "Grep":
                path = (call.get("input") or {}).get("path", "")
                if path and not path.endswith("/"):
                    grepped_files.add(path)

        for call in window:
            tool = call.get("tool", "")
            inp = call.get("input") or {}
            if tool == "Grep":
                path = inp.get("path", "")
                if path and not path.endswith("/"):
                    grepped_files.add(path)
            elif tool == "Read":
                file_path = inp.get("file_path", "")
                has_limit = inp.get("limit") is not None or inp.get("offset") is not None
                if file_path in grepped_files and not has_limit:
                    return ToolPatternRecommendation(
                        pattern_id="read-without-limit",
                        tip=_TIPS["read-without-limit"],
                    )
        return None
