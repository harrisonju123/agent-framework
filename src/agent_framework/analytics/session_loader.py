"""Shared session JSONL loader for analytics collectors.

Both AgenticMetrics and LlmMetrics scan the same per-task session logs.
This module extracts the common loading logic so it lives in one place.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def load_session_events(
    sessions_dir: Path, cutoff: datetime
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Read session JSONL files modified after cutoff and bucket events by task_id.

    Uses file mtime as a fast pre-filter — if a file hasn't been touched
    since the cutoff there's nothing new in it. Each event is then validated
    against the cutoff timestamp individually.
    """
    if not sessions_dir.exists():
        return {}

    events_by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    cutoff_ts = cutoff.timestamp()

    for path in sessions_dir.glob("*.jsonl"):
        try:
            if path.stat().st_mtime < cutoff_ts:
                continue
        except OSError:
            continue

        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Filter by event timestamp, not just file mtime
                raw_ts = event.get("ts")
                if raw_ts:
                    try:
                        event_ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                        if event_ts < cutoff:
                            continue
                    except (ValueError, TypeError):
                        pass

                task_id = event.get("task_id", path.stem)
                events_by_task[task_id].append(event)

        except OSError as e:
            logger.debug(f"Could not read session log {path}: {e}")

    return events_by_task
