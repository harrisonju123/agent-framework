"""Activity tracking for agent runtime state and events."""

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel

# fcntl is Unix-only, use fallback for Windows
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Agent operational status."""
    IDLE = "idle"
    WORKING = "working"
    DEAD = "dead"


class TaskPhase(str, Enum):
    """Task execution phases."""
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING_LLM = "executing_llm"  # When LLM is processing (Claude running)
    IMPLEMENTING = "implementing"
    TESTING = "testing"
    EXPLORING_CODEBASE = "exploring_codebase"  # Product Owner specific
    CREATING_EPIC = "creating_epic"  # Product Owner specific
    CREATING_SUBTASKS = "creating_subtasks"  # Product Owner specific
    QUEUING_TASKS = "queuing_tasks"  # Product Owner specific
    COMMITTING = "committing"
    CREATING_PR = "creating_pr"
    UPDATING_JIRA = "updating_jira"


class PhaseRecord(BaseModel):
    """Record of a completed phase."""
    name: TaskPhase
    completed: bool
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class CurrentTask(BaseModel):
    """Current task being processed."""
    id: str
    title: str
    type: str
    started_at: datetime


class AgentActivity(BaseModel):
    """Agent's current activity state."""
    agent_id: str
    status: AgentStatus
    current_task: Optional[CurrentTask] = None
    current_phase: Optional[TaskPhase] = None
    phases: List[PhaseRecord] = []
    last_updated: datetime

    def get_elapsed_seconds(self) -> Optional[int]:
        """Calculate elapsed time for current task."""
        if not self.current_task:
            return None
        elapsed = datetime.utcnow() - self.current_task.started_at
        return int(elapsed.total_seconds())


class ActivityEvent(BaseModel):
    """Event for activity stream."""
    type: str  # "start", "complete", "fail", "phase"
    agent: str
    task_id: str
    title: str
    timestamp: datetime
    duration_ms: Optional[int] = None
    retry_count: Optional[int] = None
    phase: Optional[TaskPhase] = None
    error_message: Optional[str] = None  # Error details for failed tasks
    pr_url: Optional[str] = None  # PR URL for completed tasks


class ActivityManager:
    """Manages agent activity state and event stream."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.activity_dir = self.workspace / ".agent-communication" / "activity"
        self.stream_file = self.workspace / ".agent-communication" / "activity-stream.jsonl"
        self.activity_dir.mkdir(parents=True, exist_ok=True)
        self.max_stream_events = 100

    def update_activity(self, activity: AgentActivity) -> None:
        """Update agent's activity state atomically."""
        activity.last_updated = datetime.utcnow()
        activity_file = self.activity_dir / f"{activity.agent_id}.json"

        # Atomic write: write to .tmp then mv (same pattern as queue system)
        tmp_file = activity_file.with_suffix(".tmp")
        tmp_file.write_text(activity.model_dump_json(indent=2))
        tmp_file.rename(activity_file)

    def get_activity(self, agent_id: str) -> Optional[AgentActivity]:
        """Get agent's current activity state."""
        activity_file = self.activity_dir / f"{agent_id}.json"
        if not activity_file.exists():
            return None
        try:
            data = json.loads(activity_file.read_text())
            return AgentActivity(**data)
        except Exception:
            return None

    def get_all_activities(self) -> List[AgentActivity]:
        """Get all agent activities."""
        activities = []
        for activity_file in self.activity_dir.glob("*.json"):
            try:
                data = json.loads(activity_file.read_text())
                activities.append(AgentActivity(**data))
            except Exception:
                pass
        return activities

    def append_event(self, event: ActivityEvent) -> None:
        """Append event to activity stream with file locking to prevent race conditions."""
        # Use file locking for atomic read-modify-write
        with open(self.stream_file, 'a+') as f:
            if HAS_FCNTL:
                fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                content = f.read()
                events = self._parse_events(content)
                events.append(event)
                if len(events) > self.max_stream_events:
                    events = events[-self.max_stream_events:]
                f.seek(0)
                f.truncate()
                f.write('\n'.join(e.model_dump_json() for e in events) + '\n')
            finally:
                if HAS_FCNTL:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def _parse_events(self, content: str) -> List[ActivityEvent]:
        """Parse events from stream file content."""
        events = []
        for line in content.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    events.append(ActivityEvent(**data))
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.debug(f"Failed to parse activity event: {e}")
        return events

    def get_recent_events(self, limit: int = 10) -> List[ActivityEvent]:
        """Get recent activity events."""
        events = self._read_stream()
        return events[-limit:][::-1]  # Return last N in reverse order

    def _read_stream(self) -> List[ActivityEvent]:
        """Read all events from stream file."""
        if not self.stream_file.exists():
            return []

        events = []
        for line in self.stream_file.read_text().strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    events.append(ActivityEvent(**data))
                except Exception:
                    pass
        return events

    def _write_stream(self, events: List[ActivityEvent]) -> None:
        """Write events to stream file atomically."""
        lines = [event.model_dump_json() for event in events]

        # Atomic write: write to .tmp then mv
        tmp_file = self.stream_file.with_suffix(".tmp")
        tmp_file.write_text('\n'.join(lines) + '\n')
        tmp_file.rename(self.stream_file)
