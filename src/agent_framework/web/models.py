"""Pydantic models for web dashboard API responses."""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


class AgentStatusEnum(str, Enum):
    """Agent operational status."""
    IDLE = "idle"
    WORKING = "working"
    COMPLETING = "completing"
    DEAD = "dead"


class TaskPhaseEnum(str, Enum):
    """Task execution phases."""
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING_LLM = "executing_llm"
    IMPLEMENTING = "implementing"
    TESTING = "testing"
    EXPLORING_CODEBASE = "exploring_codebase"
    CREATING_EPIC = "creating_epic"
    CREATING_SUBTASKS = "creating_subtasks"
    QUEUING_TASKS = "queuing_tasks"
    COMMITTING = "committing"
    CREATING_PR = "creating_pr"
    UPDATING_JIRA = "updating_jira"


class CurrentTaskData(BaseModel):
    """Current task being processed."""
    id: str
    title: str
    type: str
    started_at: datetime


class PhaseData(BaseModel):
    """Phase execution data."""
    name: str
    completed: bool
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class AgentData(BaseModel):
    """Agent data for API response."""
    id: str
    name: str
    queue: str
    status: AgentStatusEnum
    current_task: Optional[CurrentTaskData] = None
    current_phase: Optional[str] = None
    phases_completed: int = 0
    elapsed_seconds: Optional[int] = None
    last_updated: Optional[datetime] = None


class QueueStats(BaseModel):
    """Queue statistics."""
    queue_id: str
    agent_name: str
    pending_count: int
    oldest_task_age: Optional[int] = None  # seconds


class EventData(BaseModel):
    """Activity event data."""
    type: str  # "start", "complete", "fail", "phase"
    agent: str
    task_id: str
    title: str
    timestamp: datetime
    duration_ms: Optional[int] = None
    retry_count: Optional[int] = None
    phase: Optional[str] = None
    error_message: Optional[str] = None
    pr_url: Optional[str] = None


class FailedTaskData(BaseModel):
    """Failed task data."""
    id: str
    title: str
    jira_key: Optional[str] = None
    assigned_to: str
    retry_count: int
    last_error: Optional[str] = None
    failed_at: Optional[datetime] = None


class HealthCheck(BaseModel):
    """Individual health check result."""
    name: str
    passed: bool
    message: Optional[str] = None


class HealthReport(BaseModel):
    """System health report."""
    passed: bool
    checks: List[HealthCheck]
    warnings: List[str] = []


class DashboardState(BaseModel):
    """Complete dashboard state for WebSocket updates."""
    agents: List[AgentData]
    queues: List[QueueStats]
    events: List[EventData]
    failed_tasks: List[FailedTaskData]
    health: HealthReport
    is_paused: bool
    uptime_seconds: int


# API Response models

class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool = True
    message: str


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = False
    error: str
    detail: Optional[str] = None


class AgentActionResponse(BaseModel):
    """Response for agent actions (start/stop/restart)."""
    success: bool
    agent_id: str
    action: str
    message: str


class TaskActionResponse(BaseModel):
    """Response for task actions (retry)."""
    success: bool
    task_id: str
    action: str
    message: str


# Operation request models

class WorkRequest(BaseModel):
    """Request to create new work (like CLI `agent work`)."""
    goal: str = Field(..., min_length=10, max_length=2000)
    repository: str = Field(..., pattern=r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$')
    workflow: str = Field(default="simple", pattern=r'^(simple|standard|full)$')


class AnalyzeRequest(BaseModel):
    """Request to analyze a repository (like CLI `agent analyze`)."""
    repository: str = Field(..., pattern=r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$')
    severity: str = Field(default="high", pattern=r'^(all|critical|high|medium)$')
    max_issues: int = Field(default=50, ge=1, le=500)
    dry_run: bool = False
    focus: Optional[str] = Field(default=None, max_length=5000)


class RunTicketRequest(BaseModel):
    """Request to queue a specific JIRA ticket."""
    ticket_id: str = Field(..., pattern=r'^[A-Z]+-\d+$')
    agent: Optional[str] = Field(default=None, pattern=r'^[a-z0-9_-]+$')


class OperationResponse(BaseModel):
    """Response for operation endpoints."""
    success: bool
    task_id: Optional[str] = None
    message: str


class LogEntry(BaseModel):
    """Single log line entry."""
    agent: str
    task_id: Optional[str] = None  # For claude-cli logs, links to specific task
    source: Optional[str] = None  # 'agent' or 'claude-cli'
    line: str
    timestamp: datetime
    level: Optional[str] = None
