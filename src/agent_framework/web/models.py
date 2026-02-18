"""Pydantic models for web dashboard API responses."""

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional, List, Dict, Any

from pydantic import BaseModel, Field, field_validator


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


class ToolActivityData(BaseModel):
    """Current tool activity during LLM execution."""
    tool_name: str
    tool_input_summary: Optional[str] = None
    started_at: datetime
    tool_call_count: int = 0


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
    tool_activity: Optional[ToolActivityData] = None


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


class ActiveTaskData(BaseModel):
    """Active task (pending or in-progress) in a queue."""
    id: str
    title: str
    status: Literal["pending", "in_progress"]
    jira_key: Optional[str] = None
    assigned_to: str
    created_at: datetime
    started_at: Optional[datetime] = None
    task_type: str
    parent_task_id: Optional[str] = None


class CheckpointData(BaseModel):
    """Task awaiting checkpoint approval."""
    id: str
    title: str
    checkpoint_id: str
    checkpoint_message: str
    assigned_to: str
    paused_at: Optional[datetime] = None


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


class TeamSessionData(BaseModel):
    """Active Agent Team session."""
    team_name: str
    template: str
    started_at: Optional[datetime] = None
    source_task_id: Optional[str] = None
    status: str = "active"


class SpecializationCount(BaseModel):
    """Count of tasks processed under a given specialization profile."""
    profile: str
    count: int


class AgenticsMetrics(BaseModel):
    """Metrics for agentic features surfaced on the observability dashboard.

    Computed from session JSONL logs over a configurable time window.
    All rates are expressed as 0.0–1.0 floats; counts as integers.
    """
    # How often memory recall events were injected into prompts
    memory_recall_rate: float = 0.0
    memory_recalls_total: int = 0

    # Fraction of self-eval attempts that caught issues (verdict=FAIL)
    self_eval_catch_rate: float = 0.0
    self_eval_total: int = 0

    # How often tasks triggered a replan
    replan_trigger_rate: float = 0.0
    replan_total: int = 0
    # Fraction of replanned tasks that eventually completed (best-effort from session logs)
    replan_success_rate: float = 0.0

    # Distribution of engineer specialization profiles applied
    specialization_distribution: List[SpecializationCount] = []

    # Context budget: average utilization % across completed tasks
    avg_context_budget_utilization: float = 0.0
    context_budget_samples: int = 0

    # Wall-clock window used to compute these metrics (hours)
    window_hours: int = 24

    # Timestamp of computation
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DashboardState(BaseModel):
    """Complete dashboard state for WebSocket updates."""
    agents: List[AgentData]
    queues: List[QueueStats]
    events: List[EventData]
    failed_tasks: List[FailedTaskData]
    pending_checkpoints: List[CheckpointData]
    health: HealthReport
    is_paused: bool
    uptime_seconds: int
    active_teams: List[TeamSessionData] = []
    agentics_metrics: Optional[AgenticsMetrics] = None


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
    goal: str = Field(..., min_length=1, max_length=2000)
    repository: str = Field(..., pattern=r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$')
    workflow: str = Field(default="default")

    @field_validator("workflow")
    @classmethod
    def normalize_workflow(cls, v: str) -> str:
        if v in ("simple", "standard", "full"):
            return "default"
        return v


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


class CreateTaskRequest(BaseModel):
    """Request to create a task directly in a queue."""
    title: str = Field(..., min_length=1, max_length=500)
    description: str = Field(..., min_length=1, max_length=10000)
    task_type: str = Field(default="implementation")
    assigned_to: str = Field(default="engineer", pattern=r'^[a-z0-9_-]+$')
    repository: Optional[str] = Field(default=None, pattern=r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$')
    priority: int = Field(default=1, ge=1, le=10)


class CheckpointRejectRequest(BaseModel):
    """Request to reject a checkpoint with feedback."""
    feedback: str = Field(..., min_length=1, max_length=5000)


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


# Setup wizard models

class JIRAValidationRequest(BaseModel):
    """Request to validate JIRA credentials."""
    server: str
    email: str
    api_token: str
    project: Optional[str] = None


class JIRAValidationResponse(BaseModel):
    """Response from JIRA validation."""
    valid: bool
    message: str
    user_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class GitHubValidationRequest(BaseModel):
    """Request to validate GitHub token."""
    token: str


class GitHubValidationResponse(BaseModel):
    """Response from GitHub validation."""
    valid: bool
    user: Optional[str] = None
    rate_limit: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class RepositoryConfig(BaseModel):
    """Repository configuration for setup."""
    github_repo: str = Field(..., pattern=r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$')
    jira_project: str
    name: str


class SetupConfiguration(BaseModel):
    """Complete setup configuration."""
    jira: JIRAValidationRequest
    github: GitHubValidationRequest
    repositories: List[RepositoryConfig]
    enable_mcp: bool = False


class SetupStatusResponse(BaseModel):
    """Setup completion status."""
    initialized: bool
    jira_configured: bool
    github_configured: bool
    repositories_registered: int
    mcp_enabled: bool
    ready_to_start: bool
