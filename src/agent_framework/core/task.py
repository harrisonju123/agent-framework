"""Task model preserving the JSON schema from the Bash system."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class PlanDocument(BaseModel):
    """Structured architecture plan (inspired by PARA methodology)."""

    objectives: list[str]  # What we're trying to achieve
    approach: list[str]  # Step-by-step implementation approach
    risks: list[str] = Field(default_factory=list)  # Potential issues and mitigations
    success_criteria: list[str]  # How to verify the work is complete
    files_to_modify: list[str] = Field(default_factory=list)  # Affected files
    dependencies: list[str] = Field(default_factory=list)  # External dependencies


class TaskStatus(str, Enum):
    """Task status values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    """Task types that determine model selection."""
    # Cheap model (haiku)
    TESTING = "testing"
    VERIFICATION = "verification"
    QA_VERIFICATION = "qa_verification"
    FIX = "fix"
    BUGFIX = "bugfix"
    BUG_FIX = "bug-fix"
    COORDINATION = "coordination"
    STATUS_REPORT = "status_report"
    DOCUMENTATION = "documentation"

    # Default model (sonnet)
    IMPLEMENTATION = "implementation"
    ARCHITECTURE = "architecture"
    PLANNING = "planning"
    REVIEW = "review"
    ENHANCEMENT = "enhancement"
    PR_REQUEST = "pr_request"

    # Premium model (opus)
    ESCALATION = "escalation"


class Task(BaseModel):
    """Task model matching the Bash system's JSON schema."""

    # Required fields
    id: str
    type: TaskType
    status: TaskStatus
    priority: int
    created_by: str
    assigned_to: str
    created_at: datetime
    title: str
    description: str

    # Optional fields with defaults
    depends_on: list[str] = Field(default_factory=list)
    blocks: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)

    # Retry tracking
    retry_count: int = 0
    last_failed_at: Optional[datetime] = None

    # Execution tracking
    started_at: Optional[datetime] = None
    started_by: Optional[str] = None
    completed_at: Optional[datetime] = None
    completed_by: Optional[str] = None
    failed_at: Optional[datetime] = None
    failed_by: Optional[str] = None

    # Escalation tracking
    failed_task_id: Optional[str] = None
    needs_human_review: bool = False

    # Estimation
    estimated_effort: Optional[str] = None

    # Result tracking (for dependency context reuse)
    result_summary: Optional[str] = None
    last_error: Optional[str] = None

    # Optimization control (None=use config, True/False=force override)
    optimization_override: Optional[bool] = None
    optimization_override_reason: Optional[str] = None

    # Structured planning (PARA-inspired)
    plan: Optional[PlanDocument] = None

    class Config:
        """Pydantic config."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

    def mark_in_progress(self, agent_id: str) -> None:
        """Mark task as in progress."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.utcnow()
        self.started_by = agent_id

    def mark_completed(self, agent_id: str) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.completed_by = agent_id

    def mark_failed(self, agent_id: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.failed_at = datetime.utcnow()
        self.failed_by = agent_id

    def reset_to_pending(self) -> None:
        """Reset task to pending for retry with backoff."""
        self.status = TaskStatus.PENDING
        self.started_at = None
        self.started_by = None
        self.retry_count += 1
        self.last_failed_at = datetime.utcnow()
