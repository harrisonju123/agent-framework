"""Task model preserving the JSON schema from the Bash system."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional, List
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator


class RetryAttempt(BaseModel):
    """Record of a single retry attempt."""

    attempt_number: int
    timestamp: datetime
    error_message: str
    agent_id: str
    error_type: Optional[str] = None  # Categorized error type (network, validation, logic, etc.)
    context_snapshot: dict[str, Any] = Field(default_factory=dict)  # Relevant context at time of failure


class EscalationReport(BaseModel):
    """Structured escalation report with diagnostic information."""

    task_id: str
    original_title: str
    total_attempts: int
    attempt_history: List[RetryAttempt]
    root_cause_hypothesis: str  # AI-generated hypothesis about what went wrong
    suggested_interventions: List[str]  # Concrete actions a human can take
    failure_pattern: Optional[str] = None  # e.g., "intermittent", "consistent", "degrading"
    human_guidance: Optional[str] = None  # Human-provided guidance for retry


class PlanDocument(BaseModel):
    """Structured architecture plan (inspired by PARA methodology)."""

    objectives: list[str]  # What we're trying to achieve
    approach: list[str]  # Step-by-step implementation approach
    risks: list[str] = Field(default_factory=list)  # Potential issues and mitigations
    success_criteria: list[str]  # How to verify the work is complete
    files_to_modify: list[str] = Field(default_factory=list)  # Affected files
    dependencies: list[str] = Field(default_factory=list)  # External dependencies

    @model_validator(mode="before")
    @classmethod
    def coerce_dict_to_list(cls, data: Any) -> Any:
        """Prevent silent data corruption when LLM writes dicts instead of lists.

        Pydantic v2 lax mode iterates dicts (yielding only keys) for list[str],
        silently discarding the values. This validator intercepts that case.
        """
        if not isinstance(data, dict):
            return data
        list_fields = ("objectives", "approach", "risks", "success_criteria",
                       "files_to_modify", "dependencies")
        for field in list_fields:
            val = data.get(field)
            if isinstance(val, dict):
                val = list(val.values())
                data[field] = val
            if isinstance(val, list):
                coerced = []
                for item in val:
                    if isinstance(item, dict):
                        parts = [str(v) for v in item.values()]
                        coerced.append(" - ".join(parts))
                    else:
                        coerced.append(str(item) if not isinstance(item, str) else item)
                data[field] = coerced
        return data


class TaskStatus(str, Enum):
    """Task status values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    TESTING = "testing"  # Running tests in sandbox
    AWAITING_REVIEW = "awaiting_review"  # Tests passed, awaiting optional review
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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
    PREVIEW = "preview"

    # Premium model (opus)
    ESCALATION = "escalation"

    # Analysis workflow (default model)
    ANALYSIS = "analysis"


class Task(BaseModel):
    """Task model matching the Bash system's JSON schema."""

    model_config = ConfigDict(use_enum_values=True)

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

    # Parent-child hierarchy for task decomposition
    parent_task_id: Optional[str] = None
    subtask_ids: list[str] = Field(default_factory=list)
    decomposition_strategy: Optional[str] = None  # by_feature, by_layer, by_refactor_feature

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

    # Replan history for dynamic replanning on failure
    # Each entry: {"attempt": N, "error": "...", "revised_plan": "..."}
    replan_history: list[dict[str, Any]] = Field(default_factory=list)

    # Escalation tracking
    escalation_report: Optional[EscalationReport] = None
    retry_attempts: List[RetryAttempt] = Field(default_factory=list)

    @property
    def root_id(self) -> str:
        """The stable root identity across a chain of review hops.

        Chain tasks inherit _root_task_id from their parent so IDs like
        "review-...", "review-fix-...", and worktree keys always anchor on the
        original task rather than nesting another "chain-" prefix each cycle.
        """
        return self.context.get("_root_task_id", self.id)

    @field_serializer("created_at", "last_failed_at", "started_at", "completed_at", "failed_at")
    def serialize_datetime(self, v: Optional[datetime]) -> Optional[str]:
        return v.isoformat() if v else None

    def mark_in_progress(self, agent_id: str) -> None:
        """Mark task as in progress."""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now(UTC)
        self.started_by = agent_id

    def mark_completed(self, agent_id: str) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now(UTC)
        self.completed_by = agent_id

    def mark_failed(self, agent_id: str, error_message: Optional[str] = None, error_type: Optional[str] = None) -> None:
        """Mark task as failed and record attempt."""
        self.status = TaskStatus.FAILED
        self.failed_at = datetime.now(UTC)
        self.failed_by = agent_id

        # Record retry attempt
        if error_message:
            self.last_error = error_message
            attempt = RetryAttempt(
                attempt_number=self.retry_count + 1,
                timestamp=datetime.now(UTC),
                error_message=error_message,
                agent_id=agent_id,
                error_type=error_type,
                context_snapshot={
                    "task_type": self.type,
                    "assigned_to": self.assigned_to,
                    "has_dependencies": len(self.depends_on) > 0,
                }
            )
            self.retry_attempts.append(attempt)

    def mark_cancelled(self, cancelled_by: str, reason: Optional[str] = None) -> None:
        """Mark task as cancelled so it won't be retried."""
        self.status = TaskStatus.CANCELLED
        self.failed_at = datetime.now(UTC)
        self.failed_by = cancelled_by
        if reason:
            self.last_error = f"Cancelled: {reason}"
            self.notes.append(f"Cancelled by {cancelled_by}: {reason}")
        else:
            self.last_error = "Cancelled"
            self.notes.append(f"Cancelled by {cancelled_by}")

    def reset_to_pending(self) -> None:
        """Reset task to pending for retry with backoff."""
        self.status = TaskStatus.PENDING
        self.started_at = None
        self.started_by = None
        self.retry_count += 1
        self.last_failed_at = datetime.now(UTC)

