"""Escalation handler for failed tasks (ported from Bash system)."""

import re
import time
from datetime import UTC, datetime
from typing import List, Optional

from ..core.task import Task, TaskStatus, TaskType, EscalationReport, RetryAttempt
from ..utils.type_helpers import get_type_str


class EscalationHandler:
    """
    Handles task escalations for human review.

    Ported from scripts/async-agent-runner.sh lines 205-246.

    CRITICAL SAFETY:
    - Escalation tasks with type "escalation" CANNOT create more escalations
    - This is the single most important safeguard against infinite loops
    - If an escalation fails 5 times, it logs for human intervention only
    """

    def __init__(self, escalation_queue: str = "architect", enable_error_truncation: bool = False):
        self.escalation_queue = escalation_queue
        self.enable_error_truncation = enable_error_truncation

        # Error pattern categorization
        self._error_patterns = {
            "network": [
                r"connection.*refused",
                r"timeout",
                r"network.*unreachable",
                r"dns.*fail",
                r"could not resolve host",
            ],
            "authentication": [
                r"unauthorized",
                r"authentication.*fail",
                r"invalid.*credential",
                r"permission.*denied",
                r"403|401",
            ],
            "validation": [
                r"validation.*error",
                r"invalid.*input",
                r"schema.*mismatch",
                r"type.*error",
                r"missing required",
            ],
            "resource": [
                r"out of memory",
                r"disk.*full",
                r"too many.*open files",
                r"resource.*exhausted",
            ],
            "logic": [
                r"null.*reference",
                r"index.*out of.*range",
                r"assertion.*fail",
                r"unexpected.*state",
            ],
            "budget": [
                r"budget.*exceed",
                r"max budget",
                r"quota.*exceed",
                r"insufficient.*credits",
                r"usage.*limit.*exceed",
            ],
        }

    def truncate_error(self, error: str, max_lines: int = 35) -> str:
        """
        Intelligently truncate error messages.

        Implements Strategy 8 (Error Truncation) from the optimization plan.

        Handles multiple error formats:
        - Stack traces (preserves error type, head, tail)
        - Single-line errors (returned as-is)
        - JSON error responses (preserved)
        - Already truncated errors (not re-truncated)

        Expected savings: 3-7KB per escalation task (~50% reduction).
        """
        if not error:
            return "No error message available"

        lines = error.split('\n')

        # Don't truncate if already small enough
        if len(lines) <= max_lines:
            return error

        # Check if already truncated
        if "lines omitted" in error or "..." in error[:100]:
            return error

        # Try to find error type - search more lines to catch context before error
        error_type = ""
        for line in lines[:10]:  # Increased from 3 to 10 lines
            if any(marker in line for marker in ["Error:", "Exception:", "Traceback", "FAILED", "ERROR"]):
                error_type = line.strip()
                break

        # Keep only meaningful lines (skip empty/whitespace-only)
        meaningful_lines = [line for line in lines if line.strip()]

        # If filtering made it small enough, return it
        if len(meaningful_lines) <= max_lines:
            return '\n'.join(meaningful_lines)

        # Truncate: keep first 20 and last 10 meaningful lines
        head_lines = 20
        tail_lines = 10

        head = meaningful_lines[:head_lines]
        tail = meaningful_lines[-tail_lines:]
        omitted = len(meaningful_lines) - (head_lines + tail_lines)

        result = []
        if error_type:
            result.append(error_type)
            result.append("")  # Blank line for readability

        result.extend(head)
        result.append("")
        result.append(f"... ({omitted} lines omitted) ...")
        result.append("")
        result.extend(tail)

        return '\n'.join(result)

    def categorize_error(self, error_message: str) -> Optional[str]:
        """Categorize error message by pattern matching.

        Returns one of: network, authentication, validation, resource, logic, unknown.
        Returns None for empty input.
        """
        if not error_message:
            return None
        error_lower = error_message.lower()
        for category, patterns in self._error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, error_lower):
                    return category
        return "unknown"

    def _analyze_failure_pattern(self, attempts: List[RetryAttempt]) -> str:
        """Analyze retry attempts to determine failure pattern."""
        if not attempts:
            return "single_failure"

        error_types = [a.error_type for a in attempts if a.error_type]
        if not error_types:
            return "unknown_pattern"

        # All same error type = consistent failure
        if len(set(error_types)) == 1:
            return "consistent"

        # Multiple error types = intermittent or degrading
        # Check if errors are becoming more severe
        network_count = error_types.count("network")
        if network_count > len(error_types) / 2:
            return "intermittent_network"

        return "varied"

    def _generate_root_cause_hypothesis(self, failed_task: Task) -> str:
        """Generate hypothesis about root cause based on retry history."""
        if not failed_task.retry_attempts:
            return "Task failed without retry attempts recorded. Error may be immediate or critical."

        pattern = self._analyze_failure_pattern(failed_task.retry_attempts)
        error_types = [a.error_type for a in failed_task.retry_attempts if a.error_type]
        most_common = max(set(error_types), key=error_types.count) if error_types else "unknown"

        hypotheses = {
            "consistent": f"Consistent {most_common} errors across all attempts suggest a fundamental issue that won't resolve with retries.",
            "intermittent_network": "Intermittent network failures suggest infrastructure or connectivity issues rather than code problems.",
            "varied": "Different error types across attempts suggest environmental instability or race conditions.",
            "single_failure": "Single catastrophic failure suggests immediate blocker or invalid configuration.",
        }

        base = hypotheses.get(pattern, "Unable to determine clear pattern from retry history.")

        # Add context from last error
        if failed_task.last_error:
            last_attempt = failed_task.retry_attempts[-1] if failed_task.retry_attempts else None
            if last_attempt and last_attempt.error_type:
                base += f" Last failure type: {last_attempt.error_type}."

        return base

    def _generate_suggested_interventions(self, failed_task: Task) -> List[str]:
        """Generate actionable suggestions based on failure analysis."""
        suggestions = []
        error_types = [a.error_type for a in failed_task.retry_attempts if a.error_type]

        if not error_types:
            suggestions.append("Review task logs to identify failure type")
            suggestions.append("Check task configuration and dependencies")
            return suggestions

        most_common = max(set(error_types), key=error_types.count)

        intervention_map = {
            "network": [
                "Check network connectivity and firewall rules",
                "Verify API endpoints are accessible",
                "Review rate limiting and retry backoff settings",
                "Consider increasing timeout values",
            ],
            "authentication": [
                "Verify credentials are valid and not expired",
                "Check token refresh mechanisms",
                "Review permission levels for required operations",
                "Validate API keys and secrets configuration",
            ],
            "validation": [
                "Review input data format and schema",
                "Check for recent API or contract changes",
                "Validate all required fields are populated",
                "Review type conversions and serialization",
            ],
            "resource": [
                "Check available system resources (memory, disk, file descriptors)",
                "Review resource limits and quotas",
                "Consider scaling infrastructure",
                "Look for resource leaks or cleanup issues",
            ],
            "logic": [
                "Review code logic for edge cases",
                "Check assumptions about data state",
                "Add defensive null checks and bounds validation",
                "Review recent code changes for regressions",
            ],
            "budget": [
                "Check API usage and billing dashboard",
                "Review token budget configuration in agent settings",
                "Consider upgrading API tier or purchasing additional credits",
                "Optimize prompts to reduce token consumption",
                "Contact support for budget limit adjustment",
            ],
        }

        suggestions.extend(intervention_map.get(most_common, [
            "Review error messages and stack traces",
            "Check recent system or dependency changes",
            "Consider manual reproduction in isolated environment",
        ]))

        # Add retry-specific guidance if multiple attempts
        if len(failed_task.retry_attempts) > 1:
            suggestions.append(f"Task failed {len(failed_task.retry_attempts)} times - consider if retries are appropriate")

        return suggestions[:5]  # Limit to top 5 suggestions

    def _build_escalation_report(self, failed_task: Task) -> EscalationReport:
        """Build structured escalation report with diagnostics."""
        return EscalationReport(
            task_id=failed_task.id,
            original_title=failed_task.title,
            total_attempts=len(failed_task.retry_attempts),
            attempt_history=failed_task.retry_attempts,
            root_cause_hypothesis=self._generate_root_cause_hypothesis(failed_task),
            suggested_interventions=self._generate_suggested_interventions(failed_task),
            failure_pattern=self._analyze_failure_pattern(failed_task.retry_attempts),
        )

    def create_escalation(self, failed_task: Task, agent_id: str) -> Task:
        """
        Create an escalation task for a failed task.

        CRITICAL: This method should NEVER be called for tasks with type="escalation"
        """
        if failed_task.type == TaskType.ESCALATION:
            raise ValueError(
                "CRITICAL: Cannot create escalation for escalation task. "
                "This would cause an infinite loop."
            )

        escalation_id = f"escalation-{int(time.time())}-{failed_task.id}"

        # Truncate error if enabled (Strategy 8: Error Truncation)
        error_msg = failed_task.last_error or "Unknown error"
        if self.enable_error_truncation:
            error_msg = self.truncate_error(error_msg)

        # Build structured escalation report
        escalation_report = self._build_escalation_report(failed_task)

        escalation = Task(
            id=escalation_id,
            type=TaskType.ESCALATION,
            status=TaskStatus.PENDING,
            priority=0,  # Highest priority
            created_by=agent_id,
            assigned_to=self.escalation_queue,
            created_at=datetime.now(UTC),
            title=f"ESCALATION: Task failed after {failed_task.retry_count} retries",
            description=self._build_description(failed_task, escalation_report),
            failed_task_id=failed_task.id,
            needs_human_review=True,
            escalation_report=escalation_report,
            context={
                "original_task_id": failed_task.id,
                "original_task_type": get_type_str(failed_task.type),
                "retry_count": failed_task.retry_count,
                "error": error_msg,
                "failure_pattern": escalation_report.failure_pattern,
                "root_cause_hypothesis": escalation_report.root_cause_hypothesis,
            },
        )

        return escalation

    def _build_description(self, failed_task: Task, report: EscalationReport) -> str:
        """Build escalation description with structured report."""
        desc = (
            f"Task {failed_task.id} failed after {failed_task.retry_count} retry attempts "
            f"and has been marked as failed. This requires human intervention or product decision.\n\n"
            f"## Original Task\n"
            f"Title: {failed_task.title}\n"
            f"Type: {get_type_str(failed_task.type)}\n\n"
        )

        # Add root cause analysis
        desc += f"## Root Cause Analysis\n"
        desc += f"**Failure Pattern**: {report.failure_pattern}\n\n"
        desc += f"**Hypothesis**: {report.root_cause_hypothesis}\n\n"

        # Add attempt history summary
        if report.attempt_history:
            desc += f"## Attempt History ({len(report.attempt_history)} attempts)\n"
            for attempt in report.attempt_history[-3:]:  # Show last 3 attempts
                desc += f"- **Attempt {attempt.attempt_number}** ({attempt.timestamp.strftime('%Y-%m-%d %H:%M:%S')})\n"
                desc += f"  Agent: {attempt.agent_id}, Type: {attempt.error_type or 'unknown'}\n"
                error_preview = attempt.error_message[:100] + "..." if len(attempt.error_message) > 100 else attempt.error_message
                desc += f"  Error: {error_preview}\n\n"

        # Add suggested interventions
        desc += f"## Suggested Interventions\n"
        for i, intervention in enumerate(report.suggested_interventions, 1):
            desc += f"{i}. {intervention}\n"

        desc += f"\n## Next Steps\n"
        desc += f"Use `agent guide {failed_task.id} --hint \"<your guidance>\"` to inject human guidance and retry.\n"
        desc += f"Or review the failed task manually and decide next steps.\n"

        return desc
