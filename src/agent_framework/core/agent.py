"""Agent polling loop implementation (ported from Bash system)."""

import asyncio
import hashlib
import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Optional

from .task import Task, TaskStatus, TaskType
from .activity import ActivityManager, AgentActivity, AgentStatus, CurrentTask, ActivityEvent, TaskPhase
from ..llm.base import LLMBackend, LLMRequest, LLMResponse
from ..queue.file_queue import FileQueue
from ..safeguards.retry_handler import RetryHandler
from ..safeguards.escalation import EscalationHandler


logger = logging.getLogger(__name__)

# Constants for optimization strategies
SUMMARY_CONTEXT_MAX_CHARS = 2000
SUMMARY_MAX_LENGTH = 500
ERROR_HEAD_LINES = 20
ERROR_TAIL_LINES = 10
BUDGET_WARNING_THRESHOLD = 1.3  # 30% over budget

# Model pricing (per 1M tokens, as of 2025-01)
MODEL_PRICING = {
    "haiku": {"input": 0.25, "output": 1.25},
    "sonnet": {"input": 3.0, "output": 15.0},
    "opus": {"input": 15.0, "output": 75.0},
}


def _get_type_str(task_type) -> str:
    """Get string value from task type (handles both enum and string)."""
    return task_type.value if hasattr(task_type, 'value') else str(task_type)


@dataclass
class AgentConfig:
    """Agent configuration."""
    id: str
    name: str
    queue: str
    prompt: str
    poll_interval: int = 30
    max_retries: int = 5
    timeout: int = 1800


class Agent:
    """
    Agent with polling loop for processing tasks.

    Ported from scripts/async-agent-runner.sh with the main polling loop
    at lines 254-407.
    """

    def __init__(
        self,
        config: AgentConfig,
        llm: LLMBackend,
        queue: FileQueue,
        workspace: Path,
        jira_client=None,
        github_client=None,
        multi_repo_manager=None,
        jira_config=None,
        github_config=None,
        mcp_enabled: bool = False,
        optimization_config: Optional[dict] = None,
    ):
        self.config = config
        self.llm = llm
        self.queue = queue
        self.workspace = Path(workspace)
        self.jira_client = jira_client
        self.github_client = github_client
        self.multi_repo_manager = multi_repo_manager
        self.jira_config = jira_config
        self.github_config = github_config
        self._mcp_enabled = mcp_enabled
        self._running = False
        self._current_task_id: Optional[str] = None

        # Optimization configuration (sanitize then make immutable for thread safety)
        sanitized_config = self._sanitize_optimization_config(optimization_config or {})
        self._optimization_config = MappingProxyType(sanitized_config)

        # Log active optimizations on startup
        logger.info(f"Agent {self.config.id} optimization config: {self._get_active_optimizations()}")

        # Initialize safeguards
        self.retry_handler = RetryHandler(max_retries=config.max_retries)
        enable_error_truncation = self._optimization_config.get("enable_error_truncation", False)
        self.escalation_handler = EscalationHandler(enable_error_truncation=enable_error_truncation)

        # Heartbeat file
        self.heartbeat_file = self.workspace / ".agent-communication" / "heartbeats" / config.id

        # Activity tracking
        self.activity_manager = ActivityManager(workspace)

    async def run(self) -> None:
        """
        Main polling loop.

        Ported from scripts/async-agent-runner.sh lines 254-407.
        """
        self._running = True
        logger.info(f"Starting {self.config.id} runner")

        # Write initial IDLE state when agent starts
        from datetime import datetime
        self.activity_manager.update_activity(AgentActivity(
            agent_id=self.config.id,
            status=AgentStatus.IDLE,
            last_updated=datetime.utcnow()
        ))

        while self._running:
            # Write heartbeat every iteration
            self._write_heartbeat()

            # Poll for next task
            task = self.queue.pop(self.config.queue)

            if task:
                await self._handle_task(task)
            else:
                logger.debug(
                    f"No tasks available for {self.config.id}, "
                    f"sleeping for {self.config.poll_interval}s"
                )

            await asyncio.sleep(self.config.poll_interval)

    async def stop(self) -> None:
        """Stop the polling loop gracefully."""
        logger.info(f"Stopping {self.config.id}")
        self._running = False

        # Release current task lock if any
        if self._current_task_id:
            logger.warning(
                f"Releasing lock for current task: {self._current_task_id}"
            )
            # Lock will be automatically released by FileLock context manager

        # Write final heartbeat
        self._write_heartbeat()

    async def _handle_task(self, task: Task) -> None:
        """Handle task execution with retry/escalation logic."""
        from datetime import datetime

        logger.info(f"Found task: {task.id} - {task.title}")

        # Try to acquire lock
        lock = self.queue.acquire_lock(task.id, self.config.id)
        if not lock:
            logger.warning(f"Could not acquire lock for {task.id}, will retry later")
            return

        self._current_task_id = task.id
        task_start_time = datetime.utcnow()

        try:
            # Mark in progress
            task.mark_in_progress(self.config.id)
            self.queue.update(task)

            # Update activity: Started
            self.activity_manager.update_activity(AgentActivity(
                agent_id=self.config.id,
                status=AgentStatus.WORKING,
                current_task=CurrentTask(
                    id=task.id,
                    title=task.title,
                    type=_get_type_str(task.type),
                    started_at=task_start_time
                ),
                current_phase=TaskPhase.ANALYZING,
                last_updated=datetime.utcnow()
            ))

            # Append start event
            self.activity_manager.append_event(ActivityEvent(
                type="start",
                agent=self.config.id,
                task_id=task.id,
                title=task.title,
                timestamp=datetime.utcnow()
            ))

            # Build prompt
            prompt = self._build_prompt(task)

            # Update phase: Executing LLM (this is where most time is spent)
            self._update_phase(TaskPhase.EXECUTING_LLM)

            # Execute with LLM
            logger.info(
                f"Processing task {task.id} with model "
                f"(type: {task.type}, retries: {task.retry_count})"
            )

            response = await self.llm.complete(
                LLMRequest(
                    prompt=prompt,
                    task_type=task.type,
                    retry_count=task.retry_count,
                    context=task.context,
                )
            )

            if response.success:
                # Extract summary from response (Strategy 5: Result Summarization)
                if self._optimization_config.get("enable_result_summarization", False):
                    summary = await self._extract_summary(response.content, task)
                    task.result_summary = summary
                    logger.debug(f"Task {task.id} summary: {summary}")

                # Handle post-LLM workflow (git/PR/JIRA)
                await self._handle_success(task, response)

                # Task completed successfully
                task.mark_completed(self.config.id)
                self.queue.mark_completed(task)

                # Log token usage (Strategy 6: Token Tracking)
                total_tokens = response.input_tokens + response.output_tokens
                budget = self._get_token_budget(task.type)
                cost = self._estimate_cost(response)

                logger.info(
                    f"Completed task: {task.id} - "
                    f"tokens: {response.input_tokens} in, {response.output_tokens} out, "
                    f"total: {total_tokens}, budget: {budget}, estimated cost: ${cost:.2f}"
                )

                # Soft limit warning (don't fail, just alert)
                if self._optimization_config.get("enable_token_budget_warnings", False):
                    threshold = self._optimization_config.get("budget_warning_threshold", BUDGET_WARNING_THRESHOLD)
                    if total_tokens > budget * threshold:
                        logger.warning(
                            f"Task {task.id} EXCEEDED TOKEN BUDGET: "
                            f"{total_tokens} tokens (budget: {budget}, "
                            f"{int(threshold * 100)}% threshold: {budget * threshold:.0f})"
                        )
                        # Append to activity metrics for dashboard
                        self.activity_manager.append_event(ActivityEvent(
                            type="token_budget_exceeded",
                            agent=self.config.id,
                            task_id=task.id,
                            title=f"Token budget exceeded: {total_tokens} > {budget}",
                            timestamp=datetime.utcnow()
                        ))

                # Append complete event with duration
                duration_ms = int((datetime.utcnow() - task_start_time).total_seconds() * 1000)
                self.activity_manager.append_event(ActivityEvent(
                    type="complete",
                    agent=self.config.id,
                    task_id=task.id,
                    title=task.title,
                    timestamp=datetime.utcnow(),
                    duration_ms=duration_ms
                ))
            else:
                # Task failed - store error for escalation
                task.last_error = response.error or "Unknown error"
                logger.error(
                    f"LLM failed for task {task.id}: {task.last_error}"
                )

                # Append fail event
                self.activity_manager.append_event(ActivityEvent(
                    type="fail",
                    agent=self.config.id,
                    task_id=task.id,
                    title=task.title,
                    timestamp=datetime.utcnow(),
                    retry_count=task.retry_count
                ))

                await self._handle_failure(task)

        except Exception as e:
            # Store exception for escalation
            task.last_error = str(e)
            logger.exception(f"Error processing task {task.id}: {e}")

            # Append fail event for exceptions
            self.activity_manager.append_event(ActivityEvent(
                type="fail",
                agent=self.config.id,
                task_id=task.id,
                title=task.title,
                timestamp=datetime.utcnow(),
                retry_count=task.retry_count
            ))

            await self._handle_failure(task)

        finally:
            # Clear activity - back to IDLE
            self.activity_manager.update_activity(AgentActivity(
                agent_id=self.config.id,
                status=AgentStatus.IDLE,
                last_updated=datetime.utcnow()
            ))

            self.queue.release_lock(lock)
            self._current_task_id = None

    async def _handle_failure(self, task: Task) -> None:
        """
        Handle task failure with retry/escalation logic.

        Ported from scripts/async-agent-runner.sh lines 374-394.
        """
        if task.retry_count >= self.retry_handler.max_retries:
            # Max retries exceeded - mark as failed
            logger.error(
                f"Task {task.id} has failed {task.retry_count} times "
                f"(max: {self.retry_handler.max_retries})"
            )
            task.mark_failed(self.config.id)
            self.queue.mark_failed(task)

            # CRITICAL: Prevent infinite loop - escalations should NOT create more escalations
            if self.retry_handler.can_create_escalation(task):
                escalation = self.escalation_handler.create_escalation(
                    task, self.config.id
                )
                self.queue.push(escalation, escalation.assigned_to)
                logger.warning(
                    f"Created escalation task {escalation.id} for failed task {task.id}"
                )
            else:
                logger.error(
                    f"Escalation task {task.id} failed after {task.retry_count} retries - "
                    "NOT creating another escalation (would cause infinite loop). "
                    "This escalation requires immediate human intervention."
                )
        else:
            # Reset task to pending so it can be retried
            logger.warning(
                f"Resetting task {task.id} to pending status "
                f"(retry {task.retry_count + 1}/{self.retry_handler.max_retries})"
            )
            task.reset_to_pending()
            self.queue.update(task)

    def _sanitize_optimization_config(self, config: dict) -> dict:
        """
        Sanitize optimization config before making immutable.

        Validates and corrects invalid values, warns about issues.
        """
        config = config.copy()  # Don't modify input

        # Clamp canary percentage to valid range
        canary = config.get("canary_percentage", 0)
        if not 0 <= canary <= 100:
            logger.warning(f"Invalid canary_percentage: {canary}, clamping to [0, 100]")
            config["canary_percentage"] = max(0, min(100, canary))

        # Warn about incompatible flag combinations
        if config.get("shadow_mode") and config.get("canary_percentage", 0) > 0:
            logger.warning(
                "shadow_mode and canary_percentage both enabled. "
                "Shadow mode will use legacy prompts regardless of canary setting."
            )

        return config

    def _get_active_optimizations(self) -> Dict[str, Any]:
        """Get dict of which optimizations are currently active."""
        return {
            "minimal_prompts": self._optimization_config.get("enable_minimal_prompts", False),
            "compact_json": self._optimization_config.get("enable_compact_json", False),
            "context_dedup": self._optimization_config.get("enable_context_deduplication", False),
            "token_tracking": self._optimization_config.get("enable_token_tracking", False),
            "budget_warnings": self._optimization_config.get("enable_token_budget_warnings", False),
            "result_summarization": self._optimization_config.get("enable_result_summarization", False),
            "error_truncation": self._optimization_config.get("enable_error_truncation", False),
            "shadow_mode": self._optimization_config.get("shadow_mode", False),
            "canary_percentage": self._optimization_config.get("canary_percentage", 0),
        }

    def _should_use_optimization(self, task: Task) -> bool:
        """
        Determine if task should use optimizations based on canary percentage.

        Uses deterministic hash-based selection for consistent behavior.
        Task-level overrides take precedence over canary selection.
        """
        # Check for task-level override first
        if hasattr(task, 'optimization_override') and task.optimization_override is not None:
            reason = getattr(task, 'optimization_override_reason', 'no reason given')
            logger.info(
                f"Task {task.id} optimization override: {task.optimization_override} ({reason})"
            )
            return task.optimization_override

        canary_pct = self._optimization_config.get("canary_percentage", 0)

        if canary_pct == 0:
            return False
        elif canary_pct >= 100:
            return True
        else:
            # Use deterministic hash for consistent selection
            task_hash = int(hashlib.md5(task.id.encode()).hexdigest()[:8], 16)
            return (task_hash % 100) < canary_pct

    def _get_minimal_task_dict(self, task: Task) -> Dict[str, Any]:
        """
        Extract only prompt-relevant task fields.

        Implements Strategy 1 (Minimal Task Prompts) from the optimization plan.
        See OPTIMIZATION_IMPLEMENTATION_SUMMARY.md for details.

        Omits metadata fields that don't contribute to task execution:
        - Timestamps (created_at, started_at, completed_at)
        - Internal tracking (created_by, assigned_to, retry_count)
        - Dependency arrays (depends_on, blocks)

        Expected savings: 3-8KB per task (40-50% reduction).
        """
        # Validate essential fields
        if not task.title or not task.description:
            logger.warning(
                f"Task {task.id} missing essential fields: "
                f"title={bool(task.title)}, description={bool(task.description)}. "
                f"Falling back to full task dict."
            )
            return task.model_dump()

        minimal = {
            "title": task.title.strip(),
            "description": task.description.strip(),
            "type": _get_type_str(task.type),
        }

        # Include acceptance criteria and deliverables if present
        if task.acceptance_criteria:
            minimal["acceptance_criteria"] = task.acceptance_criteria
        if task.deliverables:
            minimal["deliverables"] = task.deliverables

        # Include notes if non-empty (can contain important context)
        if task.notes:
            minimal["notes"] = task.notes

        # Include only relevant context keys
        relevant_context = {}
        for key in ["jira_key", "jira_project", "github_repo", "mode", "user_goal", "repository_name", "epic_key"]:
            if key in task.context:
                relevant_context[key] = task.context[key]

        if relevant_context:
            minimal["context"] = relevant_context

        return minimal

    def _get_token_budget(self, task_type: TaskType) -> int:
        """
        Get expected token budget for task type.

        Implements Strategy 6 (Token Tracking) from the optimization plan.

        Budgets can be configured via optimization.token_budgets in config,
        otherwise uses sensible defaults.
        """
        # Default budgets
        default_budgets = {
            "planning": 30000,
            "implementation": 50000,
            "testing": 20000,
            "escalation": 80000,
            "review": 25000,
            "architecture": 40000,
            "coordination": 15000,
            "documentation": 15000,
            "fix": 30000,
            "bugfix": 30000,
            "bug-fix": 30000,
            "verification": 20000,
            "status_report": 10000,
            "enhancement": 40000,
        }

        # Get budget from config or use default
        budget_key = task_type.value.lower().replace("-", "_")
        configured_budgets = self._optimization_config.get("token_budgets", {})

        return configured_budgets.get(budget_key, default_budgets.get(budget_key, 40000))

    async def _extract_summary(self, response: str, task: Task, _recursion_depth: int = 0) -> str:
        """
        Extract key outcomes from agent response.

        Implements Strategy 5 (Result Summarization) from the optimization plan.

        Uses two-tier extraction:
        1. Regex patterns (free, fast) - extracts JIRA keys, PR URLs, file paths
        2. Haiku fallback (cheap) - only if regex insufficient

        Args:
            _recursion_depth: Defensive guard to prevent recursion. Currently
                            not used since we don't recurse, but kept for safety.

        Expected savings: 20-30% on follow-up tasks through context reuse.
        """
        # Guard against empty response
        if not response or not response.strip():
            return f"Task {_get_type_str(task.type)} completed (no output)"

        # Defensive guard - prevents recursion even though we don't recurse currently
        if _recursion_depth > 0:
            logger.debug("Recursion depth exceeded in summary extraction, using fallback")
            return f"Task {_get_type_str(task.type)} completed"

        # Try regex extraction first (fast, no cost)
        extracted = []

        # Extract JIRA keys (2-5 char project, 1-6 digit ticket - more realistic)
        jira_keys = re.findall(r'\b([A-Z]{2,5}-\d{1,6})\b', response)
        if jira_keys:
            # Deduplicate and limit
            jira_keys = list(set(jira_keys))[:10]
            extracted.append(f"Created/Updated: {', '.join(jira_keys)}")

        # Extract PR URLs
        pr_urls = re.findall(r'github\.com/[^/]+/[^/]+/pull/(\d+)', response)
        if pr_urls:
            pr_urls = list(set(pr_urls))[:5]
            extracted.append(f"PRs: {', '.join(pr_urls)}")

        # Extract file paths (comprehensive pattern)
        file_paths = re.findall(
            r'\b(?:src|lib|app|tests?|pkg|internal|cmd)/[^\s:;,]+\.(?:py|ts|tsx|js|jsx|go|rb|java|kt)',
            response
        )
        if file_paths:
            # Deduplicate and limit
            file_paths = list(set(file_paths))[:5]
            extracted.append(f"Modified: {', '.join(file_paths)}")

        # If regex extraction found enough, use that
        if len(extracted) >= 2:
            return " | ".join(extracted)

        # Fall back to Haiku for summarization (only if enabled and insufficient data)
        if self._optimization_config.get("enable_result_summarization", False):
            summary_prompt = f"Summarize key outcomes in 3 bullet points:\n{response[:SUMMARY_CONTEXT_MAX_CHARS]}"
            try:
                summary_response = await self.llm.complete(LLMRequest(
                    prompt=summary_prompt,
                    model="haiku"  # Cheap model for summaries
                ))

                # Verify response is valid before returning
                if summary_response.success and summary_response.content:
                    return summary_response.content[:SUMMARY_MAX_LENGTH]
                else:
                    logger.warning(f"Haiku summary failed: {summary_response.error}")
            except Exception as e:
                logger.warning(f"Failed to extract summary with Haiku: {e}")

        # Guaranteed fallback
        return extracted[0] if extracted else f"Task {_get_type_str(task.type)} completed"

    def _build_prompt(self, task: Task) -> str:
        """
        Build prompt from task.

        Supports optimization strategies:
        - Strategy 1: Minimal task prompts
        - Strategy 3: Context deduplication
        - Strategy 4: Compact JSON
        - Strategy 5: Result summarization
        - Shadow mode: Generate both for comparison

        Ported from scripts/async-agent-runner.sh lines 268-294.
        """
        shadow_mode = self._optimization_config.get("shadow_mode", False)
        use_optimizations = self._should_use_optimization(task)

        # Determine which prompt to use (or generate both for shadow mode)
        if shadow_mode:
            # Generate both prompts for comparison
            legacy_prompt = self._build_prompt_legacy(task)
            optimized_prompt = self._build_prompt_optimized(task)

            # Log comparison (DEBUG level to avoid sensitive data exposure)
            legacy_len = len(legacy_prompt)
            optimized_len = len(optimized_prompt)
            savings = legacy_len - optimized_len
            savings_pct = (savings / legacy_len * 100) if legacy_len > 0 else 0

            # Truncate task ID for security
            task_id_short = task.id[:8] + "..." if len(task.id) > 8 else task.id

            logger.debug(
                f"[SHADOW MODE] Task {task_id_short} prompt comparison: "
                f"legacy={legacy_len} chars, optimized={optimized_len} chars, "
                f"savings={savings} chars ({savings_pct:.1f}%)"
            )

            # Record metrics for analysis
            self._record_optimization_metrics(task, legacy_len, optimized_len)

            # Use legacy prompt (no behavioral change in shadow mode)
            return legacy_prompt
        elif use_optimizations:
            # Use optimized prompt
            return self._build_prompt_optimized(task)
        else:
            # Use legacy prompt (default)
            prompt = self._build_prompt_legacy(task)

        # Log prompt preview for debugging (sanitized)
        if logger.isEnabledFor(logging.DEBUG):
            prompt_preview = prompt[:500].replace(task.id, "TASK_ID")
            if hasattr(task, 'context') and task.context.get('jira_key'):
                prompt_preview = prompt_preview.replace(task.context['jira_key'], "JIRA-XXX")
            logger.debug(f"Built prompt preview (first 500 chars): {prompt_preview}...")

        return prompt

    def _record_optimization_metrics(
        self,
        task: Task,
        legacy_prompt_length: int,
        optimized_prompt_length: int
    ) -> None:
        """
        Record optimization metrics for post-deployment analysis.

        Writes metrics to .agent-communication/metrics/optimization.jsonl
        for later analysis of optimization effectiveness.
        """
        try:
            metrics = {
                "task_id": task.id,
                "task_type": _get_type_str(task.type),
                "agent_id": self.config.id,
                "legacy_prompt_chars": legacy_prompt_length,
                "optimized_prompt_chars": optimized_prompt_length,
                "savings_chars": legacy_prompt_length - optimized_prompt_length,
                "savings_percent": (
                    (legacy_prompt_length - optimized_prompt_length) / legacy_prompt_length * 100
                    if legacy_prompt_length > 0 else 0
                ),
                "canary_active": self._should_use_optimization(task),
                "optimizations_enabled": self._get_active_optimizations(),
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Write to metrics file for later analysis
            metrics_dir = self.workspace / ".agent-communication" / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_file = metrics_dir / "optimization.jsonl"

            with open(metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except PermissionError as e:
            logger.warning(f"Permission denied recording optimization metrics: {e}")
        except OSError as e:
            logger.warning(f"Failed to record optimization metrics (disk full?): {e}")
        except Exception as e:
            # Don't fail task if metrics recording fails
            logger.debug(f"Unexpected error recording optimization metrics: {e}")

    def _estimate_cost(self, response: LLMResponse) -> float:
        """
        Estimate cost based on model and token usage.

        Note: Pricing is approximate and subject to change. Check Anthropic's
        pricing page for current rates.

        Pricing as of 2025-01:
        - Haiku: $0.25/$1.25 per 1M tokens (input/output)
        - Sonnet: $3/$15 per 1M tokens
        - Opus: $15/$75 per 1M tokens
        """
        model_name_lower = response.model_used.lower()

        # Detect model family
        if "haiku" in model_name_lower:
            model_type = "haiku"
        elif "opus" in model_name_lower:
            model_type = "opus"
        elif "sonnet" in model_name_lower:
            model_type = "sonnet"
        else:
            # Unknown model, assume sonnet pricing as conservative estimate
            logger.warning(
                f"Unknown model '{response.model_used}', assuming Sonnet pricing for cost estimate"
            )
            model_type = "sonnet"

        prices = MODEL_PRICING.get(model_type, MODEL_PRICING["sonnet"])
        cost = (
            response.input_tokens / 1_000_000 * prices["input"] +
            response.output_tokens / 1_000_000 * prices["output"]
        )
        return cost

    def _build_prompt_legacy(self, task: Task) -> str:
        """Build prompt using legacy format (original implementation)."""
        task_json = task.model_dump_json(indent=2)

        # Extract integration context
        jira_key = task.context.get("jira_key")
        github_repo = task.context.get("github_repo")
        jira_project = task.context.get("jira_project")

        mcp_guidance = ""

        # Add JIRA guidance if MCP enabled and JIRA context exists
        if self._mcp_enabled and (jira_key or jira_project):
            jira_server = self.jira_config.server if self.jira_config else "jira.example.com"

            mcp_guidance += f"""
JIRA INTEGRATION (via MCP):
You have access to JIRA via MCP tools:
- Search issues: jira_search_issues(jql="project = {jira_project or 'PROJ'}")
- Get issue: jira_get_issue(issueKey="{jira_key or 'PROJ-123'}")
- Create ticket: jira_create_issue(project="{jira_project or 'PROJ'}", summary="...", description="...", issueType="Story")
- Create epic: jira_create_epic(project="{jira_project or 'PROJ'}", title="...", description="...")
- Create subtask: jira_create_subtask(parentKey="{jira_key or 'PROJ-123'}", summary="...", description="...")
- Update status: jira_transition_issue(issueKey="{jira_key or 'PROJ-123'}", transitionName="In Progress")
- Add comment: jira_add_comment(issueKey="{jira_key or 'PROJ-123'}", comment="...")

Current context:
- JIRA Server: {jira_server}
- Ticket: {jira_key or 'N/A'}
- Project: {jira_project or 'N/A'}

"""

        # Add GitHub guidance if MCP enabled and GitHub context exists
        if self._mcp_enabled and github_repo:
            owner, repo = github_repo.split("/")

            # Get formatting patterns from config
            branch_pattern = "{type}/{ticket_id}-{slug}"
            pr_title_pattern = "[{ticket_id}] {title}"
            if self.github_config:
                branch_pattern = self.github_config.branch_pattern
                pr_title_pattern = self.github_config.pr_title_pattern

            mcp_guidance += f"""
GITHUB INTEGRATION (via MCP):
Repository: {github_repo}
Branch naming: Use pattern "{branch_pattern}"
  Example: feature/{jira_key or 'PROJ-123'}-add-authentication
PR title: Use pattern "{pr_title_pattern}"
  Example: [{jira_key or 'PROJ-123'}] Add authentication feature

Available tools:
- github_create_pr(owner="{owner}", repo="{repo}",
                   title="[{jira_key or 'PROJ-123'}] Your Title",
                   body="...",
                   head="feature/{jira_key or 'PROJ-123'}-slug")
- github_add_pr_comment(owner="{owner}", repo="{repo}", prNumber=123, body="...")
- github_link_pr_to_jira(owner="{owner}", repo="{repo}", prNumber=123, jiraKey="{jira_key or 'PROJ-123'}")

NOTE: Git operations (clone, commit, push) are handled by the framework.
You focus on PR creation and management via GitHub API.

Workflow coordination:
1. Make your code changes
2. Your changes will be automatically committed after task completion
3. Create a PR using github_create_pr
4. Update JIRA using jira_transition_issue and jira_add_comment

"""

        # Add error handling guidance if MCP enabled
        if self._mcp_enabled:
            mcp_guidance += """
ERROR HANDLING:
If a tool call fails:
1. Read the error message carefully
2. If rate limited, wait and retry
3. If authentication failed, report failure
4. If invalid input, correct and retry
5. For partial failures (e.g., PR created but JIRA update failed):
   - Retry the failed operation
   - If still fails, leave completed operations and report the failure
   - Do NOT try to undo successful operations

"""

        return f"""You are {self.config.id} working on an asynchronous task.

TASK DETAILS:
{task_json}

{mcp_guidance}
YOUR RESPONSIBILITIES:
{self.config.prompt}

IMPORTANT:
- Complete the task described above
- Create any follow-up tasks by writing JSON files to other agents' queues
- Use unique task IDs (timestamp or UUID)
- Set depends_on array for tasks that depend on this one completing
- This task will be automatically marked as completed when you're done
"""

    def _build_prompt_optimized(self, task: Task) -> str:
        """
        Build prompt using optimized format.

        This method should only be called when optimizations are enabled.
        All enable checks happen in _build_prompt(), not here.

        Applies:
        - Strategy 1: Minimal task prompts (only essential fields)
        - Strategy 3: Context deduplication (no redundant info)
        - Strategy 4: Compact JSON (no whitespace)
        - Strategy 5: Result summarization (include dep summaries)
        """
        # Always use minimal fields in optimized prompts (Strategy 1)
        task_dict = self._get_minimal_task_dict(task)

        # Always use compact JSON in optimized prompts (Strategy 4)
        task_json = json.dumps(task_dict, separators=(',', ':'))

        # Extract integration context (only dynamic values, not boilerplate)
        jira_key = task.context.get("jira_key")
        github_repo = task.context.get("github_repo")
        jira_project = task.context.get("jira_project")

        # Build minimal MCP context (just dynamic values, no boilerplate)
        context_note = ""
        if self._mcp_enabled:
            if jira_key:
                context_note += f"JIRA Ticket: {jira_key}\n"
            if jira_project:
                context_note += f"JIRA Project: {jira_project}\n"
            if github_repo:
                context_note += f"GitHub Repository: {github_repo}\n"

        # Include dependency results (Strategy 5: Result Summarization)
        dep_context = ""
        enable_summarization = self._optimization_config.get("enable_result_summarization", False)
        if enable_summarization and task.depends_on:
            dep_context = "\nPREVIOUS WORK:\n"
            for dep_id in task.depends_on:
                dep_task = self.queue.get_completed(dep_id)
                if dep_task and dep_task.result_summary:
                    dep_context += f"- {dep_task.title}: {dep_task.result_summary}\n"

        # Build optimized prompt (shorter, focused on essentials)
        return f"""You are {self.config.id} working on an asynchronous task.

TASK:
{task_json}

{context_note}{dep_context}
{self.config.prompt}

IMPORTANT:
- Complete the task described above
- This task will be automatically marked as completed when you're done
"""

    def _get_working_directory(self, task: Task) -> Path:
        """Get working directory for task (target repo or framework workspace)."""
        github_repo = task.context.get("github_repo")

        if github_repo and self.multi_repo_manager:
            # Ensure repo is cloned/updated
            repo_path = self.multi_repo_manager.ensure_repo(github_repo)
            logger.info(f"Using repository: {github_repo} at {repo_path}")
            return repo_path
        else:
            # No repo context, use framework workspace
            return self.workspace

    def _update_phase(self, phase: TaskPhase):
        """Update current execution phase."""
        from datetime import datetime

        activity = self.activity_manager.get_activity(self.config.id)
        if activity:
            activity.current_phase = phase
            self.activity_manager.update_activity(activity)

            # Append phase event
            if activity.current_task:
                self.activity_manager.append_event(ActivityEvent(
                    type="phase",
                    agent=self.config.id,
                    task_id=activity.current_task.id,
                    title=activity.current_task.title,
                    timestamp=datetime.utcnow(),
                    phase=phase
                ))

    async def _handle_success(self, task: Task, llm_response) -> None:
        """
        Handle post-LLM workflow: git operations, PR creation, JIRA updates.

        DEPRECATED: When MCPs are enabled, agents handle this during execution.
        This method remains for backward compatibility.
        """
        # Skip if MCPs are enabled - agents do this during execution now
        if self._mcp_enabled:
            logger.debug("MCPs enabled - skipping post-LLM workflow")
            return

        jira_key = task.context.get("jira_key")
        if not jira_key or not self.github_client or not self.jira_client:
            logger.debug("Skipping post-LLM workflow (no JIRA key or clients not configured)")
            return

        try:
            # Get working directory (target repo or framework workspace)
            workspace = self._get_working_directory(task)
            logger.info(f"Running post-LLM workflow for {jira_key} in {workspace}")

            # Create branch
            slug = task.title.lower().replace(" ", "-")[:30]
            branch = self.github_client.format_branch_name(jira_key, slug)

            logger.info(f"Creating branch: {branch}")
            subprocess.run(
                ["git", "checkout", "-b", branch],
                cwd=workspace,
                check=True,
                capture_output=True,
            )

            # Stage and commit changes
            self._update_phase(TaskPhase.COMMITTING)
            logger.info("Committing changes")
            subprocess.run(
                ["git", "add", "-A"],
                cwd=workspace,
                check=True,
                capture_output=True,
            )

            commit_msg = f"[{jira_key}] {task.title}"
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=workspace,
                check=True,
                capture_output=True,
            )

            # Push to remote
            logger.info(f"Pushing branch to origin")
            subprocess.run(
                ["git", "push", "-u", "origin", branch],
                cwd=workspace,
                check=True,
                capture_output=True,
            )

            # Create PR
            self._update_phase(TaskPhase.CREATING_PR)
            logger.info("Creating pull request")
            pr_title = self.github_client.format_pr_title(jira_key, task.title)
            pr_body = f"Implements {jira_key}\n\n{task.description}"

            pr = self.github_client.create_pull_request(
                title=pr_title,
                body=pr_body,
                head_branch=branch,
            )

            logger.info(f"Created PR: {pr.html_url}")

            # Update JIRA
            self._update_phase(TaskPhase.UPDATING_JIRA)
            logger.info("Updating JIRA ticket")
            self.jira_client.transition_ticket(jira_key, "code_review")
            self.jira_client.add_comment(
                jira_key,
                f"Pull request created: {pr.html_url}"
            )

            logger.info(f"Post-LLM workflow complete for {jira_key}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Git operation failed: {e.stderr.decode() if e.stderr else str(e)}")
        except Exception as e:
            logger.exception(f"Error in post-LLM workflow: {e}")

    def _write_heartbeat(self) -> None:
        """Write current Unix timestamp to heartbeat file."""
        self.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_file.write_text(str(int(time.time())))
