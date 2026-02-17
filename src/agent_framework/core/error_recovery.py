"""Error recovery and replanning manager."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .config import AgentConfig
    from ..llm.base import LLMBackend
    from ..queue.file_queue import FileQueue
    from ..safeguards.retry_handler import RetryHandler
    from ..safeguards.escalation import EscalationHandler
    from ..memory.memory_store import MemoryStore
    from ..utils.rich_logging import ContextLogger
    from .session_logger import SessionLogger

from .task import Task, TaskStatus
from ..llm.base import LLMRequest
from ..utils.subprocess_utils import run_git_command


class ErrorRecoveryManager:
    """Manages error handling, retries, escalation, and dynamic replanning."""

    def __init__(
        self,
        config: "AgentConfig",
        queue: "FileQueue",
        llm: "LLMBackend",
        logger: "ContextLogger",
        session_logger: "SessionLogger",
        retry_handler: "RetryHandler",
        escalation_handler: "EscalationHandler",
        workspace: Path,
        jira_client=None,
        memory_store: Optional["MemoryStore"] = None,
        replan_config: Optional[dict] = None,
        self_eval_config: Optional[dict] = None,
    ):
        self.config = config
        self.queue = queue
        self.llm = llm
        self.logger = logger
        self.session_logger = session_logger
        self.retry_handler = retry_handler
        self.escalation_handler = escalation_handler
        self.workspace = workspace
        self.jira_client = jira_client
        self.memory_store = memory_store

        # Self-evaluation configuration
        eval_cfg = self_eval_config or {}
        self._self_eval_enabled = eval_cfg.get("enabled", False)
        self._self_eval_max_retries = eval_cfg.get("max_retries", 2)
        self._self_eval_model = eval_cfg.get("model", "haiku")

        # Dynamic replanning configuration
        replan_cfg = replan_config or {}
        self._replan_enabled = replan_cfg.get("enabled", True)
        self._replan_min_retry = replan_cfg.get("min_retry_for_replan", 2)
        self._replan_model = replan_cfg.get("model", "haiku")

    async def handle_failure(self, task: Task) -> None:
        """
        Handle task failure with retry/escalation logic.

        Ported from scripts/async-agent-runner.sh lines 374-394.
        """
        # Re-read from disk to detect external status changes (e.g. `agent cancel`)
        refreshed = self.queue.find_task(task.id)
        if refreshed and refreshed.status == TaskStatus.CANCELLED:
            self.logger.info(f"Task {task.id} was cancelled, skipping retry")
            # Preserve CANCELLED status — mark_completed would overwrite it
            self.queue.move_to_completed(refreshed)
            return

        if task.retry_count >= self.retry_handler.max_retries:
            # Max retries exceeded - mark as failed
            self.logger.error(
                f"Task {task.id} has failed {task.retry_count} times "
                f"(max: {self.retry_handler.max_retries})"
            )
            error_type = self._categorize_error(task.last_error or "")
            task.mark_failed(self.config.id, error_message=task.last_error, error_type=error_type)
            self.queue.mark_failed(task)

            # Notify JIRA about permanent failure (no status change, just a comment)
            jira_key = task.context.get("jira_key")
            if jira_key and self.jira_client:
                try:
                    self.jira_client.add_comment(
                        jira_key,
                        f"Agent {self.config.id} failed after {task.retry_count} retries: {task.last_error}",
                    )
                except Exception:
                    pass

            # CRITICAL: Prevent infinite loop - escalations should NOT create more escalations
            if self.retry_handler.can_create_escalation(task):
                escalation = self.escalation_handler.create_escalation(
                    task, self.config.id
                )
                self.queue.push(escalation, escalation.assigned_to)
                self.logger.warning(
                    f"Created escalation task {escalation.id} for failed task {task.id}"
                )
            else:
                # Escalation failed - log to escalations directory for human review
                self.logger.error(
                    f"Escalation task {task.id} failed after {task.retry_count} retries - "
                    "NOT creating another escalation (would cause infinite loop). "
                    "Logging to escalations directory for human intervention."
                )
                self._log_failed_escalation(task)
        else:
            # Dynamic replanning: generate revised approach on retry 2+
            if self._replan_enabled and task.retry_count >= self._replan_min_retry:
                await self.request_replan(task)

            # Reset task to pending so it can be retried
            self.logger.warning(
                f"Resetting task {task.id} to pending status "
                f"(retry {task.retry_count + 1}/{self.retry_handler.max_retries})"
            )
            task.reset_to_pending()
            self.queue.update(task)

    def _log_failed_escalation(self, task: Task) -> None:
        """
        Log a failed escalation to the escalations directory for human review.

        When an escalation task itself fails, we cannot create another escalation
        (infinite loop). Instead, write it to a dedicated directory where humans
        can review and resolve it.
        """
        escalations_dir = self.workspace / ".agent-communication" / "escalations"
        escalations_dir.mkdir(parents=True, exist_ok=True)

        escalation_file = escalations_dir / f"{task.id}.json"

        # Add metadata for human review
        task_dict = task.model_dump()
        task_dict["logged_at"] = datetime.now(timezone.utc).isoformat()
        task_dict["logged_by"] = self.config.id
        task_dict["requires_human_intervention"] = True
        task_dict["escalation_failed"] = True

        try:
            escalation_file.write_text(json.dumps(task_dict, indent=2, default=str))
            self.logger.info(
                f"Logged failed escalation to {escalation_file}. "
                f"Run 'bash scripts/review-escalations.sh' to review."
            )
        except Exception as e:
            self.logger.error(f"Failed to log escalation to file: {e}")
            # Last resort: at least log the full task details to the log file
            self.logger.error(f"Failed escalation details: {json.dumps(task_dict, indent=2, default=str)}")

    def _categorize_error(self, error_message: str) -> Optional[str]:
        """Categorize error message for better diagnostics.

        Delegates to EscalationHandler.categorize_error for consistent
        pattern matching across the codebase.
        """
        return self.escalation_handler.categorize_error(error_message)

    def _gather_git_evidence(self, working_dir: Path) -> str:
        """Collect git diff evidence for self-evaluation.

        Returns a formatted string with diff stat and full diff (truncated),
        or empty string on any error so eval falls back to text-only.
        """
        try:
            stat_result = run_git_command(
                ["diff", "--stat", "HEAD~1"],
                cwd=working_dir, check=False, timeout=10,
            )
            stat_text = (stat_result.stdout or "")[:500]

            diff_result = run_git_command(
                ["diff", "HEAD~1"],
                cwd=working_dir, check=False, timeout=10,
            )
            diff_text = (diff_result.stdout or "")[:2500]

            if not stat_text and not diff_text:
                return ""

            parts = ["## Git Diff (actual code changes)"]
            if stat_text:
                parts.append(f"### Summary\n```\n{stat_text}\n```")
            if diff_text:
                parts.append(f"### Diff\n```\n{diff_text}\n```")
            return "\n".join(parts)
        except Exception:
            return ""

    async def self_evaluate(
        self, task: Task, response, *, test_passed: Optional[bool] = None, working_dir: Optional[Path] = None
    ) -> bool:
        """Review agent's own output against acceptance criteria.

        Uses a cheap model to check for obvious gaps. Feeds git diff and
        test results as objective evidence so the evaluator can make an
        evidence-based verdict instead of relying solely on agent prose.

        Returns True if evaluation passed (or disabled/skipped).
        Returns False if task was reset for retry.
        """
        eval_retries = task.context.get("_self_eval_count", 0)
        if eval_retries >= self._self_eval_max_retries:
            self.logger.debug(
                f"Self-eval retry limit reached ({eval_retries}), proceeding"
            )
            return True

        # Build evaluation prompt from acceptance criteria
        criteria = task.acceptance_criteria
        if not criteria:
            return True

        criteria_text = "\n".join(f"- {c}" for c in criteria)
        response_preview = response.content[:4000] if response.content else ""

        # Gather objective evidence
        git_evidence = self._gather_git_evidence(working_dir) if working_dir else ""
        if test_passed is True:
            test_section = "## Test Results\nPASSED"
        elif test_passed is False:
            test_section = "## Test Results\nFAILED"
        else:
            test_section = "## Test Results\nNot run"

        eval_prompt = f"""Review this agent's output against the acceptance criteria.
Reply with PASS if all criteria are met, or FAIL followed by specific gaps.

## Acceptance Criteria
{criteria_text}

{test_section}

{git_evidence}

## Agent Output (summary)
{response_preview}

## Rules
- If tests PASSED and git diff shows changes consistent with acceptance criteria, verdict is PASS.
- When in doubt and tests passed, default to PASS.

Verdict:"""

        try:
            eval_response = await self.llm.complete(LLMRequest(
                prompt=eval_prompt,
                model=self._self_eval_model,
            ))

            if not eval_response.success or not eval_response.content:
                self.logger.warning("Self-eval LLM call failed, proceeding without eval")
                return True

            verdict = eval_response.content.strip()
            passed = verdict.upper().startswith("PASS")

            self.session_logger.log(
                "self_eval",
                verdict="PASS" if passed else "FAIL",
                model=self._self_eval_model,
                criteria_count=len(criteria),
                eval_attempt=eval_retries + 1,
                has_git_evidence=bool(git_evidence),
                has_test_evidence=test_passed is not None,
            )

            if passed:
                self.logger.info(f"Self-evaluation PASSED for task {task.id}")
                return True

            # Failed self-eval — reset for retry with critique
            self.logger.warning(
                f"Self-evaluation FAILED for task {task.id} "
                f"(attempt {eval_retries + 1}/{self._self_eval_max_retries}): "
                f"{verdict[:200]}"
            )

            task.context["_self_eval_count"] = eval_retries + 1
            task.context["_self_eval_critique"] = verdict[:1000]
            task.notes.append(f"Self-eval failed (attempt {eval_retries + 1}): {verdict[:200]}")

            # Reset without consuming queue retry
            task.status = TaskStatus.PENDING
            task.started_at = None
            task.started_by = None
            self.queue.update(task)

            return False

        except Exception as e:
            self.logger.warning(f"Self-evaluation error (non-fatal): {e}")
            return True

    async def request_replan(self, task: Task) -> None:
        """Generate a revised approach based on what failed.

        Called on retry 2+ to avoid repeating the same failing approach.
        Stores the revised plan in task.replan_history and task.context
        so the next prompt attempt sees what was tried and the new approach.

        Injects relevant memories (conventions, test commands, repo structure)
        to give context about what's been learned from previous work on this repo.
        """
        error = task.last_error or "Unknown error"
        error_type = self._categorize_error(error)
        previous_attempts = task.replan_history or []

        # Extract original plan context if available
        original_plan = ""
        if task.plan:
            # Format PlanDocument if available
            original_plan = f"\n## Original Plan\nApproach: {', '.join(task.plan.approach[:3])}\n"
        elif task.context.get("plan"):
            # Fallback to context-based plan
            plan_ctx = task.context["plan"]
            if isinstance(plan_ctx, dict) and "approach" in plan_ctx:
                original_plan = f"\n## Original Plan\nApproach: {', '.join(plan_ctx['approach'][:3])}\n"

        attempts_text = ""
        if previous_attempts:
            for attempt in previous_attempts:
                approach = attempt.get('approach_tried', 'not recorded')
                attempts_text += (
                    f"\n- Attempt {attempt.get('attempt', '?')}: "
                    f"{attempt.get('error', 'no error recorded')} "
                    f"(tried: {approach})"
                )

        # Build memory context for replanning — prioritize categories that help with recovery
        memory_context = self._build_replan_memory_context(task)

        replan_prompt = f"""A task has failed {task.retry_count} times. Generate a REVISED approach.

## Task
{task.title}: {task.description[:1000]}
{original_plan}
## Latest Error
Type: {error_type or 'unknown'}
Details: {error[:500]}

## Previous Attempts{attempts_text if attempts_text else ' (first replan)'}
{memory_context}
## Instructions
Provide a revised approach in 3-5 bullet points. Focus on what to do DIFFERENTLY.
Do NOT repeat the same approach. Consider: different implementation strategy,
breaking the task into smaller steps, or working around the root cause."""

        try:
            replan_response = await self.llm.complete(LLMRequest(
                prompt=replan_prompt,
                model=self._replan_model,
            ))

            if replan_response.success and replan_response.content:
                revised_plan = replan_response.content.strip()[:2000]

                # Extract approach summary from the revised plan (first bullet point)
                approach_tried = "previous approach"
                if task.replan_history:
                    # Use the previous attempt's revised plan as the approach tried
                    prev_plan = task.replan_history[-1].get("revised_plan", "")
                    if prev_plan:
                        # Extract first line/bullet as summary
                        first_line = prev_plan.split('\n')[0].strip('- •*').strip()
                        approach_tried = first_line[:100] if first_line else "previous approach"

                # Determine files involved from context or error
                files_involved = []
                if "files_to_modify" in task.context:
                    files_involved = task.context["files_to_modify"][:5]
                elif task.plan and task.plan.files_to_modify:
                    files_involved = task.plan.files_to_modify[:5]

                # Store in replan history with enriched context
                history_entry = {
                    "attempt": task.retry_count,
                    "error": error[:500],
                    "error_type": error_type,
                    "approach_tried": approach_tried,
                    "files_involved": files_involved,
                    "revised_plan": revised_plan,
                }
                task.replan_history.append(history_entry)

                # Store in context for prompt injection
                task.context["_revised_plan"] = revised_plan
                task.context["_replan_attempt"] = task.retry_count

                self.session_logger.log(
                    "replan",
                    retry=task.retry_count,
                    previous_error=error[:500],
                    revised_plan=revised_plan,
                    model=self._replan_model,
                )

                self.logger.info(
                    f"Generated revised plan for task {task.id} "
                    f"(retry {task.retry_count}): {revised_plan[:100]}..."
                )
            else:
                self.logger.warning(
                    f"Replan LLM call failed for task {task.id}: "
                    f"{replan_response.error}"
                )

        except Exception as e:
            self.logger.warning(f"Replanning error (non-fatal): {e}")

    def _build_replan_memory_context(self, task: Task) -> str:
        """Build memory context specifically for replanning.

        Prioritizes categories that help with task recovery:
        - conventions: coding standards, patterns to follow
        - test_commands: how to run/fix tests
        - repo_structure: where key files live

        Returns empty string if memory disabled or no relevant memories found.
        """
        if not self.memory_store or not self.memory_store.enabled:
            return ""

        repo_slug = self._get_repo_slug(task)
        if not repo_slug:
            return ""

        # Prioritize categories useful for recovery — past_failures first so
        # the LLM sees what recovery strategies worked before
        priority_categories = ["past_failures", "conventions", "test_commands", "repo_structure"]
        memories = []

        for category in priority_categories:
            category_memories = self.memory_store.recall(
                repo_slug=repo_slug,
                agent_type=self.config.base_id,
                category=category,
                limit=5,
            )
            memories.extend(category_memories)

        if not memories:
            return ""

        # Format as a context section
        lines = ["\n## Relevant Context from Previous Work"]
        lines.append("You've worked on this repo before. Here's what you know:\n")

        for mem in memories[:10]:  # Cap at 10 total memories
            lines.append(f"- [{mem.category}] {mem.content}")

        lines.append("")  # trailing newline
        return "\n".join(lines)

    def _get_repo_slug(self, task: Task) -> Optional[str]:
        """Extract github_repo from task context."""
        return task.context.get("github_repo")

    def store_replan_outcome(self, task: Task, repo_slug: str) -> None:
        """Persist successful recovery pattern as a past_failures memory.

        Called after a task that went through replanning completes successfully.
        Stores the error→resolution pair so future replans can reference it.
        """
        if not self.memory_store or not self.memory_store.enabled:
            return

        if not task.replan_history:
            return

        last_entry = task.replan_history[-1]
        error_type = last_entry.get("error_type", "unknown")
        files = last_entry.get("files_involved", [])
        files_str = ", ".join(files[:3]) if files else "unknown files"
        revised_plan = last_entry.get("revised_plan", "")

        # First line of the revised plan as a concise summary
        plan_summary = revised_plan.split("\n")[0].strip("- *").strip()
        if not plan_summary:
            plan_summary = revised_plan[:100]

        content = f"{error_type} in {files_str}: {plan_summary} → resolved"

        self.memory_store.remember(
            repo_slug=repo_slug,
            agent_type=self.config.base_id,
            category="past_failures",
            content=content,
            task_id=task.id,
            tags=[error_type],
        )

    def inject_replan_context(self, prompt: str, task: Task) -> str:
        """Append revised plan and attempt history to prompt if available."""
        revised_plan = task.context.get("_revised_plan")
        if not revised_plan:
            return prompt

        self_eval_critique = task.context.get("_self_eval_critique", "")

        replan_section = f"""

## REVISED APPROACH (retry {task.retry_count})

Previous attempts failed. Use this revised approach:

{revised_plan}
"""
        if self_eval_critique:
            replan_section += f"""
## Self-Evaluation Feedback
{self_eval_critique}
"""

        if task.replan_history:
            replan_section += "\n## Previous Attempt History\n"
            for entry in task.replan_history[:-1]:  # Skip current, already shown above
                error_type = entry.get('error_type', 'unknown')
                approach = entry.get('approach_tried', 'not recorded')
                files = entry.get('files_involved', [])
                files_str = f" (files: {', '.join(files[:3])})" if files else ""

                replan_section += (
                    f"- Attempt {entry.get('attempt', '?')}: "
                    f"Tried '{approach}' → {error_type} error{files_str}\n"
                    f"  Error: {entry.get('error', 'unknown')[:100]}\n"
                )

        return prompt + replan_section
