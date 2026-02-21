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
    from .feedback_bus import FeedbackBus

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
        feedback_bus: Optional["FeedbackBus"] = None,
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
        self.feedback_bus = feedback_bus

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

        if task.status == TaskStatus.COMPLETED:
            self.logger.warning(
                f"Task {task.id} is already completed — skipping error recovery. "
                f"Last error: {task.last_error}"
            )
            return

        if task.retry_count >= self.retry_handler.max_retries:
            # Max retries exceeded - mark as failed
            self.logger.error(
                f"Task {task.id} has failed {task.retry_count} times "
                f"(max: {self.retry_handler.max_retries})"
            )
            error_type = self._categorize_error(task.last_error or "") or "unknown"
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

                # Only record the antipattern when no escalation path exists. If an
                # escalation is created it may resolve the issue — we shouldn't label
                # approaches "unresolved dead ends" that the escalation might fix.
                repo_slug = self._get_repo_slug(task)
                if repo_slug:
                    try:
                        self.store_failure_antipattern(task, repo_slug, error_type)
                    except Exception as e:
                        self.logger.warning(f"Failed to store failure antipattern: {e}")
        else:
            # Dynamic replanning: generate revised approach on retry 2+
            if self._replan_enabled and task.retry_count >= self._replan_min_retry:
                await self.request_replan(task)

            # Clear stale upstream context from failed attempt — prevents the
            # retried agent from seeing its own previous output as upstream input
            task.context.pop("upstream_summary", None)
            task.context.pop("upstream_context_file", None)
            task.context.pop("upstream_source_agent", None)
            task.context.pop("upstream_source_step", None)

            # Increment chain state attempt counter so retries have full history
            self._increment_chain_state_attempt(task)

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

    def _try_diff_strategies(self, working_dir: Path) -> tuple[str, str]:
        """Try multiple git diff strategies, return first non-empty (stat, diff).

        Priority order captures the most common agent scenarios:
        1. HEAD — uncommitted working tree changes (agent wrote but didn't commit)
        2. --staged — staged but uncommitted (edge: fresh repo with no HEAD)
        3. HEAD~1 — agent already committed its work
        """
        strategies = [
            ["HEAD"],          # uncommitted changes (staged + unstaged vs HEAD)
            ["--staged"],      # fallback: fresh repo with no HEAD yet
            ["HEAD~1"],        # last commit (agent already committed)
        ]
        for ref in strategies:
            try:
                stat_result = run_git_command(
                    ["diff", "--stat"] + ref,
                    cwd=working_dir, check=False, timeout=10,
                )
                stat_text = (stat_result.stdout or "").strip()
                if not stat_text:
                    continue

                diff_result = run_git_command(
                    ["diff"] + ref,
                    cwd=working_dir, check=False, timeout=10,
                )
                diff_text = (diff_result.stdout or "").strip()
                return stat_text[:500], diff_text[:2500]
            except Exception as e:
                self.logger.debug(f"Diff strategy {ref} failed: {e}")
                continue
        return "", ""

    def _build_checklist_report(self, task: Task, git_evidence: str) -> str:
        """Cross-reference requirements checklist against git diff evidence.

        Gives the self-eval LLM structured signal about which deliverables
        appear in the code changes vs which are missing.
        """
        checklist = task.context.get("requirements_checklist")
        if not checklist:
            return ""

        # Extract modified file names from git evidence for matching
        diff_lower = git_evidence.lower() if git_evidence else ""

        lines = ["## Requirements Checklist Status"]
        matched = 0

        for item in checklist:
            item_id = item.get("id", "?")
            desc = item.get("description", "")
            files = item.get("files", [])

            # Check if any associated file appears in the diff,
            # or if keywords from the description appear
            found = False
            match_hint = ""

            for f in files:
                fname = Path(f).name.lower()
                if fname in diff_lower:
                    found = True
                    match_hint = f"{f} modified"
                    break

            if not found:
                # Keyword heuristic: check if distinctive words from description appear in diff
                words = [w.lower() for w in desc.split() if len(w) > 6]
                matching_words = [w for w in words[:5] if w in diff_lower]
                if len(matching_words) >= 2:
                    found = True
                    match_hint = f"keywords matched: {', '.join(matching_words)}"

            if found:
                matched += 1
                lines.append(f"  ✅ {item_id}. {desc} ({match_hint})")
            else:
                lines.append(f"  ❌ {item_id}. {desc} (no matching files in diff)")

        total = len(checklist)
        lines.insert(1, f"{matched}/{total} items appear in code changes:")

        if matched < total:
            lines.append(
                f"\n⚠️  {total - matched} deliverable(s) appear to be missing. "
                "FAIL unless the agent's output explains why they were intentionally omitted."
            )

        lines.append("")
        return "\n".join(lines)

    def has_deliverables(self, task: Task, working_dir: Path) -> bool:
        """Check whether the agent produced any git-visible code changes.

        Reuses _try_diff_strategies() — returns True if any strategy finds
        a non-empty diff, False when all strategies come back empty (likely
        context window exhaustion where Claude exits 0 without writing code).
        """
        stat_text, _ = self._try_diff_strategies(working_dir)
        if stat_text:
            return True

        self.logger.warning(
            f"Deliverable gate: no git changes detected for task {task.id} "
            f"in {working_dir}"
        )
        self.session_logger.log(
            "deliverable_gate",
            task_id=task.id,
            working_dir=str(working_dir),
            result="no_changes",
        )
        return False

    def gather_git_evidence(self, working_dir: Path) -> str:
        """Collect git diff evidence for self-evaluation.

        Returns a formatted string with diff stat and full diff (truncated),
        or empty string on any error so eval falls back to text-only.
        """
        try:
            stat_text, diff_text = self._try_diff_strategies(working_dir)

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
        git_evidence = self.gather_git_evidence(working_dir) if working_dir else ""
        if test_passed is True:
            test_section = "## Test Results\nPASSED"
        elif test_passed is False:
            test_section = "## Test Results\nFAILED"
        else:
            test_section = "## Test Results\nNot run"

        # No objective evidence → evaluating prose alone causes false negatives
        if not git_evidence and test_passed is None:
            self.logger.warning(
                f"Self-eval skipped for task {task.id}: no git diff or test results to evaluate"
            )
            self.session_logger.log(
                "self_eval",
                verdict="AUTO_PASS",
                reason="no_objective_evidence",
                eval_attempt=eval_retries + 1,
            )
            return True

        # Weight instructions based on available evidence
        diff_instruction = ""
        if git_evidence:
            diff_instruction = (
                "- PRIORITIZE the git diff over the conversation summary when evaluating. "
                "The diff shows actual code changes.\n"
            )

        # Build checklist completion report if requirements checklist exists
        checklist_report = self._build_checklist_report(task, git_evidence)

        eval_prompt = f"""Review this agent's output against the acceptance criteria.
Reply with PASS if all criteria are met, or FAIL followed by specific gaps.

## Acceptance Criteria
{criteria_text}

{test_section}

{git_evidence}

{checklist_report}## Agent Output (conversation summary — NOT code)
{response_preview}

## Rules
- If tests PASSED and git diff shows changes consistent with acceptance criteria, verdict is PASS.
- When in doubt and tests passed, default to PASS.
- Do NOT fail solely because the conversation summary is vague or informal.
{diff_instruction}Verdict:"""

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
                f"{verdict[:500]}"
            )

            task.context["_self_eval_count"] = eval_retries + 1
            task.context["_self_eval_critique"] = verdict
            task.notes.append(f"Self-eval failed (attempt {eval_retries + 1}): {verdict[:500]}")

            # Store missed criteria patterns via feedback bus
            if self.feedback_bus and self.feedback_bus.enabled:
                repo_slug = self._get_repo_slug(task)
                if repo_slug:
                    try:
                        stored = self.feedback_bus.store_self_eval_failure(
                            repo_slug=repo_slug,
                            agent_type=self.config.base_id,
                            task_id=task.id,
                            critique=verdict,
                            acceptance_criteria=criteria,
                        )
                        if stored:
                            self.session_logger.log(
                                "feedback_bus_self_eval",
                                task_id=task.id,
                                memories_stored=stored,
                            )
                    except Exception as e:
                        self.logger.debug(f"Feedback bus self-eval store failed (non-fatal): {e}")

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
        - past_failures: tag-filtered by error type so the LLM sees patterns
          relevant to the current failure mode (falls back to unfiltered)
        - conventions: coding standards, patterns to follow
        - test_commands: how to run/fix tests
        - repo_structure: where key files live
        - shared namespace: cross-agent conventions and decisions

        Returns empty string if memory disabled or no relevant memories found.
        """
        if not self.memory_store or not self.memory_store.enabled:
            return ""

        repo_slug = self._get_repo_slug(task)
        if not repo_slug:
            return ""

        error_type = self._categorize_error(task.last_error or "") if task.last_error else None
        memories = []
        seen_content: set[str] = set()

        def _add(mems):
            for m in mems:
                if m.content not in seen_content:
                    memories.append(m)
                    seen_content.add(m.content)

        # past_failures: tag-filtered first so error-specific patterns rank higher;
        # only fall back to unfiltered when the filter was actually applied
        tags = [error_type] if error_type else None
        past_failures = self.memory_store.recall(
            repo_slug=repo_slug,
            agent_type=self.config.base_id,
            category="past_failures",
            tags=tags,
            limit=5,
        )
        if tags and not past_failures:
            past_failures = self.memory_store.recall(
                repo_slug=repo_slug,
                agent_type=self.config.base_id,
                category="past_failures",
                limit=5,
            )
        _add(past_failures)

        # Shared-namespace memories (architectural decisions, cross-agent conventions)
        # collected before the broad agent-scoped categories so they aren't crowded out.
        # Tag-filtered for the same reason as agent-scoped past_failures — but no
        # unfiltered fallback, since shared entries are supplementary and an empty result
        # simply means no cross-agent pattern matches this error type.
        _add(self.memory_store.recall(
            repo_slug=repo_slug,
            agent_type="shared",
            category="past_failures",
            tags=tags,
            limit=3,
        ))
        _add(self.memory_store.recall(
            repo_slug=repo_slug,
            agent_type="shared",
            category="conventions",
            limit=3,
        ))

        for category in ("conventions", "test_commands", "repo_structure"):
            _add(self.memory_store.recall(
                repo_slug=repo_slug,
                agent_type=self.config.base_id,
                category=category,
                limit=5,
            ))

        if not memories:
            return ""

        # Format as a context section
        lines = ["\n## Relevant Context from Previous Work"]
        lines.append("You've worked on this repo before. Here's what you know:\n")

        for mem in memories[:10]:  # Cap at 10 total memories
            lines.append(f"- [{mem.category}] {mem.content}")

        lines.append("")  # trailing newline
        return "\n".join(lines)

    def _increment_chain_state_attempt(self, task: Task) -> None:
        """Increment the chain state attempt counter on retry.

        Non-fatal — chain state is supplementary to the retry mechanism.
        """
        try:
            from .chain_state import load_chain_state, save_chain_state

            root_task_id = task.root_id
            state = load_chain_state(self.workspace, root_task_id)
            if state is None:
                return

            state.attempt += 1

            # Record the failed step's error
            if state.steps:
                last_step = state.steps[-1]
                if not last_step.error and task.last_error:
                    last_step.error = task.last_error[:500]

            save_chain_state(self.workspace, state)
            self.logger.debug(
                f"Chain state: incremented attempt to {state.attempt} "
                f"for root task {root_task_id}"
            )
        except Exception as e:
            self.logger.debug(f"Failed to increment chain state attempt: {e}")

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
            source_task_id=task.id,
            tags=[error_type],
        )

    def store_failure_antipattern(self, task: Task, repo_slug: str, error_type: str) -> None:
        """Persist an unresolved failure pattern as a past_failures memory.

        Called only when no escalation path exists — if an escalation is created,
        the approaches tried here might still succeed at a higher level. Only true
        dead ends (no escalation, no recovery) are worth recording as antipatterns.
        """
        if not self.memory_store or not self.memory_store.enabled:
            return

        # Nothing to learn if the task never reached replanning
        if not task.replan_history:
            return

        approaches = [
            entry["approach_tried"]
            for entry in task.replan_history
            if entry.get("approach_tried")
        ]

        # Union files across all retry attempts — earlier attempts may have touched
        # different files than the last one
        all_files: set[str] = set()
        for entry in task.replan_history:
            all_files.update(entry.get("files_involved", []))
        top_files = sorted(all_files)[:3]
        overflow = len(all_files) - len(top_files)
        files_str = ", ".join(top_files) + (f" (+{overflow} more)" if overflow else "") if all_files else "unknown files"

        approaches_str = "; ".join(approaches) if approaches else "unknown approaches"
        content = (
            f"{error_type} in {files_str}: "
            f"tried [{approaches_str}] → unresolved after {task.retry_count} retries"
        )

        self.memory_store.remember(
            repo_slug=repo_slug,
            agent_type=self.config.base_id,
            category="past_failures",
            content=content,
            source_task_id=task.id,
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
