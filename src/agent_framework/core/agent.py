"""Agent polling loop implementation (ported from Bash system)."""

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .task import Task, TaskStatus, TaskType
from ..llm.base import LLMBackend, LLMRequest
from ..queue.file_queue import FileQueue
from ..safeguards.retry_handler import RetryHandler
from ..safeguards.escalation import EscalationHandler


logger = logging.getLogger(__name__)


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
    ):
        self.config = config
        self.llm = llm
        self.queue = queue
        self.workspace = Path(workspace)
        self.jira_client = jira_client
        self.github_client = github_client
        self._running = False
        self._current_task_id: Optional[str] = None

        # Initialize safeguards
        self.retry_handler = RetryHandler(max_retries=config.max_retries)
        self.escalation_handler = EscalationHandler()

        # Heartbeat file
        self.heartbeat_file = self.workspace / ".agent-communication" / "heartbeats" / config.id

    async def run(self) -> None:
        """
        Main polling loop.

        Ported from scripts/async-agent-runner.sh lines 254-407.
        """
        self._running = True
        logger.info(f"Starting {self.config.id} runner")

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
        logger.info(f"Found task: {task.id} - {task.title}")

        # Try to acquire lock
        lock = self.queue.acquire_lock(task.id, self.config.id)
        if not lock:
            logger.debug(f"Could not acquire lock for {task.id}, will retry later")
            return

        self._current_task_id = task.id

        try:
            # Mark in progress
            task.mark_in_progress(self.config.id)
            self.queue.update(task)

            # Build prompt
            prompt = self._build_prompt(task)

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
                )
            )

            if response.success:
                # Handle post-LLM workflow (git/PR/JIRA)
                await self._handle_success(task, response)

                # Task completed successfully
                task.mark_completed(self.config.id)
                self.queue.mark_completed(task)
                logger.info(f"Completed task: {task.id}")
            else:
                # Task failed
                logger.error(
                    f"LLM failed for task {task.id}: {response.error}"
                )
                await self._handle_failure(task)

        except Exception as e:
            logger.exception(f"Error processing task {task.id}: {e}")
            await self._handle_failure(task)

        finally:
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

    def _build_prompt(self, task: Task) -> str:
        """
        Build prompt from task.

        Ported from scripts/async-agent-runner.sh lines 268-294.
        """
        task_json = task.model_dump_json(indent=2)

        return f"""You are {self.config.id} working on an asynchronous task.

TASK DETAILS:
{task_json}

YOUR RESPONSIBILITIES:
{self.config.prompt}

IMPORTANT:
- Complete the task described above
- Create any follow-up tasks by writing JSON files to other agents' queues
- Use unique task IDs (timestamp or UUID)
- Set depends_on array for tasks that depend on this one completing
- This task will be automatically marked as completed when you're done
"""

    async def _handle_success(self, task: Task, llm_response) -> None:
        """
        Handle post-LLM workflow: git operations, PR creation, JIRA updates.

        Only runs if task has JIRA context and clients are configured.
        """
        jira_key = task.context.get("jira_key")
        if not jira_key or not self.github_client or not self.jira_client:
            logger.debug("Skipping post-LLM workflow (no JIRA key or clients not configured)")
            return

        try:
            logger.info(f"Running post-LLM workflow for {jira_key}")

            # Create branch
            slug = task.title.lower().replace(" ", "-")[:30]
            branch = self.github_client.format_branch_name(jira_key, slug)

            logger.info(f"Creating branch: {branch}")
            subprocess.run(
                ["git", "checkout", "-b", branch],
                cwd=self.workspace,
                check=True,
                capture_output=True,
            )

            # Stage and commit changes
            logger.info("Committing changes")
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.workspace,
                check=True,
                capture_output=True,
            )

            commit_msg = f"[{jira_key}] {task.title}"
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=self.workspace,
                check=True,
                capture_output=True,
            )

            # Push to remote
            logger.info(f"Pushing branch to origin")
            subprocess.run(
                ["git", "push", "-u", "origin", branch],
                cwd=self.workspace,
                check=True,
                capture_output=True,
            )

            # Create PR
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
