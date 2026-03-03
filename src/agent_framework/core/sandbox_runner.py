"""Sandbox test execution for Docker-based testing."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..utils.rich_logging import ContextLogger
    from ..queue.file_queue import FileQueue
    from .activity import ActivityManager
    from .git_operations import GitOperationsManager

from .task import Task, TaskStatus, TaskType
from .activity import ActivityEvent


# Optional sandbox imports (only used if Docker is available)
try:
    from ..sandbox import DockerExecutor, GoTestRunner, TestResult
    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    DockerExecutor = None
    GoTestRunner = None
    TestResult = None


class SandboxRunner:
    """Runs tests in Docker sandbox and handles test failures."""

    def __init__(
        self,
        config,
        logger: "ContextLogger",
        queue: "FileQueue",
        activity_manager: "ActivityManager",
        git_ops: "GitOperationsManager",
        test_runner=None,
    ):
        self.config = config
        self.logger = logger
        self.queue = queue
        self.activity_manager = activity_manager
        self._git_ops = git_ops
        self._test_runner = test_runner

    @staticmethod
    def create_test_runner(config, logger) -> Optional[Any]:
        """Initialize sandbox for isolated test execution. Returns test_runner or None."""
        if not config.enable_sandbox or not SANDBOX_AVAILABLE:
            return None
        try:
            executor = DockerExecutor(image=config.sandbox_image)
            if executor.health_check():
                logger.info(f"Agent {config.id} sandbox enabled with image {config.sandbox_image}")
                return GoTestRunner(executor=executor)
            else:
                logger.warning(f"Docker not available, sandbox disabled for {config.id}")
        except Exception as e:
            logger.warning(f"Failed to initialize sandbox for {config.id}: {e}")
        return None

    async def run_sandbox_tests(self, task: Task) -> Optional[Any]:
        """Run tests in Docker sandbox if enabled and applicable.

        Returns TestResult if tests were run, None if sandbox not enabled/applicable.
        """
        if not self._test_runner:
            return None

        if task.type not in (TaskType.IMPLEMENTATION, TaskType.FIX, TaskType.BUGFIX, TaskType.ENHANCEMENT):
            return None

        github_repo = task.context.get("github_repo")
        if not github_repo:
            self.logger.debug(f"Skipping sandbox tests for {task.id}: no github_repo in context")
            return None

        repo_path = self._git_ops.get_working_directory(task)
        if not repo_path.exists():
            self.logger.warning(f"Repository path does not exist: {repo_path}")
            return None

        task.status = TaskStatus.TESTING
        self.queue.update(task)

        self.logger.info(f"Running tests in sandbox for {task.id} at {repo_path}")

        try:
            test_result = self._test_runner.run_sync(
                repo_path=repo_path,
                packages="./...",
                verbose=True,
            )

            self.logger.info(f"Test result for {task.id}: {test_result.summary}")

            self.activity_manager.append_event(ActivityEvent(
                type="test_complete" if test_result.success else "test_fail",
                agent=self.config.id,
                task_id=task.id,
                title=test_result.summary,
                timestamp=datetime.now(timezone.utc)
            ))

            return test_result

        except Exception as e:
            self.logger.exception(f"Error running sandbox tests for {task.id}: {e}")
            if TestResult:
                return TestResult(
                    success=False,
                    total=0,
                    passed=0,
                    failed=0,
                    skipped=0,
                    duration_seconds=0,
                    error_message=str(e),
                )
            return None

    async def handle_test_failure(self, task: Task, llm_response, test_result) -> None:
        """Handle test failure by feeding results back to agent for fixing."""
        test_retry = task.context.get("_test_retry_count", 0)
        task.context["_test_retry_count"] = test_retry + 1

        failure_report = self._test_runner.format_failure_report(test_result)

        task.context["_test_failure_report"] = failure_report
        task.notes.append(f"Test failure (attempt {test_retry + 1}): {test_result.error_message}")

        task.reset_to_pending()
        self.queue.update(task)

        self.logger.info(f"Task {task.id} reset for test fix retry (attempt {test_retry + 1})")
