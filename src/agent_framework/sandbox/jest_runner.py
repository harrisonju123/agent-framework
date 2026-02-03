"""Jest runner with structured output parsing."""

import json
import logging
from pathlib import Path
from typing import Optional

from .docker_executor import DockerExecutor, ExecutionResult
from .test_runner import TestResult, TestCase, TestStatus

logger = logging.getLogger(__name__)


class JestRunner:
    """Run and parse Jest tests in Docker containers.

    Uses `jest --json` for structured output parsing.
    """

    def __init__(
        self,
        executor: Optional[DockerExecutor] = None,
        image: str = "node:18",
    ):
        """Initialize Jest runner.

        Args:
            executor: DockerExecutor instance (created if not provided)
            image: Docker image to use for Node.js tests
        """
        self.executor = executor or DockerExecutor(image=image)

    async def run(
        self,
        repo_path: Path,
        test_path: Optional[str] = None,
        verbose: bool = True,
        timeout: int = 300,
    ) -> TestResult:
        """Run Jest and return structured results.

        Args:
            repo_path: Path to the Node.js repository
            test_path: Path pattern to test (default: all tests)
            verbose: Enable verbose output
            timeout: Test timeout in seconds (Jest default is in milliseconds)

        Returns:
            TestResult with parsed test output
        """
        # Build jest command with JSON output
        cmd_parts = ["npx", "jest", "--json"]

        if verbose:
            cmd_parts.append("--verbose")

        if test_path:
            cmd_parts.append(test_path)

        # Add timeout (Jest uses milliseconds)
        cmd_parts.extend(["--testTimeout", str(timeout * 1000)])

        test_cmd = " ".join(cmd_parts)

        logger.info(f"Running Jest: {test_cmd}")

        # Run in Docker
        result = self.executor.run_tests(repo_path, test_cmd)

        # Parse the JSON output
        return self._parse_json_output(result)

    def run_sync(
        self,
        repo_path: Path,
        test_path: Optional[str] = None,
        verbose: bool = True,
        timeout: int = 300,
    ) -> TestResult:
        """Synchronous version of run() for non-async contexts."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.run(repo_path, test_path, verbose, timeout)
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                self.run(repo_path, test_path, verbose, timeout)
            )

    def _parse_json_output(self, result: ExecutionResult) -> TestResult:
        """Parse jest --json output into TestResult.

        Jest JSON output structure:
        {
          "success": bool,
          "testResults": [
            {
              "name": "test file path",
              "status": "passed" | "failed",
              "assertionResults": [
                {
                  "fullName": "test name",
                  "status": "passed" | "failed" | "pending",
                  "duration": ms,
                  "failureMessages": [...]
                }
              ]
            }
          ],
          "numTotalTests": int,
          "numPassedTests": int,
          "numFailedTests": int,
          "numPendingTests": int
        }
        """
        test_cases = []
        passed = 0
        failed = 0
        skipped = 0

        try:
            # Jest outputs JSON to stdout
            data = json.loads(result.stdout)

            passed = data.get("numPassedTests", 0)
            failed = data.get("numFailedTests", 0)
            skipped = data.get("numPendingTests", 0)

            # Parse individual test results
            for test_file in data.get("testResults", []):
                file_name = test_file.get("name", "unknown")

                for assertion in test_file.get("assertionResults", []):
                    test_name = assertion.get("fullName", "")
                    status_str = assertion.get("status", "")
                    duration_ms = assertion.get("duration", 0)
                    failure_messages = assertion.get("failureMessages", [])

                    # Map Jest status to our TestStatus
                    if status_str == "passed":
                        status = TestStatus.PASS
                    elif status_str == "failed":
                        status = TestStatus.FAIL
                    elif status_str == "pending":
                        status = TestStatus.SKIP
                    else:
                        status = TestStatus.SKIP

                    test_cases.append(TestCase(
                        package=file_name,
                        name=test_name,
                        status=status,
                        duration_seconds=duration_ms / 1000.0,
                        output="\n".join(failure_messages) if failure_messages else ""
                    ))

        except json.JSONDecodeError:
            # Fallback to text parsing if JSON is invalid
            logger.warning("Failed to parse Jest JSON output, falling back to text parsing")

        total = passed + failed + skipped
        success = result.exit_code == 0 and failed == 0

        # Determine error message if failed
        error_message = None
        if not success:
            if failed > 0:
                failed_names = [t.name for t in test_cases if t.status == TestStatus.FAIL]
                error_message = f"Failed tests: {', '.join(failed_names[:5])}"
                if len(failed_names) > 5:
                    error_message += f" (+{len(failed_names) - 5} more)"
            elif result.exit_code != 0:
                error_message = result.stderr or "Build or runtime error"

        return TestResult(
            success=success,
            total=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            duration_seconds=result.duration_seconds,
            test_cases=test_cases,
            raw_output=result.stdout,
            error_message=error_message,
        )
