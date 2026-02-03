"""RSpec runner with structured output parsing."""

import json
import logging
from pathlib import Path
from typing import Optional

from .docker_executor import DockerExecutor, ExecutionResult
from .test_runner import TestResult, TestCase, TestStatus

logger = logging.getLogger(__name__)


class RspecRunner:
    """Run and parse RSpec tests in Docker containers.

    Uses `rspec --format json` for structured output parsing.
    """

    def __init__(
        self,
        executor: Optional[DockerExecutor] = None,
        image: str = "ruby:3.2",
    ):
        """Initialize RSpec runner.

        Args:
            executor: DockerExecutor instance (created if not provided)
            image: Docker image to use for Ruby tests
        """
        self.executor = executor or DockerExecutor(image=image)

    async def run(
        self,
        repo_path: Path,
        spec_path: str = "spec/",
        verbose: bool = True,
        timeout: int = 300,
    ) -> TestResult:
        """Run RSpec and return structured results.

        Args:
            repo_path: Path to the Ruby repository
            spec_path: Path pattern to test (default: spec/)
            verbose: Enable verbose output
            timeout: Test timeout in seconds

        Returns:
            TestResult with parsed test output
        """
        # Build rspec command with JSON output
        json_file = "/tmp/rspec-results.json"
        cmd_parts = ["rspec", spec_path, "--format", "json", "--out", json_file]

        if verbose:
            # Also output progress format to stdout
            cmd_parts.extend(["--format", "progress"])

        test_cmd = " ".join(cmd_parts)

        logger.info(f"Running RSpec: {test_cmd}")

        # Run in Docker
        result = self.executor.run_tests(repo_path, test_cmd)

        # Parse the JSON output
        return self._parse_json_output(result)

    def run_sync(
        self,
        repo_path: Path,
        spec_path: str = "spec/",
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
                    self.run(repo_path, spec_path, verbose, timeout)
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                self.run(repo_path, spec_path, verbose, timeout)
            )

    def _parse_json_output(self, result: ExecutionResult) -> TestResult:
        """Parse rspec --format json output into TestResult.

        RSpec JSON output structure:
        {
          "version": "3.x.x",
          "examples": [
            {
              "description": "test description",
              "full_description": "full test name",
              "status": "passed" | "failed" | "pending",
              "file_path": "spec/file_spec.rb",
              "line_number": 10,
              "run_time": 0.00123,
              "exception": { "message": "..." }  // if failed
            }
          ],
          "summary": {
            "duration": 1.234,
            "example_count": 10,
            "failure_count": 1,
            "pending_count": 0
          }
        }
        """
        test_cases = []
        passed = 0
        failed = 0
        skipped = 0

        try:
            # Parse JSON output (look for JSON in stdout or try to read the file)
            data = None
            if result.stdout.strip().startswith("{"):
                data = json.loads(result.stdout)
            else:
                # Try to extract JSON from mixed output
                for line in result.stdout.split("\n"):
                    if line.strip().startswith("{"):
                        try:
                            data = json.loads(line)
                            break
                        except json.JSONDecodeError:
                            continue

            if data:
                summary = data.get("summary", {})
                passed = summary.get("example_count", 0) - summary.get("failure_count", 0) - summary.get("pending_count", 0)
                failed = summary.get("failure_count", 0)
                skipped = summary.get("pending_count", 0)

                # Parse individual test results
                for example in data.get("examples", []):
                    status_str = example.get("status", "")
                    file_path = example.get("file_path", "unknown")
                    full_desc = example.get("full_description", "")
                    run_time = example.get("run_time", 0.0)
                    exception = example.get("exception", {})

                    # Map RSpec status to our TestStatus
                    if status_str == "passed":
                        status = TestStatus.PASS
                    elif status_str == "failed":
                        status = TestStatus.FAIL
                    elif status_str == "pending":
                        status = TestStatus.SKIP
                    else:
                        status = TestStatus.SKIP

                    output = ""
                    if exception:
                        output = exception.get("message", "")

                    test_cases.append(TestCase(
                        package=file_path,
                        name=full_desc,
                        status=status,
                        duration_seconds=run_time,
                        output=output
                    ))

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse RSpec JSON output: {e}")

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
