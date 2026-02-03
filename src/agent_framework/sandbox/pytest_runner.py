"""Pytest runner with structured output parsing."""

import json
import logging
from pathlib import Path
from typing import Optional

from .docker_executor import DockerExecutor, ExecutionResult
from .test_runner import TestResult, TestCase, TestStatus

logger = logging.getLogger(__name__)


class PytestRunner:
    """Run and parse pytest tests in Docker containers.

    Uses `pytest --json-report` for structured output parsing.
    """

    def __init__(
        self,
        executor: Optional[DockerExecutor] = None,
        image: str = "python:3.11",
    ):
        """Initialize pytest runner.

        Args:
            executor: DockerExecutor instance (created if not provided)
            image: Docker image to use for Python tests
        """
        self.executor = executor or DockerExecutor(image=image)

    async def run(
        self,
        repo_path: Path,
        test_path: str = "tests/",
        verbose: bool = True,
        markers: Optional[str] = None,
        timeout: int = 300,
    ) -> TestResult:
        """Run pytest and return structured results.

        Args:
            repo_path: Path to the Python repository
            test_path: Path pattern to test (default: tests/)
            verbose: Enable verbose output
            markers: Pytest markers to filter tests
            timeout: Test timeout in seconds

        Returns:
            TestResult with parsed test output
        """
        # Build pytest command with JSON output
        cmd_parts = ["pytest", test_path]

        if verbose:
            cmd_parts.append("-v")

        if markers:
            cmd_parts.extend(["-m", markers])

        # Use junit-xml for structured output (more widely supported than json-report)
        junit_file = "/tmp/pytest-results.xml"
        cmd_parts.extend(["--junit-xml", junit_file])

        # Add timeout
        cmd_parts.extend(["--timeout", str(timeout)])

        test_cmd = " ".join(cmd_parts)

        logger.info(f"Running pytest: {test_cmd}")

        # Run in Docker
        result = self.executor.run_tests(repo_path, test_cmd)

        # Parse the output
        return self._parse_output(result)

    def run_sync(
        self,
        repo_path: Path,
        test_path: str = "tests/",
        verbose: bool = True,
        markers: Optional[str] = None,
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
                    self.run(repo_path, test_path, verbose, markers, timeout)
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                self.run(repo_path, test_path, verbose, markers, timeout)
            )

    def _parse_output(self, result: ExecutionResult) -> TestResult:
        """Parse pytest output into TestResult.

        Parses pytest's text output for test results.
        """
        test_cases = []
        passed = 0
        failed = 0
        skipped = 0

        # Parse pytest output lines
        lines = result.stdout.strip().split("\n")

        for line in lines:
            # Look for test result lines (pytest format: test.py::test_name PASSED)
            if " PASSED" in line:
                passed += 1
                parts = line.split("::")
                if len(parts) >= 2:
                    package = parts[0].strip()
                    name = parts[1].split(" ")[0].strip()
                    test_cases.append(TestCase(
                        package=package,
                        name=name,
                        status=TestStatus.PASS,
                        duration_seconds=0.0,
                        output=""
                    ))
            elif " FAILED" in line:
                failed += 1
                parts = line.split("::")
                if len(parts) >= 2:
                    package = parts[0].strip()
                    name = parts[1].split(" ")[0].strip()
                    test_cases.append(TestCase(
                        package=package,
                        name=name,
                        status=TestStatus.FAIL,
                        duration_seconds=0.0,
                        output=""
                    ))
            elif " SKIPPED" in line:
                skipped += 1

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
