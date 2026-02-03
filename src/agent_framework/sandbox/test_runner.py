"""Go test runner with structured output parsing."""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

from .docker_executor import DockerExecutor, ExecutionResult

logger = logging.getLogger(__name__)


class TestStatus(str, Enum):
    """Individual test status."""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class TestCase:
    """Individual test case result."""
    package: str
    name: str
    status: TestStatus
    duration_seconds: float = 0.0
    output: str = ""


@dataclass
class TestResult:
    """Aggregated test run result."""
    success: bool
    total: int
    passed: int
    failed: int
    skipped: int
    duration_seconds: float
    test_cases: List[TestCase] = field(default_factory=list)
    raw_output: str = ""
    error_message: Optional[str] = None

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASS" if self.success else "FAIL"
        return (
            f"{status}: {self.passed}/{self.total} tests passed "
            f"({self.failed} failed, {self.skipped} skipped) "
            f"in {self.duration_seconds:.1f}s"
        )

    @property
    def failed_tests(self) -> List[TestCase]:
        """Get list of failed tests."""
        return [t for t in self.test_cases if t.status == TestStatus.FAIL]


class GoTestRunner:
    """Run and parse Go tests in Docker containers.

    Uses `go test -json` for structured output parsing.
    """

    def __init__(
        self,
        executor: Optional[DockerExecutor] = None,
        image: str = "golang:1.22",
    ):
        """Initialize Go test runner.

        Args:
            executor: DockerExecutor instance (created if not provided)
            image: Docker image to use for Go tests
        """
        self.executor = executor or DockerExecutor(image=image)

    async def run(
        self,
        repo_path: Path,
        packages: str = "./...",
        verbose: bool = True,
        race: bool = False,
        cover: bool = False,
        timeout: str = "5m",
    ) -> TestResult:
        """Run Go tests and return structured results.

        Args:
            repo_path: Path to the Go repository
            packages: Package pattern to test (default: ./...)
            verbose: Enable verbose output
            race: Enable race detector
            cover: Enable coverage
            timeout: Test timeout (Go duration format)

        Returns:
            TestResult with parsed test output
        """
        # Build go test command with JSON output for parsing
        cmd_parts = ["go", "test", "-json"]

        if verbose:
            cmd_parts.append("-v")
        if race:
            cmd_parts.append("-race")
        if cover:
            cmd_parts.append("-cover")

        cmd_parts.extend(["-timeout", timeout, packages])

        test_cmd = " ".join(cmd_parts)

        logger.info(f"Running Go tests: {test_cmd}")

        # Run in Docker
        result = self.executor.run_tests(repo_path, test_cmd)

        # Parse the JSON output
        return self._parse_json_output(result)

    def run_sync(
        self,
        repo_path: Path,
        packages: str = "./...",
        verbose: bool = True,
        race: bool = False,
        cover: bool = False,
        timeout: str = "5m",
    ) -> TestResult:
        """Synchronous version of run() for non-async contexts."""
        import asyncio

        # Check if we're in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, can't use run_until_complete
            # Just call the async method directly via coroutine
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.run(repo_path, packages, verbose, race, cover, timeout)
                )
                return future.result()
        except RuntimeError:
            # No event loop, safe to use asyncio.run
            return asyncio.run(
                self.run(repo_path, packages, verbose, race, cover, timeout)
            )

    def _parse_json_output(self, result: ExecutionResult) -> TestResult:
        """Parse go test -json output into TestResult.

        Go test JSON output consists of one JSON object per line with fields:
        - Time: timestamp
        - Action: run, output, pass, fail, skip
        - Package: package name
        - Test: test name (if applicable)
        - Output: output text (for action=output)
        - Elapsed: duration in seconds (for pass/fail)
        """
        test_cases: List[TestCase] = []
        test_outputs: dict = {}  # (package, test) -> output lines
        package_status: dict = {}  # package -> status

        passed = 0
        failed = 0
        skipped = 0

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Not JSON, might be a build error or other output
                continue

            action = event.get("Action", "")
            package = event.get("Package", "")
            test_name = event.get("Test", "")
            output = event.get("Output", "")
            elapsed = event.get("Elapsed", 0)

            # Track output for tests
            key = (package, test_name)
            if output:
                if key not in test_outputs:
                    test_outputs[key] = []
                test_outputs[key].append(output)

            # Handle test completion events
            if action == "pass" and test_name:
                passed += 1
                test_cases.append(TestCase(
                    package=package,
                    name=test_name,
                    status=TestStatus.PASS,
                    duration_seconds=elapsed,
                    output="".join(test_outputs.get(key, [])),
                ))
            elif action == "fail" and test_name:
                failed += 1
                test_cases.append(TestCase(
                    package=package,
                    name=test_name,
                    status=TestStatus.FAIL,
                    duration_seconds=elapsed,
                    output="".join(test_outputs.get(key, [])),
                ))
            elif action == "skip" and test_name:
                skipped += 1
                test_cases.append(TestCase(
                    package=package,
                    name=test_name,
                    status=TestStatus.SKIP,
                    duration_seconds=elapsed,
                    output="".join(test_outputs.get(key, [])),
                ))
            elif action in ("pass", "fail") and not test_name:
                # Package-level result
                package_status[package] = action

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
                # Build error or other issue
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

    def format_failure_report(self, result: TestResult) -> str:
        """Format a detailed failure report for LLM consumption.

        This is designed to be fed back to the agent so it can fix failures.
        """
        if result.success:
            return result.summary

        lines = [
            "## Test Failure Report",
            "",
            f"**Summary:** {result.summary}",
            "",
        ]

        if result.error_message:
            lines.extend([
                "### Error",
                result.error_message,
                "",
            ])

        failed_tests = result.failed_tests
        if failed_tests:
            lines.append("### Failed Tests")
            lines.append("")

            for test in failed_tests[:10]:  # Limit to 10 failures
                lines.append(f"#### {test.package}/{test.name}")
                lines.append("")
                if test.output:
                    # Truncate long output
                    output = test.output
                    if len(output) > 2000:
                        output = output[:2000] + "\n... (truncated)"
                    lines.append("```")
                    lines.append(output.strip())
                    lines.append("```")
                    lines.append("")

            if len(failed_tests) > 10:
                lines.append(f"... and {len(failed_tests) - 10} more failed tests")

        return "\n".join(lines)
