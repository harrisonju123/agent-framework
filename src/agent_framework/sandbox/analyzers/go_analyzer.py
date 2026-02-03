"""Go static analysis using golangci-lint."""

import json
import logging
from pathlib import Path
from typing import Optional

from ..docker_executor import DockerExecutor, ExecutionResult
from ..static_analyzer import AnalysisResult, Finding, Severity

logger = logging.getLogger(__name__)


class GoAnalyzer:
    """Run golangci-lint for Go code analysis.

    golangci-lint includes multiple linters:
    - gosec: Security vulnerabilities
    - govet: Correctness issues
    - staticcheck: Code quality
    - errcheck: Unchecked errors
    - and many more
    """

    def __init__(
        self,
        executor: Optional[DockerExecutor] = None,
        image: str = "golangci/golangci-lint:latest",
    ):
        """Initialize Go analyzer.

        Args:
            executor: DockerExecutor instance (created if not provided)
            image: Docker image to use for golangci-lint
        """
        self.executor = executor or DockerExecutor(image=image)

    async def analyze(
        self,
        repo_path: Path,
        config_file: Optional[str] = None,
    ) -> AnalysisResult:
        """Run golangci-lint and return structured results.

        Args:
            repo_path: Path to the Go repository
            config_file: Path to .golangci.yml config (optional)

        Returns:
            AnalysisResult with findings
        """
        # Build golangci-lint command with JSON output
        cmd_parts = ["golangci-lint", "run", "--out-format", "json"]

        if config_file:
            cmd_parts.extend(["--config", config_file])

        # Run on all packages
        cmd_parts.append("./...")

        cmd = " ".join(cmd_parts)

        logger.info(f"Running golangci-lint: {cmd}")

        # Run in Docker
        result = self.executor.run_command(repo_path, cmd)

        # Parse the JSON output
        return self._parse_json_output(result)

    def _parse_json_output(self, result: ExecutionResult) -> AnalysisResult:
        """Parse golangci-lint JSON output into AnalysisResult.

        golangci-lint JSON structure:
        {
          "Issues": [
            {
              "FromLinter": "gosec",
              "Text": "error message",
              "Severity": "error|warning",
              "SourceLines": [...],
              "Pos": {
                "Filename": "file.go",
                "Line": 10,
                "Column": 5
              }
            }
          ]
        }
        """
        findings = []

        try:
            data = json.loads(result.stdout)

            for issue in data.get("Issues", []):
                linter = issue.get("FromLinter", "unknown")
                message = issue.get("Text", "")
                severity_str = issue.get("Severity", "warning")
                pos = issue.get("Pos", {})

                file_path = pos.get("Filename", "unknown")
                line = pos.get("Line", 0)
                column = pos.get("Column", 0)

                # Map golangci-lint severity to our severity levels
                # Security issues (gosec) are critical
                if linter == "gosec" or "security" in message.lower():
                    severity = Severity.CRITICAL
                elif severity_str == "error":
                    severity = Severity.HIGH
                elif linter in ("govet", "staticcheck", "errcheck"):
                    severity = Severity.HIGH
                else:
                    severity = Severity.MEDIUM

                findings.append(Finding(
                    file_path=file_path,
                    line=line,
                    column=column,
                    severity=severity,
                    rule_id=linter,
                    message=message,
                    description=""
                ))

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse golangci-lint JSON output: {e}")
            return AnalysisResult(
                success=False,
                language="go",
                tool="golangci-lint",
                error_message=f"JSON parse error: {e}",
                raw_output=result.stdout
            )

        # Success if exit code 0 or only warnings
        success = result.exit_code == 0 or all(
            f.severity in (Severity.MEDIUM, Severity.LOW) for f in findings
        )

        return AnalysisResult(
            success=success,
            language="go",
            tool="golangci-lint",
            findings=findings,
            duration_seconds=result.duration_seconds,
            raw_output=result.stdout,
            error_message=None if success else "Analysis found critical issues"
        )
