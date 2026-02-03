"""Ruby static analysis using RuboCop."""

import json
import logging
from pathlib import Path
from typing import Optional

from ..docker_executor import DockerExecutor, ExecutionResult
from ..static_analyzer import AnalysisResult, Finding, Severity

logger = logging.getLogger(__name__)


class RubyAnalyzer:
    """Run RuboCop for Ruby code analysis."""

    def __init__(
        self,
        executor: Optional[DockerExecutor] = None,
        image: str = "ruby:3.2",
    ):
        """Initialize Ruby analyzer.

        Args:
            executor: DockerExecutor instance (created if not provided)
            image: Docker image to use for Ruby
        """
        self.executor = executor or DockerExecutor(image=image)

    async def analyze(
        self,
        repo_path: Path,
        source_path: str = "lib/",
    ) -> AnalysisResult:
        """Run RuboCop and return structured results.

        Args:
            repo_path: Path to the Ruby repository
            source_path: Path to source code (default: lib/)

        Returns:
            AnalysisResult with findings
        """
        # Install RuboCop if not present
        install_cmd = "gem install rubocop --quiet"
        self.executor.run_command(repo_path, install_cmd)

        # Build rubocop command with JSON output
        cmd = f"rubocop {source_path} --format json"

        logger.info(f"Running RuboCop: {cmd}")

        # Run in Docker
        result = self.executor.run_command(repo_path, cmd)

        # Parse the JSON output
        return self._parse_json_output(result)

    def _parse_json_output(self, result: ExecutionResult) -> AnalysisResult:
        """Parse RuboCop JSON output into AnalysisResult.

        RuboCop JSON structure:
        {
          "files": [
            {
              "path": "lib/file.rb",
              "offenses": [
                {
                  "severity": "warning|error|convention",
                  "message": "...",
                  "cop_name": "Style/StringLiterals",
                  "corrected": false,
                  "location": {
                    "start_line": 10,
                    "start_column": 5,
                    "last_line": 10,
                    "last_column": 20
                  }
                }
              ]
            }
          ],
          "summary": {
            "offense_count": 5,
            "target_file_count": 3,
            "inspected_file_count": 3
          }
        }
        """
        findings = []

        try:
            data = json.loads(result.stdout)

            for file_result in data.get("files", []):
                file_path = file_result.get("path", "unknown")

                for offense in file_result.get("offenses", []):
                    severity_str = offense.get("severity", "convention")
                    cop_name = offense.get("cop_name", "rubocop")
                    message = offense.get("message", "")
                    location = offense.get("location", {})

                    line = location.get("start_line", 0)
                    column = location.get("start_column", 0)

                    # Map RuboCop severity to our severity
                    # Check for security-related cops
                    if "Security" in cop_name:
                        severity = Severity.CRITICAL
                    elif severity_str in ("error", "fatal"):
                        severity = Severity.HIGH
                    elif severity_str == "warning":
                        severity = Severity.MEDIUM
                    else:  # convention, refactor, info
                        severity = Severity.LOW

                    findings.append(Finding(
                        file_path=file_path,
                        line=line,
                        column=column,
                        severity=severity,
                        rule_id=cop_name,
                        message=message,
                        description=""
                    ))

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse RuboCop JSON output: {e}")
            return AnalysisResult(
                success=False,
                language="ruby",
                tool="rubocop",
                error_message=f"JSON parse error: {e}",
                raw_output=result.stdout
            )

        # Success if no critical issues
        critical_count = len([f for f in findings if f.severity == Severity.CRITICAL])
        success = critical_count == 0

        return AnalysisResult(
            success=success,
            language="ruby",
            tool="rubocop",
            findings=findings,
            duration_seconds=result.duration_seconds,
            raw_output=result.stdout,
            error_message=None if success else f"Found {critical_count} critical security issues"
        )
