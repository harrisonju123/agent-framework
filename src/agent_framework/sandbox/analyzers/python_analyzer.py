"""Python static analysis using pylint, mypy, and bandit."""

import json
import logging
from pathlib import Path
from typing import Optional, List

from ..docker_executor import DockerExecutor, ExecutionResult
from ..static_analyzer import AnalysisResult, Finding, Severity

logger = logging.getLogger(__name__)


class PythonAnalyzer:
    """Run multiple Python analysis tools.

    - pylint: Code quality and style
    - mypy: Type checking
    - bandit: Security vulnerabilities
    """

    def __init__(
        self,
        executor: Optional[DockerExecutor] = None,
        image: str = "python:3.11",
    ):
        """Initialize Python analyzer.

        Args:
            executor: DockerExecutor instance (created if not provided)
            image: Docker image to use for Python analysis
        """
        self.executor = executor or DockerExecutor(image=image)

    async def analyze(
        self,
        repo_path: Path,
        source_path: str = "src/",
    ) -> AnalysisResult:
        """Run Python analyzers and return structured results.

        Args:
            repo_path: Path to the Python repository
            source_path: Path to source code (default: src/)

        Returns:
            AnalysisResult with findings
        """
        findings: List[Finding] = []

        # Install tools (in Docker image)
        install_cmd = "pip install --quiet pylint mypy bandit"
        self.executor.run_command(repo_path, install_cmd)

        # Run bandit (security) first - highest priority
        bandit_result = await self._run_bandit(repo_path, source_path)
        findings.extend(bandit_result)

        # Run pylint (code quality)
        pylint_result = await self._run_pylint(repo_path, source_path)
        findings.extend(pylint_result)

        # Run mypy (type checking)
        mypy_result = await self._run_mypy(repo_path, source_path)
        findings.extend(mypy_result)

        # Determine success
        critical_count = len([f for f in findings if f.severity == Severity.CRITICAL])
        success = critical_count == 0

        return AnalysisResult(
            success=success,
            language="python",
            tool="pylint+mypy+bandit",
            findings=findings,
            duration_seconds=0.0,
            raw_output="",
            error_message=None if success else f"Found {critical_count} critical security issues"
        )

    async def _run_bandit(self, repo_path: Path, source_path: str) -> List[Finding]:
        """Run bandit security scanner."""
        cmd = f"bandit -r {source_path} -f json"
        result = self.executor.run_command(repo_path, cmd)

        findings = []
        try:
            data = json.loads(result.stdout)

            for issue in data.get("results", []):
                severity_str = issue.get("issue_severity", "LOW")
                confidence = issue.get("issue_confidence", "LOW")

                # Map bandit severity to our severity
                if severity_str == "HIGH" and confidence == "HIGH":
                    severity = Severity.CRITICAL
                elif severity_str in ("HIGH", "MEDIUM"):
                    severity = Severity.HIGH
                else:
                    severity = Severity.MEDIUM

                findings.append(Finding(
                    file_path=issue.get("filename", "unknown"),
                    line=issue.get("line_number", 0),
                    column=issue.get("col_offset", 0),
                    severity=severity,
                    rule_id=issue.get("test_id", "bandit"),
                    message=issue.get("issue_text", ""),
                    description=f"Confidence: {confidence}"
                ))

        except json.JSONDecodeError:
            logger.warning("Failed to parse bandit JSON output")

        return findings

    async def _run_pylint(self, repo_path: Path, source_path: str) -> List[Finding]:
        """Run pylint code quality checker."""
        cmd = f"pylint {source_path} --output-format=json"
        result = self.executor.run_command(repo_path, cmd)

        findings = []
        try:
            data = json.loads(result.stdout)

            for issue in data:
                message_id = issue.get("message-id", "")
                severity_type = issue.get("type", "")

                # Map pylint message types to severity
                if severity_type == "error":
                    severity = Severity.HIGH
                elif severity_type == "warning":
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW

                findings.append(Finding(
                    file_path=issue.get("path", "unknown"),
                    line=issue.get("line", 0),
                    column=issue.get("column", 0),
                    severity=severity,
                    rule_id=message_id,
                    message=issue.get("message", ""),
                    description=issue.get("symbol", "")
                ))

        except json.JSONDecodeError:
            logger.warning("Failed to parse pylint JSON output")

        return findings

    async def _run_mypy(self, repo_path: Path, source_path: str) -> List[Finding]:
        """Run mypy type checker."""
        cmd = f"mypy {source_path} --show-column-numbers --no-error-summary"
        result = self.executor.run_command(repo_path, cmd)

        findings = []

        # Parse mypy text output (format: file.py:line:col: error: message)
        for line in result.stdout.split("\n"):
            if not line.strip() or ":" not in line:
                continue

            parts = line.split(":", 4)
            if len(parts) < 5:
                continue

            file_path = parts[0].strip()
            try:
                line_num = int(parts[1].strip())
                col_num = int(parts[2].strip())
            except ValueError:
                continue

            level = parts[3].strip()
            message = parts[4].strip()

            # Map mypy level to severity
            if level == "error":
                severity = Severity.MEDIUM
            else:
                severity = Severity.LOW

            findings.append(Finding(
                file_path=file_path,
                line=line_num,
                column=col_num,
                severity=severity,
                rule_id="mypy",
                message=message,
                description="Type checking issue"
            ))

        return findings
