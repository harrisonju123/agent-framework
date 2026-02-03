"""JavaScript/TypeScript static analysis using ESLint."""

import json
import logging
from pathlib import Path
from typing import Optional

from ..docker_executor import DockerExecutor, ExecutionResult
from ..static_analyzer import AnalysisResult, Finding, Severity

logger = logging.getLogger(__name__)


class JavaScriptAnalyzer:
    """Run ESLint for JavaScript/TypeScript analysis."""

    def __init__(
        self,
        executor: Optional[DockerExecutor] = None,
        image: str = "node:18",
    ):
        """Initialize JavaScript/TypeScript analyzer.

        Args:
            executor: DockerExecutor instance (created if not provided)
            image: Docker image to use for Node.js
        """
        self.executor = executor or DockerExecutor(image=image)

    async def analyze(
        self,
        repo_path: Path,
        source_path: str = "src/",
    ) -> AnalysisResult:
        """Run ESLint and return structured results.

        Args:
            repo_path: Path to the JavaScript/TypeScript repository
            source_path: Path to source code (default: src/)

        Returns:
            AnalysisResult with findings
        """
        # Check if ESLint is configured
        has_config = (
            (repo_path / ".eslintrc.js").exists() or
            (repo_path / ".eslintrc.json").exists() or
            (repo_path / "eslint.config.js").exists()
        )

        if not has_config:
            logger.warning("No ESLint config found, using default")

        # Build eslint command with JSON output
        cmd = f"npx eslint {source_path} --format json"

        logger.info(f"Running ESLint: {cmd}")

        # Run in Docker
        result = self.executor.run_command(repo_path, cmd)

        # Parse the JSON output
        return self._parse_json_output(result)

    def _parse_json_output(self, result: ExecutionResult) -> AnalysisResult:
        """Parse ESLint JSON output into AnalysisResult.

        ESLint JSON structure:
        [
          {
            "filePath": "/path/to/file.js",
            "messages": [
              {
                "ruleId": "no-unused-vars",
                "severity": 2,
                "message": "...",
                "line": 10,
                "column": 5,
                "nodeType": "Identifier"
              }
            ],
            "errorCount": 1,
            "warningCount": 0
          }
        ]
        """
        findings = []

        try:
            data = json.loads(result.stdout)

            for file_result in data:
                file_path = file_result.get("filePath", "unknown")

                for message in file_result.get("messages", []):
                    rule_id = message.get("ruleId", "eslint")
                    severity_level = message.get("severity", 1)
                    msg = message.get("message", "")
                    line = message.get("line", 0)
                    column = message.get("column", 0)

                    # Map ESLint severity (1=warning, 2=error) to our severity
                    # Check for security-related rules
                    if any(sec in rule_id.lower() for sec in ["security", "xss", "injection", "eval"]):
                        severity = Severity.CRITICAL
                    elif severity_level == 2:  # ESLint error
                        severity = Severity.HIGH
                    elif severity_level == 1:  # ESLint warning
                        severity = Severity.MEDIUM
                    else:
                        severity = Severity.LOW

                    findings.append(Finding(
                        file_path=file_path,
                        line=line,
                        column=column,
                        severity=severity,
                        rule_id=rule_id,
                        message=msg,
                        description=""
                    ))

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse ESLint JSON output: {e}")
            return AnalysisResult(
                success=False,
                language="javascript",
                tool="eslint",
                error_message=f"JSON parse error: {e}",
                raw_output=result.stdout
            )

        # Success if no critical or high severity issues
        critical_count = len([f for f in findings if f.severity == Severity.CRITICAL])
        success = critical_count == 0

        return AnalysisResult(
            success=success,
            language="javascript",
            tool="eslint",
            findings=findings,
            duration_seconds=result.duration_seconds,
            raw_output=result.stdout,
            error_message=None if success else f"Found {critical_count} critical issues"
        )
