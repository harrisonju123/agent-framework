"""Static analysis orchestration for multiple languages."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict

from .docker_executor import DockerExecutor

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Issue severity level."""
    CRITICAL = "critical"  # Security vulnerabilities, syntax errors
    HIGH = "high"          # Important code quality issues
    MEDIUM = "medium"      # Style violations, minor issues
    LOW = "low"            # Suggestions, informational


@dataclass
class Finding:
    """Individual static analysis finding."""
    file_path: str
    line: int
    column: int
    severity: Severity
    rule_id: str
    message: str
    description: str = ""


@dataclass
class AnalysisResult:
    """Aggregated static analysis result."""
    success: bool
    language: str
    tool: str
    findings: List[Finding] = field(default_factory=list)
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    duration_seconds: float = 0.0
    raw_output: str = ""
    error_message: Optional[str] = None

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASS" if self.success else "FAIL"
        total = len(self.findings)
        return (
            f"{status}: {total} issues found "
            f"({self.critical_count} critical, {self.high_count} high, "
            f"{self.medium_count} medium, {self.low_count} low)"
        )

    @property
    def critical_findings(self) -> List[Finding]:
        """Get list of critical findings."""
        return [f for f in self.findings if f.severity == Severity.CRITICAL]

    @property
    def blocking_issues(self) -> bool:
        """Check if there are blocking (critical) issues."""
        return self.critical_count > 0


class LanguageDetector:
    """Detect programming language from repository files."""

    @staticmethod
    def detect_language(repo_path: Path) -> Optional[str]:
        """Detect primary language from repository structure.

        Returns:
            Language name (go, python, javascript, ruby) or None
        """
        # Check for language-specific files
        if list(repo_path.glob("**/*.go")):
            return "go"
        elif list(repo_path.glob("**/pytest.ini")) or list(repo_path.glob("**/setup.py")):
            return "python"
        elif list(repo_path.glob("**/package.json")):
            # Check if it's TypeScript or JavaScript
            if list(repo_path.glob("**/tsconfig.json")):
                return "typescript"
            return "javascript"
        elif list(repo_path.glob("**/Gemfile")) or list(repo_path.glob("**/*.gemspec")):
            return "ruby"

        return None

    @staticmethod
    def detect_analyzer(language: str) -> Optional[str]:
        """Get recommended analyzer for a language.

        Args:
            language: Language name (go, python, javascript, typescript, ruby)

        Returns:
            Analyzer name or None
        """
        analyzers = {
            "go": "golangci-lint",
            "python": "pylint",
            "javascript": "eslint",
            "typescript": "eslint",
            "ruby": "rubocop",
        }
        return analyzers.get(language)


class StaticAnalyzer:
    """Orchestrate static analysis across multiple languages."""

    def __init__(self, executor: Optional[DockerExecutor] = None):
        """Initialize static analyzer.

        Args:
            executor: DockerExecutor instance (created if not provided)
        """
        self.executor = executor or DockerExecutor()

    async def analyze(
        self,
        repo_path: Path,
        language: Optional[str] = None,
        analyzer: Optional[str] = None,
    ) -> AnalysisResult:
        """Run static analysis on a repository.

        Args:
            repo_path: Path to the repository
            language: Language override (auto-detected if not provided)
            analyzer: Analyzer override (auto-selected if not provided)

        Returns:
            AnalysisResult with findings
        """
        # Detect language if not provided
        if not language:
            language = LanguageDetector.detect_language(repo_path)
            if not language:
                return AnalysisResult(
                    success=False,
                    language="unknown",
                    tool="none",
                    error_message="Could not detect language from repository structure"
                )

        # Select analyzer if not provided
        if not analyzer:
            analyzer = LanguageDetector.detect_analyzer(language)
            if not analyzer:
                return AnalysisResult(
                    success=False,
                    language=language,
                    tool="none",
                    error_message=f"No analyzer configured for language: {language}"
                )

        logger.info(f"Running static analysis: language={language}, analyzer={analyzer}")

        # Import language-specific analyzer
        if language == "go":
            from .analyzers.go_analyzer import GoAnalyzer
            lang_analyzer = GoAnalyzer(self.executor)
        elif language == "python":
            from .analyzers.python_analyzer import PythonAnalyzer
            lang_analyzer = PythonAnalyzer(self.executor)
        elif language in ("javascript", "typescript"):
            from .analyzers.javascript_analyzer import JavaScriptAnalyzer
            lang_analyzer = JavaScriptAnalyzer(self.executor)
        elif language == "ruby":
            from .analyzers.ruby_analyzer import RubyAnalyzer
            lang_analyzer = RubyAnalyzer(self.executor)
        else:
            return AnalysisResult(
                success=False,
                language=language,
                tool=analyzer,
                error_message=f"Unsupported language: {language}"
            )

        # Run analysis
        result = await lang_analyzer.analyze(repo_path)

        # Count findings by severity
        result.critical_count = len([f for f in result.findings if f.severity == Severity.CRITICAL])
        result.high_count = len([f for f in result.findings if f.severity == Severity.HIGH])
        result.medium_count = len([f for f in result.findings if f.severity == Severity.MEDIUM])
        result.low_count = len([f for f in result.findings if f.severity == Severity.LOW])

        return result

    def format_findings_report(self, result: AnalysisResult, max_findings: int = 20) -> str:
        """Format findings report for LLM consumption.

        Args:
            result: Analysis result
            max_findings: Maximum number of findings to include

        Returns:
            Markdown-formatted report
        """
        lines = [
            "## Static Analysis Report",
            "",
            f"**Language:** {result.language}",
            f"**Tool:** {result.tool}",
            f"**Summary:** {result.summary}",
            "",
        ]

        if result.error_message:
            lines.extend([
                "### Error",
                result.error_message,
                "",
            ])
            return "\n".join(lines)

        # Group findings by severity
        critical = [f for f in result.findings if f.severity == Severity.CRITICAL]
        high = [f for f in result.findings if f.severity == Severity.HIGH]
        medium = [f for f in result.findings if f.severity == Severity.MEDIUM]
        low = [f for f in result.findings if f.severity == Severity.LOW]

        # Show critical findings first
        if critical:
            lines.append("### 🔴 Critical Issues")
            lines.append("")
            for finding in critical[:max_findings]:
                lines.append(f"**{finding.file_path}:{finding.line}** - {finding.rule_id}")
                lines.append(f"{finding.message}")
                if finding.description:
                    lines.append(f"_{finding.description}_")
                lines.append("")

        # Show high severity
        if high:
            lines.append("### 🟠 High Priority Issues")
            lines.append("")
            for finding in high[:max_findings]:
                lines.append(f"**{finding.file_path}:{finding.line}** - {finding.rule_id}")
                lines.append(f"{finding.message}")
                lines.append("")

        # Summarize medium and low
        if medium:
            lines.append(f"### 🟡 Medium Issues ({len(medium)})")
            lines.append("")
            for finding in medium[:5]:
                lines.append(f"- {finding.file_path}:{finding.line} - {finding.message}")
            if len(medium) > 5:
                lines.append(f"- ... and {len(medium) - 5} more")
            lines.append("")

        if low:
            lines.append(f"### 🔵 Low Priority / Suggestions ({len(low)})")
            lines.append("")

        return "\n".join(lines)
