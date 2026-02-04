"""Finding aggregation logic for repository analysis.

Groups issues by file/module location (flow-based grouping) rather than by issue type.
This keeps related code together for easier remediation.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class FindingSeverity(str, Enum):
    """Severity levels for findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FindingType(str, Enum):
    """Types of findings from static analysis."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    CODE_QUALITY = "code_quality"
    STYLE = "style"


@dataclass
class Finding:
    """A single finding from static analysis."""
    file_path: str
    line: int
    column: Optional[int]
    severity: FindingSeverity
    finding_type: FindingType
    rule_id: str  # e.g., "gosec:G101", "bandit:B201"
    message: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
            "severity": self.severity.value,
            "finding_type": self.finding_type.value,
            "rule_id": self.rule_id,
            "message": self.message,
            "suggestion": self.suggestion,
            "code_snippet": self.code_snippet,
        }


@dataclass
class FileGroup:
    """A group of findings for a single file or module."""
    path: str
    findings: list[Finding] = field(default_factory=list)
    is_module: bool = False  # True if this represents a directory/module

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == FindingSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == FindingSeverity.HIGH)

    @property
    def medium_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == FindingSeverity.MEDIUM)

    @property
    def low_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == FindingSeverity.LOW)

    @property
    def total_count(self) -> int:
        return len(self.findings)

    @property
    def severity_summary(self) -> str:
        """Generate human-readable severity summary."""
        parts = []
        if self.critical_count > 0:
            parts.append(f"{self.critical_count} critical")
        if self.high_count > 0:
            parts.append(f"{self.high_count} high")
        if self.medium_count > 0:
            parts.append(f"{self.medium_count} medium")
        if self.low_count > 0:
            parts.append(f"{self.low_count} low")
        return ", ".join(parts) if parts else "no issues"

    def to_jira_description(self) -> str:
        """Generate JIRA subtask description with all findings."""
        lines = [f"## Issues in `{self.path}`\n"]

        # Group by severity for better readability
        by_severity: dict[FindingSeverity, list[Finding]] = defaultdict(list)
        for finding in self.findings:
            by_severity[finding.severity].append(finding)

        severity_order = [
            FindingSeverity.CRITICAL,
            FindingSeverity.HIGH,
            FindingSeverity.MEDIUM,
            FindingSeverity.LOW,
        ]
        severity_icons = {
            FindingSeverity.CRITICAL: "🔴",
            FindingSeverity.HIGH: "🟠",
            FindingSeverity.MEDIUM: "🟡",
            FindingSeverity.LOW: "🔵",
        }

        for severity in severity_order:
            findings = by_severity.get(severity, [])
            if not findings:
                continue

            icon = severity_icons[severity]
            lines.append(f"\n### {icon} {severity.value.upper()} ({len(findings)})\n")

            for finding in findings:
                lines.append(f"- **Line {finding.line}** [{finding.rule_id}]: {finding.message}")
                if finding.suggestion:
                    lines.append(f"  - *Suggestion:* {finding.suggestion}")
                if finding.code_snippet:
                    lines.append(f"  ```\n  {finding.code_snippet}\n  ```")

        return "\n".join(lines)


@dataclass
class AnalysisResult:
    """Complete analysis result for a repository."""
    repository: str
    total_findings: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    file_groups: list[FileGroup]
    analysis_date: str

    def to_summary_table(self) -> str:
        """Generate markdown summary table."""
        lines = [
            "| File | Critical | High | Medium | Low | Total |",
            "|------|----------|------|--------|-----|-------|",
        ]
        for group in self.file_groups:
            lines.append(
                f"| {group.path} | {group.critical_count} | {group.high_count} | "
                f"{group.medium_count} | {group.low_count} | {group.total_count} |"
            )
        return "\n".join(lines)


class FindingAggregator:
    """Aggregates findings by file/module location (flow-based grouping)."""

    def __init__(
        self,
        severity_filter: str = "medium",
        max_issues: int = 50,
        module_threshold: int = 5,
    ):
        """Initialize the aggregator.

        Args:
            severity_filter: Minimum severity to include ("all", "critical", "high", "medium")
            max_issues: Maximum number of file groups (subtasks) to create
            module_threshold: If a directory has >= this many files with issues,
                              group them into a single module ticket
        """
        self.severity_filter = severity_filter
        self.max_issues = max_issues
        self.module_threshold = module_threshold

    def filter_by_severity(self, findings: list[Finding]) -> list[Finding]:
        """Filter findings based on severity threshold."""
        if self.severity_filter == "all":
            return findings

        severity_order = {
            FindingSeverity.CRITICAL: 0,
            FindingSeverity.HIGH: 1,
            FindingSeverity.MEDIUM: 2,
            FindingSeverity.LOW: 3,
        }

        filter_thresholds = {
            "critical": 0,
            "high": 1,
            "medium": 2,
        }

        threshold = filter_thresholds.get(self.severity_filter, 2)
        return [f for f in findings if severity_order[f.severity] <= threshold]

    def aggregate(self, findings: list[Finding], repository: str) -> AnalysisResult:
        """Aggregate findings into file groups.

        Strategy: Group by file/module (flow-based)
        - All issues in a single file → 1 ticket
        - If a directory has many files with issues → 1 module ticket
        """
        from datetime import datetime

        # Filter by severity first
        filtered = self.filter_by_severity(findings)

        # Group by file path
        by_file: dict[str, list[Finding]] = defaultdict(list)
        for finding in filtered:
            by_file[finding.file_path].append(finding)

        # Check for modules that should be grouped
        by_dir: dict[str, list[str]] = defaultdict(list)
        for file_path in by_file.keys():
            parent = str(Path(file_path).parent)
            by_dir[parent].append(file_path)

        # Build file groups, potentially merging into module groups
        file_groups: list[FileGroup] = []
        processed_files: set[str] = set()

        # First pass: identify modules to group
        for dir_path, files in by_dir.items():
            if len(files) >= self.module_threshold and dir_path != ".":
                # Create module group
                module_findings: list[Finding] = []
                for file_path in files:
                    module_findings.extend(by_file[file_path])
                    processed_files.add(file_path)

                file_groups.append(FileGroup(
                    path=f"{dir_path}/",
                    findings=sorted(module_findings, key=lambda f: (f.file_path, f.line)),
                    is_module=True,
                ))

        # Second pass: individual files not in modules
        for file_path, file_findings in by_file.items():
            if file_path not in processed_files:
                file_groups.append(FileGroup(
                    path=file_path,
                    findings=sorted(file_findings, key=lambda f: f.line),
                    is_module=False,
                ))

        # Sort by total issues (descending) to prioritize high-impact files
        file_groups.sort(key=lambda g: g.total_count, reverse=True)

        # Apply max_issues limit
        file_groups = file_groups[:self.max_issues]

        # Calculate totals
        all_findings = [f for g in file_groups for f in g.findings]

        return AnalysisResult(
            repository=repository,
            total_findings=len(all_findings),
            critical_count=sum(1 for f in all_findings if f.severity == FindingSeverity.CRITICAL),
            high_count=sum(1 for f in all_findings if f.severity == FindingSeverity.HIGH),
            medium_count=sum(1 for f in all_findings if f.severity == FindingSeverity.MEDIUM),
            low_count=sum(1 for f in all_findings if f.severity == FindingSeverity.LOW),
            file_groups=file_groups,
            analysis_date=datetime.utcnow().strftime("%Y-%m-%d"),
        )

    def to_jira_epic_data(
        self,
        result: AnalysisResult,
        jira_project: str,
    ) -> dict:
        """Convert analysis result to JIRA epic with subtasks format.

        Returns data structure compatible with jira_create_epic_with_subtasks MCP tool.
        """
        epic_summary = f"Repository Analysis: {result.repository} - {result.analysis_date}"
        epic_description = f"""## Repository Analysis Report

**Repository:** {result.repository}
**Analysis Date:** {result.analysis_date}
**Total Findings:** {result.total_findings} across {len(result.file_groups)} files/modules

### Summary by Severity
- 🔴 Critical: {result.critical_count}
- 🟠 High: {result.high_count}
- 🟡 Medium: {result.medium_count}
- 🔵 Low: {result.low_count}

### Files with Issues
{result.to_summary_table()}
"""

        subtasks = []
        for i, group in enumerate(result.file_groups, 1):
            subtask_summary = f"{group.path} - {group.total_count} issues ({group.severity_summary})"
            subtasks.append({
                "summary": subtask_summary[:250],  # JIRA summary limit
                "description": group.to_jira_description(),
                "issue_type": "Sub-task",
            })

        return {
            "project_key": jira_project,
            "epic_summary": epic_summary,
            "epic_description": epic_description,
            "subtasks": subtasks,
        }
