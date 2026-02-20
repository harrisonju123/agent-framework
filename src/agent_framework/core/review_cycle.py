"""QA Review & Fix Cycle Management.

Manages the QA → Engineer review feedback loop, including:
- Parsing QA review outcomes
- Extracting structured findings
- Building review and fix tasks
- Escalating to architect after too many cycles
"""

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .task import Task, TaskStatus, TaskType
from ..utils.type_helpers import strip_chain_prefixes

# Cap review cycles to prevent infinite QA ↔ Engineer loops
MAX_REVIEW_CYCLES = 3

REVIEW_OUTCOME_PATTERNS = {
    "request_changes": [r'\bREQUEST_CHANGES\b', r'\bCHANGES REQUESTED\b'],
    "critical_issues": [r'\bCRITICAL\b.*?:', r'severity:\s*CRITICAL'],
    "major_issues": [r'\bMAJOR\b.*?:', r'\bHIGH\b.*?:'],
    "test_failures": [r'tests?\s+fail', r'[1-9]\d*\s+failed'],
    "approve": [r'\bAPPROVE[D]?\b', r'\bLGTM\b'],
}

# Severity patterns matched case-sensitively — uppercase tags only, avoids prose false positives
_CASE_SENSITIVE_KEYS = frozenset({"critical_issues", "major_issues"})

# Line-anchored severity tags for default-deny detection
_SEVERITY_TAG_RE = re.compile(r'^(CRITICAL|HIGH|MAJOR|MEDIUM|MINOR|LOW|SUGGESTION)\b', re.MULTILINE)


@dataclass
class QAFinding:
    """Structured QA finding with file location, severity, and details."""
    file: str
    line_number: Optional[int]
    severity: str  # CRITICAL|HIGH|MAJOR|MEDIUM|LOW|MINOR|SUGGESTION
    description: str
    suggested_fix: Optional[str]
    category: str  # security|performance|correctness|readability|testing|best_practices


@dataclass
class ReviewOutcome:
    """Parsed result of a QA review."""
    approved: bool
    has_critical_issues: bool
    has_test_failures: bool
    has_change_requests: bool
    findings_summary: str
    has_major_issues: bool = False
    structured_findings: List['QAFinding'] = None

    def __post_init__(self):
        if self.structured_findings is None:
            self.structured_findings = []

    @property
    def needs_fix(self) -> bool:
        return self.has_critical_issues or self.has_test_failures or self.has_change_requests or self.has_major_issues


class ReviewCycleManager:
    """Manages QA review and fix cycles.

    Handles:
    - Automatic review task creation when PRs are created
    - Parsing QA review outcomes
    - Extracting structured findings
    - Building fix tasks with checklists
    - Escalating to architect after too many cycles
    """

    def __init__(
        self,
        config,
        queue,
        logger,
        agent_definition,
        session_logger,
        activity_manager,
        feedback_bus=None,
    ):
        """Initialize ReviewCycleManager.

        Args:
            config: AgentConfig instance
            queue: FileQueue for task management
            logger: Logger instance
            agent_definition: AgentDefinition for agent metadata
            session_logger: SessionLogger for structured logging
            activity_manager: ActivityManager for status tracking
            feedback_bus: Optional FeedbackBus for cross-feature learning
        """
        self.config = config
        self.queue = queue
        self.logger = logger
        self.agent_definition = agent_definition
        self.session_logger = session_logger
        self.activity_manager = activity_manager
        self.feedback_bus = feedback_bus

    def extract_pr_info_from_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """
        Extract PR information from LLM response content.

        Parses the response for GitHub PR URLs created via MCP tools.
        Returns dict with pr_url, pr_number, owner, repo if found.
        """
        # Pattern for GitHub PR URLs: https://github.com/{owner}/{repo}/pull/{number}
        pr_pattern = r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)'

        match = re.search(pr_pattern, response_content)
        if match:
            owner, repo, pr_number = match.groups()
            return {
                "pr_url": match.group(0),
                "pr_number": int(pr_number),
                "owner": owner,
                "repo": repo,
                "github_repo": f"{owner}/{repo}",
            }
        return None

    def get_pr_info(self, task: Task, response) -> Optional[dict]:
        """Extract PR information from response or task context."""
        # Try extracting from response content
        pr_info = self.extract_pr_info_from_response(response.content)
        if pr_info:
            return pr_info

        # Check task context
        pr_url = task.context.get("pr_url")
        if not pr_url:
            return None

        match = re.search(r'https://github\.com/([^/]+)/([^/]+)/pull/(\d+)', pr_url)
        if not match:
            return None

        owner, repo, pr_number = match.groups()
        return {
            "pr_url": pr_url,
            "pr_number": int(pr_number),
            "owner": owner,
            "repo": repo,
            "github_repo": f"{owner}/{repo}",
        }

    def build_review_task(self, task: Task, pr_info: dict) -> Task:
        """Build code review task for a PR."""
        jira_key = task.context.get("jira_key", "UNKNOWN")
        pr_number = pr_info["pr_number"]
        # Stable root ID keeps review task IDs flat across chain hops
        root_task_id = task.root_id
        clean_title = strip_chain_prefixes(task.title)[:50]

        # ID is stable per (root task, PR number). If a PR is closed and a new
        # one opened for the same root task with the same number (rare GitHub
        # reuse), the second review would be silently skipped by the dedup guard
        # in queue_code_review_if_needed — acceptable given how unlikely this is.
        return Task(
            id=f"review-{root_task_id}-{pr_number}",
            type=TaskType.REVIEW,
            status=TaskStatus.PENDING,
            priority=task.priority,
            created_by=self.config.id,
            assigned_to="qa",
            created_at=datetime.now(timezone.utc),
            title=f"Review PR #{pr_number} - [{jira_key}] {clean_title}",
            description=f"""Automated code review request for PR #{pr_number}.

## PR Information
- **PR URL**: {pr_info['pr_url']}
- **Repository**: {pr_info['github_repo']}
- **JIRA Ticket**: {jira_key}
- **Created by**: {self.config.id} agent

## Review Instructions
1. Fetch PR details and diff using `github_get_pr` and `github_get_pr_diff` MCP tools
2. Check CI status using `github_get_check_runs` with the PR branch
3. Review the diff against standard review criteria:
   - Correctness: Logic errors, edge cases, error handling
   - Security: Vulnerabilities, input validation, secrets
   - Performance: Inefficient patterns, N+1 queries
   - Readability: Code clarity, naming, documentation
   - Best Practices: Language conventions, test coverage
4. If CI checks are failing, include CI failures in your findings as CRITICAL
5. Post review comments on PR
6. Update JIRA with review summary
7. Transition JIRA status if appropriate (Approved/Changes Requested)
""",
            context={
                "jira_key": jira_key,
                "jira_url": task.context.get("jira_url"),
                "pr_number": pr_number,
                "pr_url": pr_info["pr_url"],
                "github_repo": pr_info["github_repo"],
                "branch_name": task.context.get("branch_name"),
                "workflow": task.context.get("workflow", "default"),
                "review_mode": True,
                "source_task_id": task.id,
                "source_agent": self.config.id,
                "implementation_branch": task.context.get("implementation_branch"),
                # Carry review cycle count so QA → Engineer loop is capped
                "_review_cycle_count": task.context.get("_review_cycle_count", 0),
            },
        )

    def purge_orphaned_review_tasks(self) -> None:
        """Remove REVIEW/FIX tasks for PRs that already have an ESCALATION.

        On restart, stale review-chain tasks from before the cycle-count guard
        may still be queued.  If an escalation already exists for a PR, every
        REVIEW/FIX task for that PR is orphaned — processing them would restart
        a parallel chain the architect already owns.
        """
        queue_dir = self.queue.queue_dir
        completed_dir = self.queue.completed_dir

        # Step 1: collect PR URLs that have been escalated
        escalated_prs: set[str] = set()
        for search_dir in (queue_dir / "architect", completed_dir):
            if not search_dir.is_dir():
                continue
            for f in search_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if data.get("type") == TaskType.ESCALATION.value:
                    pr_url = data.get("context", {}).get("pr_url")
                    if pr_url:
                        escalated_prs.add(pr_url)

        if not escalated_prs:
            return

        # Step 2: remove REVIEW/FIX tasks whose pr_url matches an escalated PR
        purged = 0
        for sub in queue_dir.iterdir():
            if not sub.is_dir():
                continue
            for f in sub.glob("*.json"):
                try:
                    data = json.loads(f.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                task_type = data.get("type")
                if task_type not in (TaskType.REVIEW.value, TaskType.FIX.value):
                    continue
                pr_url = data.get("context", {}).get("pr_url")
                if pr_url in escalated_prs:
                    f.unlink()
                    purged += 1

        if purged:
            self.logger.info(
                f"Purged {purged} orphaned review-chain task(s) for "
                f"{len(escalated_prs)} escalated PR(s)"
            )

    def queue_code_review_if_needed(self, task: Task, response) -> None:
        """
        Automatically queue a code review task if a PR was created.

        This ensures every PR gets reviewed by the QA agent,
        regardless of whether the creating agent remembered to queue the review.
        """
        # Skip if this agent IS the QA (avoid infinite loop); use base_id for replica support
        if self.config.base_id == "qa":
            return

        # Skip if task type is already a review or escalation
        if task.type in (TaskType.REVIEW, TaskType.ESCALATION):
            return

        # Chain tasks are routed by the workflow DAG which already includes
        # the QA step — creating a separate review task would duplicate it.
        if task.context.get("chain_step"):
            self.logger.debug(f"Skipping review for chain task {task.id}: DAG handles QA routing")
            return

        # Skip if this task already hit the escalation threshold — the review
        # loop escalated to the architect, so spawning another review would
        # restart a parallel chain (fork bomb).
        if task.context.get("_review_cycle_count", 0) >= MAX_REVIEW_CYCLES:
            self.logger.debug(
                f"Skipping review for {task.id}: cycle count "
                f"{task.context['_review_cycle_count']} >= {MAX_REVIEW_CYCLES} (already escalated)"
            )
            return

        # Get PR information
        pr_info = self.get_pr_info(task, response)
        if not pr_info:
            self.logger.debug(f"No PR found in task {task.id} - skipping code review queue")
            return

        # Build and queue review task
        review_task = self.build_review_task(task, pr_info)
        pr_number = pr_info["pr_number"]

        # Deduplicate: skip if this exact review task is already queued
        review_path = self.queue.queue_dir / "qa" / f"{review_task.id}.json"
        if review_path.exists():
            self.logger.debug(f"Review task {review_task.id} already queued, skipping")
            return

        try:
            self.queue.push(review_task, "qa")
            self.logger.info(
                f"🔍 Queued code review for PR #{pr_number} ({pr_info['github_repo']}) -> qa"
            )

            # Store PR URL in original task context for tracking
            task.context["pr_url"] = pr_info["pr_url"]
            task.context["pr_number"] = pr_number
            task.context["code_review_task_id"] = review_task.id

        except Exception as e:
            self.logger.error(f"Failed to queue code review for PR #{pr_number}: {e}")

    def queue_review_fix_if_needed(self, task: Task, response, sync_jira_status_callback) -> None:
        """Deterministically queue a fix task to engineer when QA finds issues.

        Mirrors queue_code_review_if_needed: that method hard-codes the
        Engineer → QA direction; this method hard-codes QA → Engineer.

        Args:
            task: Review task
            response: QA response
            sync_jira_status_callback: Callback to sync JIRA status (from Agent)
        """
        if self.config.base_id != "qa":
            return
        if task.type != TaskType.REVIEW:
            return

        outcome = self.parse_review_outcome(response.content)
        if outcome.approved and not outcome.needs_fix:
            sync_jira_status_callback(task, "Approved", comment=f"QA approved by {self.config.id}")
            return
        # Ambiguous (neither approved nor flagged) → treat as needs_fix
        if not outcome.needs_fix and not outcome.approved:
            self.logger.info("Ambiguous QA verdict (no APPROVE/issues) — treating as needs_fix")
            outcome = replace(
                outcome,
                has_major_issues=True,
                findings_summary=outcome.findings_summary or response.content[:500],
            )

        cycle_count = task.context.get("_review_cycle_count", 0) + 1

        if cycle_count > MAX_REVIEW_CYCLES:
            self.escalate_review_to_architect(task, outcome, cycle_count)
            sync_jira_status_callback(
                task, "Changes Requested",
                comment=f"Escalated to architect after {cycle_count} review cycles",
            )
            return

        # Store QA findings in feedback bus for cross-task learning
        if self.feedback_bus and outcome.structured_findings:
            repo_slug = task.context.get("github_repo")
            if repo_slug:
                try:
                    self.feedback_bus.on_qa_findings(
                        task=task,
                        findings=outcome.structured_findings,
                        repo_slug=repo_slug,
                    )
                except Exception as e:
                    self.logger.debug(f"Feedback bus QA findings storage failed (non-fatal): {e}")

        fix_task = self.build_review_fix_task(task, outcome, cycle_count)

        # Deduplicate: skip if fix task file already exists in engineer queue
        fix_path = self.queue.queue_dir / "engineer" / f"{fix_task.id}.json"
        if fix_path.exists():
            self.logger.debug(f"Review fix task {fix_task.id} already queued, skipping")
            return

        try:
            self.queue.push(fix_task, "engineer")
            self.logger.info(
                f"🔧 Queued review fix (cycle {cycle_count}/{MAX_REVIEW_CYCLES}) -> engineer"
            )
            sync_jira_status_callback(
                task, "Changes Requested",
                comment=f"Review cycle {cycle_count}: {outcome.findings_summary[:200]}",
            )
        except Exception as e:
            self.logger.error(f"Failed to queue review fix task: {e}")

    def parse_review_outcome(self, content: str) -> ReviewOutcome:
        """Parse QA response for review verdict using regex patterns."""
        if not content:
            return ReviewOutcome(
                approved=False, has_critical_issues=False,
                has_test_failures=False, has_change_requests=False,
                findings_summary="",
            )

        _NEGATIONS = ('no ', 'zero ', '0 ', 'without ', 'not ')

        def _matches(key: str) -> bool:
            flags = 0 if key in _CASE_SENSITIVE_KEYS else re.IGNORECASE
            for p in REVIEW_OUTCOME_PATTERNS[key]:
                m = re.search(p, content, flags)
                if m:
                    prefix = content[max(0, m.start() - 20):m.start()].lower()
                    if any(neg in prefix for neg in _NEGATIONS):
                        continue
                    return True
            return False

        approved = _matches("approve")
        has_critical = _matches("critical_issues")
        has_major = _matches("major_issues")
        has_test_fail = _matches("test_failures")
        has_changes = _matches("request_changes")

        # CRITICAL/MAJOR/HIGH override explicit APPROVE
        if has_critical or has_major or has_test_fail or has_changes:
            approved = False

        # Default-deny: severity-tagged findings without explicit APPROVE → needs fix.
        # Only exact APPROVE/LGTM keywords count as approval.
        if not approved and not (has_critical or has_major or has_test_fail or has_changes):
            if _SEVERITY_TAG_RE.search(content):
                has_major = True

        findings_summary, structured_findings = self.extract_review_findings(content)

        return ReviewOutcome(
            approved=approved,
            has_critical_issues=has_critical,
            has_test_failures=has_test_fail,
            has_change_requests=has_changes,
            has_major_issues=has_major,
            findings_summary=findings_summary,
            structured_findings=structured_findings,
        )

    def extract_review_findings(self, content: str) -> tuple[str, List[QAFinding]]:
        """Extract findings from QA review output.

        Returns:
            tuple: (findings_summary: str, structured_findings: List[QAFinding])
            Tries new structured parser first, falls back to legacy regex extraction.
        """
        # Try new structured parsing first (handles code fences and inline JSON)
        structured_findings = self.parse_structured_findings(content)

        if not structured_findings:
            structured_findings = []

        # Extract severity-tagged lines for text summary
        findings_text = []
        for line in content.splitlines():
            stripped = line.strip()
            # Case-sensitive: we want structured output tags (CRITICAL, HIGH, …),
            # not prose that happens to contain the word.
            if re.match(r'^(CRITICAL|HIGH|MAJOR|MEDIUM|MINOR|LOW|SUGGESTION)\b', stripped):
                findings_text.append(stripped)

        # If we have structured findings, build summary from them
        if structured_findings:
            summary_lines = []
            for finding in structured_findings:
                location = f"{finding.file}"
                if finding.line_number:
                    location += f":{finding.line_number}"
                summary_lines.append(f"{finding.severity}: {finding.description} ({location})")
            findings_summary = "\n".join(summary_lines)
        elif findings_text:
            findings_summary = "\n".join(findings_text)
        else:
            # Fall back to first 500 chars if no tagged lines found
            findings_summary = content[:500]

        return findings_summary, structured_findings

    def parse_structured_findings(self, content: str) -> Optional[List[QAFinding]]:
        """Extract structured JSON findings from QA response.

        Looks for JSON blocks in code fences (```json...```) or inline.
        Supports both array format and object format with 'findings' key.
        Handles multi-object code fences (e.g. verdict + findings in one block)
        and findings split across multiple code fences.
        Falls back to None if no structured findings found.
        """
        # Iterate ALL code fence blocks — re.search only matched the first one
        for fence_match in re.finditer(r'```json\s*([\s\S]*?)\s*```', content, re.DOTALL):
            block = fence_match.group(1).strip()
            if not block:
                continue
            findings = self._try_parse_findings_json(block)
            if findings:
                return findings

        # Inline fallback: scan for JSON objects/arrays using raw_decode
        findings = self._try_parse_findings_json(content)
        if findings:
            return findings

        return None

    def _make_qa_finding(self, item: dict) -> Optional[QAFinding]:
        """Build a QAFinding from a dict, returning None if item is not a dict."""
        if not isinstance(item, dict):
            return None
        return QAFinding(
            file=item.get('file', ''),
            line_number=item.get('line_number') or item.get('line'),
            severity=item.get('severity', 'UNKNOWN'),
            description=item.get('description', ''),
            suggested_fix=item.get('suggested_fix'),
            category=item.get('category', 'general'),
        )

    def _extract_findings_from_data(self, data) -> Optional[List[QAFinding]]:
        """Convert a parsed JSON value into a list of QAFinding if it contains findings."""
        if isinstance(data, dict) and 'findings' in data:
            items = data['findings']
        elif isinstance(data, list):
            items = data
        else:
            return None

        parsed = [f for f in (self._make_qa_finding(i) for i in items) if f is not None]
        return parsed if parsed else None

    def _try_parse_findings_json(self, text: str) -> Optional[List[QAFinding]]:
        """Try to extract findings from a text block that may contain one or more JSON values.

        Fast path: json.loads for single-object blocks.
        Slow path: json.JSONDecoder().raw_decode() loop for multi-object blocks
        (e.g. {"verdict":"pass"} followed by {"findings":[...]}).
        """
        # Fast path — single JSON value covers the whole block
        try:
            data = json.loads(text)
            result = self._extract_findings_from_data(data)
            if result:
                return result
        except (json.JSONDecodeError, TypeError):
            pass

        # Slow path — walk through concatenated JSON values
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(text):
            # Skip whitespace and non-JSON characters
            if text[idx] not in ('{', '['):
                idx += 1
                continue
            try:
                data, end_idx = decoder.raw_decode(text, idx)
                result = self._extract_findings_from_data(data)
                if result:
                    return result
                idx = end_idx
            except json.JSONDecodeError:
                idx += 1

        return None

    def format_findings_checklist(self, findings: List[QAFinding]) -> str:
        """Format structured findings as numbered checklist."""
        lines = []

        for i, finding in enumerate(findings, 1):
            # Format location
            location = ""
            if finding.file and finding.line_number:
                location = f" ({finding.file}:{finding.line_number})"
            elif finding.file:
                location = f" ({finding.file})"

            # Format severity with emoji
            severity_emoji = {
                "CRITICAL": "🔴",
                "HIGH": "🟠",
                "MAJOR": "🟡",
                "MEDIUM": "🔵",
                "MINOR": "⚪",
                "LOW": "⚪",
                "SUGGESTION": "💡",
            }
            emoji = severity_emoji.get(finding.severity, "")

            lines.append(f"### {i}. {emoji} {finding.severity}: {finding.category.title()}{location}")
            lines.append(f"**Issue**: {finding.description}")

            if finding.suggested_fix:
                lines.append(f"**Suggested Fix**: {finding.suggested_fix}")

            lines.append("")  # blank line

        return "\n".join(lines)

    def build_review_fix_task(self, task: Task, outcome: ReviewOutcome, cycle_count: int) -> Task:
        """Build fix task with structured checklist."""
        jira_key = task.context.get("jira_key", "UNKNOWN")
        pr_url = task.context.get("pr_url", "")
        pr_number = task.context.get("pr_number", "")

        # Check if we have structured findings
        has_structured_findings = outcome.structured_findings and len(outcome.structured_findings) > 0

        # Build description with numbered checklist or legacy format
        if has_structured_findings:
            # Generate checklist using existing formatter
            checklist = self.format_findings_checklist(outcome.structured_findings)
            total_count = len(outcome.structured_findings)

            description = f"""QA review found {total_count} issue(s) that need fixing.

## Summary
{outcome.findings_summary}

## Issues to Address

{checklist}

## Instructions
1. Review each finding above
2. Fix the issues in the specified files/lines
3. Run tests to verify fixes: `pytest tests/`
4. Run linting: `pylint src/` or appropriate linter
5. Commit and push your changes
6. The review will be automatically re-queued

## Context
- **PR**: {pr_url}
- **JIRA**: {jira_key}
- **Review Cycle**: {cycle_count} of {MAX_REVIEW_CYCLES}
"""
        else:
            # Legacy format (backward compatible)
            description = f"""QA review found issues that need fixing.

## Review Findings
{outcome.findings_summary}

## Instructions
1. Review the findings above
2. Fix the identified issues
3. Run tests to verify fixes
4. Commit and push your changes

## Context
- **PR**: {pr_url}
- **JIRA**: {jira_key}
- **Review Cycle**: {cycle_count} of {MAX_REVIEW_CYCLES}
"""

        # Build context (strip review_* keys, preserve essential context)
        fix_context = {
            k: v for k, v in task.context.items()
            if not k.startswith("review_")
        }
        fix_context["_review_cycle_count"] = cycle_count
        fix_context["pr_url"] = pr_url
        fix_context["pr_number"] = pr_number
        fix_context["github_repo"] = task.context.get("github_repo")
        fix_context["jira_key"] = jira_key
        fix_context["workflow"] = task.context.get("workflow", "full")

        # Preserve engineer's branch so fix cycle reuses the same worktree
        if task.context.get("implementation_branch"):
            fix_context["worktree_branch"] = task.context["implementation_branch"]

        # Store structured findings in context for programmatic access
        if has_structured_findings:
            # Serialize findings to dict for context storage
            findings_dicts = []
            for finding in outcome.structured_findings:
                findings_dicts.append({
                    "file": finding.file,
                    "line_number": finding.line_number,
                    "severity": finding.severity,
                    "category": finding.category,
                    "description": finding.description,
                    "suggested_fix": finding.suggested_fix,
                })

            fix_context["structured_findings"] = {
                "findings": findings_dicts,
                "summary": outcome.findings_summary,
                "total_count": len(outcome.structured_findings),
                "critical_count": sum(1 for f in outcome.structured_findings if f.severity == "CRITICAL"),
                "high_count": sum(1 for f in outcome.structured_findings if f.severity == "HIGH"),
                "major_count": sum(1 for f in outcome.structured_findings if f.severity == "MAJOR"),
            }

        # Build acceptance criteria
        if has_structured_findings:
            total_count = len(outcome.structured_findings)
            acceptance_criteria = [
                f"All {total_count} issues addressed",
                "Tests pass",
                "Linting passes",
            ]
        else:
            acceptance_criteria = [
                "All identified issues addressed",
                "Tests pass",
            ]

        root_task_id = task.root_id
        return Task(
            id=f"review-fix-{root_task_id[:12]}-c{cycle_count}",
            type=TaskType.FIX,
            status=TaskStatus.PENDING,
            priority=task.priority,
            created_by=self.config.id,
            assigned_to="engineer",
            created_at=datetime.now(timezone.utc),
            title=f"Fix review issues (cycle {cycle_count}) - [{jira_key}]",
            description=description,
            context=fix_context,
            acceptance_criteria=acceptance_criteria,
        )

    def escalate_review_to_architect(self, task: Task, outcome: ReviewOutcome, cycle_count: int) -> None:
        """Escalate to architect after too many failed review cycles."""
        jira_key = task.context.get("jira_key", "UNKNOWN")

        root_task_id = task.root_id
        escalation_task = Task(
            id=f"review-escalation-{root_task_id[:20]}",
            type=TaskType.ESCALATION,
            status=TaskStatus.PENDING,
            priority=max(1, task.priority - 1),  # Lower number = higher priority
            created_by=self.config.id,
            assigned_to="architect",
            created_at=datetime.now(timezone.utc),
            title=f"Review escalation ({cycle_count} cycles) - [{jira_key}]",
            description=f"""QA and Engineer failed to resolve review issues after {cycle_count} cycles.

## Last Review Findings
{outcome.findings_summary}

## Action Required
- Replan the implementation approach
- Consider breaking the task into smaller pieces
- Provide more detailed architectural guidance
""",
            context={
                **task.context,
                "source_task_id": task.id,
                "source_agent": self.config.id,
                "_review_cycle_count": cycle_count,
                "escalation_reason": f"Review loop exceeded {MAX_REVIEW_CYCLES} cycles",
                "_root_task_id": root_task_id,
                "_global_cycle_count": task.context.get("_global_cycle_count", 0),
                "_chain_depth": task.context.get("_chain_depth", 0),
                # Carry structured findings so architect can see exact QA issues
                **({"structured_findings": task.context["structured_findings"]}
                   if "structured_findings" in task.context else {}),
            },
        )

        try:
            self.queue.push(escalation_task, "architect")
            self.logger.warning(
                f"⚠️  Review escalated to architect after {cycle_count} cycles"
            )
        except Exception as e:
            self.logger.error(f"Failed to escalate review to architect: {e}")
