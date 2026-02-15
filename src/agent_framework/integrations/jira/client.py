"""JIRA client for ticket management."""

import time
from datetime import datetime, timezone
from typing import List, Optional

from jira import JIRA
from jira.resources import Issue

from ...core.task import Task, TaskStatus, TaskType
from ...core.config import JIRAConfig


class JIRAClient:
    """JIRA API client for ticket operations."""

    def __init__(self, config: JIRAConfig):
        self.config = config
        self.jira = JIRA(
            server=config.server,
            basic_auth=(config.email, config.api_token),
        )
        # Cache index of the JQL pattern that worked for get_epic_issues
        self._working_jql_index: Optional[int] = None

    def pull_unassigned_tickets(self, max_results: int = 10) -> List[Issue]:
        """Pull unassigned tickets from backlog using JQL filter."""
        issues = self.jira.search_issues(
            jql_str=self.config.backlog_filter,
            maxResults=max_results,
        )
        return list(issues)

    def create_ticket(
        self,
        summary: str,
        description: str,
        issue_type: str = "Story",
        labels: Optional[List[str]] = None,
    ) -> Issue:
        """Create a new JIRA ticket."""
        issue_dict = {
            "project": {"key": self.config.project},
            "summary": summary,
            "description": description,
            "issuetype": {"name": issue_type},
        }

        if labels:
            issue_dict["labels"] = labels

        issue = self.jira.create_issue(fields=issue_dict)
        return issue

    def transition_ticket(self, ticket_id: str, status: str) -> None:
        """Transition ticket to new status."""
        # Get transition ID from config
        transition_id = self.config.transitions.get(status.lower().replace(" ", "_"))
        if not transition_id:
            raise ValueError(f"Unknown transition: {status}")

        issue = self.jira.issue(ticket_id)
        self.jira.transition_issue(issue, transition_id)

    def add_comment(self, ticket_id: str, comment: str) -> None:
        """Add a comment to a ticket."""
        issue = self.jira.issue(ticket_id)
        self.jira.add_comment(issue, comment)

    def update_custom_field(self, ticket_id: str, field_id: str, value: str) -> None:
        """Update a custom field (e.g., PR link)."""
        issue = self.jira.issue(ticket_id)
        issue.update(fields={field_id: value})

    def create_epic(
        self,
        summary: str,
        description: str,
        project: str,
        labels: Optional[List[str]] = None,
    ) -> Issue:
        """Create a JIRA epic."""
        issue_dict = {
            "project": {"key": project},
            "summary": summary,
            "description": description,
            "issuetype": {"name": "Epic"},
        }
        if labels:
            issue_dict["labels"] = labels

        return self.jira.create_issue(fields=issue_dict)

    def create_subtask(
        self,
        parent_key: str,
        summary: str,
        description: str,
        project: str,
        labels: Optional[List[str]] = None,
    ) -> Issue:
        """Create a subtask under a parent issue."""
        issue_dict = {
            "project": {"key": project},
            "summary": summary,
            "description": description,
            "issuetype": {"name": "Sub-task"},
            "parent": {"key": parent_key},
        }
        if labels:
            issue_dict["labels"] = labels

        return self.jira.create_issue(fields=issue_dict)

    def create_epic_with_subtasks(
        self,
        epic_summary: str,
        epic_description: str,
        subtasks: List[dict],
        project: str,
    ) -> dict:
        """Create epic with subtasks in one operation.

        Args:
            epic_summary: Epic title
            epic_description: Epic description
            subtasks: List of dicts with 'summary', 'description', 'agent', 'depends_on'
            project: JIRA project key

        Returns:
            Dict mapping task IDs to JIRA issues
        """
        # Create epic
        epic = self.create_epic(epic_summary, epic_description, project)

        # Create subtasks
        issues = {"epic": epic}
        for i, subtask in enumerate(subtasks):
            issue = self.create_subtask(
                parent_key=epic.key,
                summary=subtask["summary"],
                description=subtask["description"],
                project=project,
                labels=[f"agent:{subtask['agent']}"],
            )
            issues[f"subtask_{i}"] = issue

        return issues

    def get_epic_issues(self, epic_key: str) -> List[Issue]:
        """Get all issues belonging to an epic.

        Args:
            epic_key: JIRA key of the epic (e.g., PROJ-123)

        Returns:
            List of issues linked to the epic
        """
        # Try different JQL queries since epic link field varies by JIRA setup
        jql_queries = [
            # Standard epic link (JIRA Software)
            f'"Epic Link" = {epic_key}',
            # Parent link (JIRA next-gen / Team-managed)
            f'parent = {epic_key}',
            # Issues in epic (some JIRA configurations)
            f'"Parent Link" = {epic_key}',
        ]

        # Try cached pattern first to avoid unnecessary API calls
        if self._working_jql_index is not None:
            try:
                jql = jql_queries[self._working_jql_index]
                issues = self.jira.search_issues(
                    jql_str=jql,
                    maxResults=100,
                    fields="summary,description,issuetype,status,created,priority",
                )
                if issues:
                    return list(issues)
            except Exception:
                self._working_jql_index = None

        for idx, jql in enumerate(jql_queries):
            try:
                issues = self.jira.search_issues(
                    jql_str=jql,
                    maxResults=100,
                    fields="summary,description,issuetype,status,created,priority",
                )
                if issues:
                    self._working_jql_index = idx
                    return list(issues)
            except Exception:
                continue

        return []

    def get_epic_with_subtasks(self, epic_key: str) -> dict:
        """Get epic and all its linked issues.

        Args:
            epic_key: JIRA key of the epic

        Returns:
            Dict with 'epic' (Issue) and 'issues' (List[Issue])
        """
        epic = self.jira.issue(epic_key)
        issues = self.get_epic_issues(epic_key)

        return {
            "epic": epic,
            "issues": issues,
        }

    def issue_to_task(self, issue: Issue, assigned_to: str) -> Task:
        """Convert JIRA issue to internal Task."""
        task_id = f"jira-{issue.key}-{int(time.time())}"

        # Determine task type from issue type
        task_type = TaskType.IMPLEMENTATION
        if hasattr(issue.fields, 'issuetype'):
            issue_type = issue.fields.issuetype.name.lower()
            if "bug" in issue_type:
                task_type = TaskType.FIX
            elif "test" in issue_type:
                task_type = TaskType.TESTING

        # Extract acceptance criteria if available
        acceptance_criteria = []
        description = issue.fields.description or ""

        # Parse JIRA ISO datetime string to datetime object
        created_str = str(issue.fields.created)
        try:
            # Handle JIRA datetime format (ISO 8601 with timezone)
            created_at = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
        except ValueError:
            created_at = datetime.now(timezone.utc)

        task = Task(
            id=task_id,
            type=task_type,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="jira",
            assigned_to=assigned_to,
            created_at=created_at,
            title=issue.fields.summary,
            description=description,
            acceptance_criteria=acceptance_criteria,
            context={
                "jira_key": issue.key,
                "jira_url": f"{self.config.server}/browse/{issue.key}",
            },
        )

        return task
