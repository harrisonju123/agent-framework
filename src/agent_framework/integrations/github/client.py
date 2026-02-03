"""GitHub client for PR management."""

from typing import List, Optional

from github import Github
from github.PullRequest import PullRequest
from github.Repository import Repository

from ...core.config import GitHubConfig


class GitHubClient:
    """GitHub API client for PR operations."""

    def __init__(self, config: GitHubConfig):
        self.config = config
        self.gh = Github(config.token)
        self.repo: Repository = self.gh.get_repo(f"{config.owner}/{config.repo}")

    def create_branch(self, branch_name: str, from_branch: str = "main") -> None:
        """Create a new branch."""
        source_branch = self.repo.get_branch(from_branch)
        self.repo.create_git_ref(
            ref=f"refs/heads/{branch_name}",
            sha=source_branch.commit.sha,
        )

    def create_pull_request(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
        draft: bool = False,
        labels: Optional[List[str]] = None,
    ) -> PullRequest:
        """Create a pull request."""
        pr = self.repo.create_pull(
            title=title,
            body=body,
            head=head_branch,
            base=base_branch,
            draft=draft,
        )

        if labels:
            pr.add_to_labels(*labels)

        return pr

    def get_pr_by_branch(self, branch_name: str) -> Optional[PullRequest]:
        """Get PR for a given branch."""
        pulls = self.repo.get_pulls(state="open", head=f"{self.config.owner}:{branch_name}")
        for pr in pulls:
            return pr
        return None

    def add_pr_comment(self, pr_number: int, comment: str) -> None:
        """Add a comment to a PR."""
        pr = self.repo.get_pull(pr_number)
        pr.create_issue_comment(comment)

    def link_pr_to_jira(
        self,
        pr: PullRequest,
        jira_ticket_id: str,
        jira_url: str,
    ) -> None:
        """Add JIRA ticket reference to PR description."""
        current_body = pr.body or ""

        # Add JIRA link to body
        jira_section = f"\n\n## JIRA Ticket\n[{jira_ticket_id}]({jira_url})\n"

        if jira_section not in current_body:
            new_body = current_body + jira_section
            pr.edit(body=new_body)

    def format_branch_name(self, ticket_id: str, slug: str, branch_type: str = "feature") -> str:
        """Format branch name according to pattern."""
        return self.config.branch_pattern.format(
            type=branch_type,
            ticket_id=ticket_id,
            slug=slug,
        )

    def format_pr_title(self, ticket_id: str, title: str) -> str:
        """Format PR title according to pattern."""
        return self.config.pr_title_pattern.format(
            ticket_id=ticket_id,
            title=title,
        )
