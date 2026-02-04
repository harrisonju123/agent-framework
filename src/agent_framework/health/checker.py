"""Health check module for validating system configuration."""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class CheckStatus(Enum):
    """Health check status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class CheckResult:
    """Result of a health check."""
    name: str
    status: CheckStatus
    message: str
    fix_action: Optional[str] = None
    documentation: Optional[str] = None


class HealthChecker:
    """Validate system configuration and connectivity."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.config_dir = workspace / "config"

    def run_all_checks(self) -> List[CheckResult]:
        """Run comprehensive health checks."""
        return [
            self.check_config_files(),
            self.check_environment_variables(),
            self.check_jira_connectivity(),
            self.check_github_connectivity(),
            self.check_directory_structure(),
            self.check_agent_definitions(),
        ]

    def check_config_files(self) -> CheckResult:
        """Verify all required config files exist."""
        required_files = [
            "config/agent-framework.yaml",
            "config/agents.yaml",
            "config/jira.yaml",
            "config/github.yaml"
        ]

        missing = [f for f in required_files if not (self.workspace / f).exists()]

        if missing:
            return CheckResult(
                name="Config Files",
                status=CheckStatus.FAILED,
                message=f"Missing files: {', '.join(missing)}",
                fix_action="Run setup wizard: agent dashboard and click Setup",
                documentation="README.md#configuration"
            )

        return CheckResult(
            name="Config Files",
            status=CheckStatus.PASSED,
            message="All config files present"
        )

    def check_environment_variables(self) -> CheckResult:
        """Verify required environment variables are set."""
        import os

        required_vars = ["JIRA_SERVER", "JIRA_EMAIL", "JIRA_API_TOKEN", "GITHUB_TOKEN"]
        missing = [var for var in required_vars if not os.getenv(var)]

        if missing:
            return CheckResult(
                name="Environment Variables",
                status=CheckStatus.FAILED,
                message=f"Missing variables: {', '.join(missing)}",
                fix_action="Run setup wizard to configure credentials",
                documentation="README.md#environment-variables"
            )

        return CheckResult(
            name="Environment Variables",
            status=CheckStatus.PASSED,
            message="All environment variables set"
        )

    def check_jira_connectivity(self) -> CheckResult:
        """Test JIRA API connection."""
        try:
            from ..core.config import load_jira_config
            from ..integrations.jira.client import JIRAClient

            jira_config_path = self.config_dir / "jira.yaml"
            if not jira_config_path.exists():
                return CheckResult(
                    name="JIRA Connection",
                    status=CheckStatus.SKIPPED,
                    message="JIRA config not found"
                )

            jira_config = load_jira_config(jira_config_path)
            if not jira_config:
                return CheckResult(
                    name="JIRA Connection",
                    status=CheckStatus.FAILED,
                    message="Invalid JIRA config",
                    fix_action="Run setup wizard to reconfigure JIRA"
                )

            client = JIRAClient(jira_config)
            # Test connection by fetching user info
            _ = client.jira.myself()

            return CheckResult(
                name="JIRA Connection",
                status=CheckStatus.PASSED,
                message=f"Connected to {jira_config.server}"
            )

        except Exception as e:
            error_str = str(e).lower()

            if "401" in error_str or "unauthorized" in error_str:
                return CheckResult(
                    name="JIRA Connection",
                    status=CheckStatus.FAILED,
                    message="Authentication failed - invalid credentials",
                    fix_action="Run setup wizard to update JIRA credentials",
                    documentation="docs/TROUBLESHOOTING.md#jira-auth"
                )
            elif "connection" in error_str or "timeout" in error_str:
                return CheckResult(
                    name="JIRA Connection",
                    status=CheckStatus.FAILED,
                    message="Cannot reach JIRA server",
                    fix_action="Check network connection and JIRA server URL"
                )
            else:
                return CheckResult(
                    name="JIRA Connection",
                    status=CheckStatus.WARNING,
                    message=f"JIRA connection issue: {str(e)[:100]}"
                )

    def check_github_connectivity(self) -> CheckResult:
        """Test GitHub API connection."""
        try:
            import os
            import requests

            token = os.getenv("GITHUB_TOKEN")
            if not token:
                return CheckResult(
                    name="GitHub Connection",
                    status=CheckStatus.FAILED,
                    message="GitHub token not set",
                    fix_action="Run setup wizard to configure GitHub"
                )

            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            response = requests.get("https://api.github.com/user", headers=headers, timeout=10)

            if response.status_code == 200:
                user_data = response.json()
                return CheckResult(
                    name="GitHub Connection",
                    status=CheckStatus.PASSED,
                    message=f"Connected as {user_data.get('login', 'unknown')}"
                )
            elif response.status_code == 401:
                return CheckResult(
                    name="GitHub Connection",
                    status=CheckStatus.FAILED,
                    message="Authentication failed - invalid token",
                    fix_action="Generate new token at https://github.com/settings/tokens",
                    documentation="docs/TROUBLESHOOTING.md#github"
                )
            else:
                return CheckResult(
                    name="GitHub Connection",
                    status=CheckStatus.WARNING,
                    message=f"Unexpected response: {response.status_code}"
                )

        except requests.exceptions.Timeout:
            return CheckResult(
                name="GitHub Connection",
                status=CheckStatus.FAILED,
                message="Connection timeout",
                fix_action="Check network connection"
            )
        except Exception as e:
            return CheckResult(
                name="GitHub Connection",
                status=CheckStatus.WARNING,
                message=f"Connection issue: {str(e)[:100]}"
            )

    def check_directory_structure(self) -> CheckResult:
        """Verify required directories exist."""
        required_dirs = ["workspace", "workspace/queues", "workspace/activity"]

        missing = [d for d in required_dirs if not (self.workspace / d).exists()]

        if missing:
            return CheckResult(
                name="Directory Structure",
                status=CheckStatus.WARNING,
                message=f"Missing directories: {', '.join(missing)}",
                fix_action="Directories will be created automatically on first run"
            )

        return CheckResult(
            name="Directory Structure",
            status=CheckStatus.PASSED,
            message="All directories present"
        )

    def check_agent_definitions(self) -> CheckResult:
        """Verify agents.yaml is valid."""
        try:
            from ..core.config import load_agents

            agents_path = self.config_dir / "agents.yaml"
            if not agents_path.exists():
                return CheckResult(
                    name="Agent Definitions",
                    status=CheckStatus.FAILED,
                    message="agents.yaml not found",
                    fix_action="Run setup wizard to generate config"
                )

            agents = load_agents(agents_path)
            enabled_count = sum(1 for a in agents if a.enabled)

            if enabled_count == 0:
                return CheckResult(
                    name="Agent Definitions",
                    status=CheckStatus.WARNING,
                    message="No agents enabled",
                    fix_action="Enable at least one agent in config/agents.yaml"
                )

            return CheckResult(
                name="Agent Definitions",
                status=CheckStatus.PASSED,
                message=f"{enabled_count} agent(s) configured and enabled"
            )

        except Exception as e:
            return CheckResult(
                name="Agent Definitions",
                status=CheckStatus.FAILED,
                message=f"Invalid agents.yaml: {str(e)[:100]}",
                fix_action="Run setup wizard to regenerate config"
            )
