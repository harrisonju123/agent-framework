"""Translate technical errors to user-friendly messages."""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class UserFriendlyError:
    """User-friendly error representation."""
    original_error: Exception
    title: str
    explanation: str
    actions: List[str]
    documentation: Optional[str] = None
    show_technical: bool = False


class ErrorTranslator:
    """Translate technical errors to user-friendly messages."""

    ERROR_PATTERNS = {
        # Circuit breaker
        r"CircuitBreakerOpenError": {
            "title": "System safeguards activated",
            "explanation": "The system detected too many failures and paused operations to prevent further issues. This is a safety feature.",
            "actions": [
                "Check recent failures: agent status",
                "View system health: agent doctor",
                "Reset safeguards: agent check --fix",
                "Contact support if issue persists"
            ],
            "documentation": "docs/SAFEGUARDS.md"
        },

        # MCP errors
        r"mcp.*server.*not.*found|mcp.*not.*available": {
            "title": "Real-time integrations not set up",
            "explanation": "The agent tried to use JIRA/GitHub but the MCP servers aren't configured. This is optional for basic usage.",
            "actions": [
                "Option 1: Disable MCP in config (set llm.use_mcp: false)",
                "Option 2: Set up MCP servers (see MCP_SETUP.md)",
                "Option 3: Continue - agents will work without real-time access"
            ],
            "documentation": "docs/MCP_SETUP.md"
        },

        # JIRA auth errors
        r"JIRA.*401|Unauthorized.*JIRA": {
            "title": "JIRA authentication failed",
            "explanation": "Your JIRA API token is invalid or expired.",
            "actions": [
                "Generate new API token: https://id.atlassian.com/manage-profile/security/api-tokens",
                "Update credentials: Run setup wizard in dashboard",
                "Verify JIRA server URL is correct"
            ],
            "documentation": "docs/TROUBLESHOOTING.md#jira"
        },

        # GitHub auth errors
        r"GitHub.*401|Bad credentials": {
            "title": "GitHub authentication failed",
            "explanation": "Your GitHub personal access token is invalid or lacks required permissions.",
            "actions": [
                "Generate new token: https://github.com/settings/tokens (needs 'repo' scope)",
                "Update credentials: Run setup wizard in dashboard",
                "Check token hasn't expired"
            ],
            "documentation": "docs/TROUBLESHOOTING.md#github"
        },

        # Rate limiting
        r"rate.*limit|429|too many requests": {
            "title": "API rate limit exceeded",
            "explanation": "You've made too many requests to JIRA or GitHub. Rate limits reset periodically.",
            "actions": [
                "Wait 15-60 minutes for rate limit to reset",
                "Check rate limits: agent status",
                "Consider reducing polling frequency in config"
            ]
        },

        # LLM proxy errors (must precede generic network pattern)
        r"connection.*refused.*ANTHROPIC_BASE_URL|Cannot reach LLM proxy": {
            "title": "Cannot reach LLM proxy",
            "explanation": "The agent cannot connect to the configured LiteLLM proxy server.",
            "actions": [
                "Verify proxy_url in config/agent-framework.yaml",
                "Check the proxy server is running",
                "Check network connectivity: curl <proxy_url>/health",
                "Run health check: agent doctor"
            ],
            "documentation": "docs/TROUBLESHOOTING.md#proxy"
        },

        r"proxy.*401|proxy.*unauthorized|ANTHROPIC_AUTH_TOKEN.*invalid": {
            "title": "LLM proxy authentication failed",
            "explanation": "The proxy rejected the authentication token.",
            "actions": [
                "Check proxy_auth_token in config/agent-framework.yaml",
                "Verify the token hasn't expired",
                "Contact your proxy administrator for a new token"
            ],
            "documentation": "docs/TROUBLESHOOTING.md#proxy"
        },

        # Network errors
        r"connection.*refused|connection.*timeout|network.*unreachable": {
            "title": "Cannot connect to service",
            "explanation": "Unable to reach JIRA or GitHub servers. This could be a network issue or service outage.",
            "actions": [
                "Check your internet connection",
                "Verify service URLs in configuration",
                "Check if JIRA/GitHub is experiencing an outage",
                "Try again in a few minutes"
            ]
        },

        # Config errors
        r"config.*not.*found|no such file.*config": {
            "title": "Configuration missing",
            "explanation": "Required configuration files were not found. The system needs to be set up first.",
            "actions": [
                "Run setup: Open dashboard and click Setup button",
                "Or use CLI: agent init"
            ],
            "documentation": "README.md#setup"
        }
    }

    def translate(self, error: Exception) -> UserFriendlyError:
        """Convert exception to user-friendly format."""
        error_str = str(error)
        error_type = type(error).__name__
        full_error = f"{error_type}: {error_str}"

        # Try to match error patterns
        for pattern, translation in self.ERROR_PATTERNS.items():
            if re.search(pattern, full_error, re.IGNORECASE):
                return UserFriendlyError(
                    original_error=error,
                    title=translation["title"],
                    explanation=translation["explanation"],
                    actions=translation["actions"],
                    documentation=translation.get("documentation"),
                    show_technical=False
                )

        # Fallback for unknown errors
        return UserFriendlyError(
            original_error=error,
            title="Unexpected error",
            explanation=str(error),
            actions=[
                "Run health check: agent doctor",
                "Check logs for details",
                "Report issue: https://github.com/your-org/agent-framework/issues"
            ],
            show_technical=True
        )

    def format_for_cli(self, friendly_error: UserFriendlyError) -> str:
        """Format error for CLI display."""
        output = f"[bold red]{friendly_error.title}[/]\n\n"
        output += f"{friendly_error.explanation}\n\n"

        # Actions
        output += "[bold]How to fix:[/]\n"
        for i, action in enumerate(friendly_error.actions, 1):
            output += f"  {i}. {action}\n"

        # Documentation link
        if friendly_error.documentation:
            output += f"\n[dim]📖 Learn more: {friendly_error.documentation}[/]"

        # Technical details (if needed)
        if friendly_error.show_technical:
            output += f"\n\n[dim]Technical details:[/]\n[dim]{friendly_error.original_error}[/]"

        return output
