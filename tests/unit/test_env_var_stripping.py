"""Tests for sensitive environment variable stripping in Claude CLI subprocess."""

from agent_framework.llm.claude_cli_backend import _SENSITIVE_ENV_VARS


class TestSensitiveEnvVars:
    def test_jira_api_token_is_stripped(self):
        assert 'JIRA_API_TOKEN' in _SENSITIVE_ENV_VARS

    def test_jira_email_is_stripped(self):
        assert 'JIRA_EMAIL' in _SENSITIVE_ENV_VARS

    def test_github_token_is_not_stripped(self):
        """GITHUB_TOKEN must remain — needed for git push auth via credential helper."""
        assert 'GITHUB_TOKEN' not in _SENSITIVE_ENV_VARS

    def test_set_is_frozen(self):
        """Prevent accidental mutation."""
        assert isinstance(_SENSITIVE_ENV_VARS, frozenset)
