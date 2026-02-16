"""Tests for ErrorTranslator — verifies user-friendly error messages."""

import pytest

from agent_framework.errors.translator import ErrorTranslator, UserFriendlyError


class TestBudgetErrorTranslation:
    """Tests for budget error translation to user-friendly messages."""

    def test_budget_exceeded_error_translation(self):
        """Test that budget exceeded error is translated properly."""
        translator = ErrorTranslator()
        error = Exception("budget exceeded")

        result = translator.translate(error)

        assert isinstance(result, UserFriendlyError)
        assert result.title == "Budget or quota exceeded"
        assert "budget" in result.explanation.lower()
        assert len(result.actions) > 0

    def test_max_budget_error_translation(self):
        """Test that max budget error is translated properly."""
        translator = ErrorTranslator()
        error = Exception("max budget reached")

        result = translator.translate(error)

        assert result.title == "Budget or quota exceeded"
        assert "budget" in result.explanation.lower()

    def test_quota_exceeded_error_translation(self):
        """Test that quota exceeded error is translated properly."""
        translator = ErrorTranslator()
        error = Exception("quota exceeded")

        result = translator.translate(error)

        assert result.title == "Budget or quota exceeded"

    def test_insufficient_credits_error_translation(self):
        """Test that insufficient credits error is translated properly."""
        translator = ErrorTranslator()
        error = Exception("insufficient credits")

        result = translator.translate(error)

        assert result.title == "Budget or quota exceeded"
        assert "credits" in result.explanation.lower() or "quota" in result.explanation.lower()

    def test_usage_limit_exceeded_error_translation(self):
        """Test that usage limit exceeded error is translated properly."""
        translator = ErrorTranslator()
        error = Exception("usage limit exceeded")

        result = translator.translate(error)

        assert result.title == "Budget or quota exceeded"

    def test_budget_error_case_insensitive(self):
        """Test that budget error translation is case insensitive."""
        translator = ErrorTranslator()

        # Test various case combinations
        for error_msg in ["BUDGET EXCEEDED", "Budget Exceeded", "BuDgEt ExCeEdeD"]:
            error = Exception(error_msg)
            result = translator.translate(error)
            assert result.title == "Budget or quota exceeded"

    def test_budget_error_has_actionable_guidance(self):
        """Test that budget errors include actionable guidance."""
        translator = ErrorTranslator()
        error = Exception("budget exceeded")

        result = translator.translate(error)

        # Check that we have actionable steps
        assert len(result.actions) >= 2
        action_text = " ".join(result.actions).lower()
        assert any(keyword in action_text for keyword in [
            "account", "usage", "budget", "credits", "plan", "upgrade"
        ])

    def test_budget_error_has_documentation_link(self):
        """Test that budget errors include documentation reference."""
        translator = ErrorTranslator()
        error = Exception("budget exceeded")

        result = translator.translate(error)

        assert result.documentation is not None
        assert "budget" in result.documentation.lower() or "docs" in result.documentation.lower()

    def test_budget_error_original_exception_preserved(self):
        """Test that original exception is preserved in translation."""
        translator = ErrorTranslator()
        original_error = Exception("budget exceeded")

        result = translator.translate(original_error)

        assert result.original_error is original_error

    def test_budget_error_not_marked_technical(self):
        """Test that budget errors don't show technical details by default."""
        translator = ErrorTranslator()
        error = Exception("budget exceeded")

        result = translator.translate(error)

        assert result.show_technical is False


class TestErrorTranslatorFormatting:
    """Tests for error formatting for CLI display."""

    def test_format_budget_error_for_cli(self):
        """Test that budget error formats correctly for CLI."""
        translator = ErrorTranslator()
        error = Exception("budget exceeded")
        friendly_error = translator.translate(error)

        formatted = translator.format_for_cli(friendly_error)

        assert "Budget or quota exceeded" in formatted
        assert "How to fix:" in formatted
        assert len(formatted) > 0

    def test_formatted_output_includes_all_actions(self):
        """Test that formatted output includes all action items."""
        translator = ErrorTranslator()
        error = Exception("budget exceeded")
        friendly_error = translator.translate(error)

        formatted = translator.format_for_cli(friendly_error)

        # Count the numbered items in formatted output
        for i, action in enumerate(friendly_error.actions, 1):
            assert f"{i}." in formatted

    def test_formatted_output_includes_documentation_link(self):
        """Test that formatted output includes documentation link if present."""
        translator = ErrorTranslator()
        error = Exception("budget exceeded")
        friendly_error = translator.translate(error)

        formatted = translator.format_for_cli(friendly_error)

        if friendly_error.documentation:
            assert friendly_error.documentation in formatted


class TestNonBudgetErrors:
    """Tests to ensure other error types still work correctly."""

    def test_authentication_error_translation(self):
        """Test that authentication errors are still translated properly."""
        translator = ErrorTranslator()
        error = Exception("JIRA 401 Unauthorized")

        result = translator.translate(error)

        assert "authentication" in result.title.lower() or "failed" in result.title.lower()

    def test_network_error_translation(self):
        """Test that network errors are still translated properly."""
        translator = ErrorTranslator()
        error = Exception("connection refused")

        result = translator.translate(error)

        assert "connect" in result.title.lower() or "network" in result.title.lower()

    def test_unknown_error_fallback(self):
        """Test that unknown errors have reasonable fallback."""
        translator = ErrorTranslator()
        error = Exception("some completely unknown error message")

        result = translator.translate(error)

        assert result.title == "Unexpected error"
        assert result.show_technical is True
        assert len(result.actions) > 0
