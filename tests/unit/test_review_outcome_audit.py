"""Tests for _parse_review_outcome_audited() in ReviewCycleManager."""

from unittest.mock import MagicMock

from agent_framework.core.review_cycle import ReviewCycleManager


def _make_manager():
    """Build a ReviewCycleManager with minimal deps for unit tests."""
    return ReviewCycleManager(
        config=MagicMock(id="qa", base_id="qa"),
        queue=MagicMock(),
        logger=MagicMock(),
        agent_definition=MagicMock(),
        session_logger=MagicMock(),
        activity_manager=MagicMock(),
    )


class TestParseReviewOutcomeAudited:
    def test_approved_match_details(self):
        mgr = _make_manager()
        content = "Everything looks good. APPROVED."
        outcome, audit = mgr._parse_review_outcome_audited(content)

        assert outcome.approved is True
        assert not outcome.needs_fix
        assert len(audit.matched_patterns) >= 1

        approve_matches = [p for p in audit.matched_patterns if p.category == "approve"]
        assert len(approve_matches) >= 1
        assert "APPROVED" in approve_matches[0].matched_text
        assert approve_matches[0].suppressed_by_negation is False

    def test_negation_suppression(self):
        """'no CRITICAL:' should suppress CRITICAL match (pattern requires colon)."""
        mgr = _make_manager()
        # Pattern is r'\bCRITICAL\b.*?:' so we need a colon after CRITICAL
        content = "APPROVED. There are no CRITICAL: issues in this code."
        outcome, audit = mgr._parse_review_outcome_audited(content)

        assert outcome.approved is True
        # CRITICAL match is negation-suppressed, not in matched_patterns
        critical_matches = [p for p in audit.matched_patterns if p.category == "critical_issues"]
        assert len(critical_matches) == 0
        suppressed = [p for p in audit.negation_suppressed if p.category == "critical_issues"]
        assert len(suppressed) >= 1
        assert suppressed[0].suppressed_by_negation is True

    def test_critical_override(self):
        """CRITICAL finding overrides APPROVE — override_applied should be True."""
        mgr = _make_manager()
        content = "APPROVED overall, but CRITICAL: security vulnerability in auth.py"
        outcome, audit = mgr._parse_review_outcome_audited(content)

        assert outcome.approved is False
        assert outcome.has_critical_issues is True
        assert audit.override_applied is True

    def test_severity_default_deny(self):
        """Severity tags without explicit APPROVE trigger default-deny."""
        mgr = _make_manager()
        content = "MEDIUM: Consider adding error handling\nLOW: Style nit"
        outcome, audit = mgr._parse_review_outcome_audited(content)

        assert outcome.has_major_issues is True
        assert audit.severity_tag_default_deny is True

    def test_empty_content(self):
        mgr = _make_manager()
        outcome, audit = mgr._parse_review_outcome_audited("")

        assert outcome.approved is False
        assert not outcome.needs_fix
        assert len(audit.matched_patterns) == 0
        assert audit.content_snippet == ""

    def test_needs_fix_match(self):
        mgr = _make_manager()
        content = "CHANGES REQUESTED: Please fix the broken tests. 3 failed."
        outcome, audit = mgr._parse_review_outcome_audited(content)

        assert outcome.needs_fix is True
        assert outcome.has_change_requests is True
        assert outcome.has_test_failures is True
        change_matches = [p for p in audit.matched_patterns if p.category == "request_changes"]
        assert len(change_matches) >= 1

    def test_content_snippet_truncated(self):
        mgr = _make_manager()
        content = "A" * 500
        _, audit = mgr._parse_review_outcome_audited(content)
        assert len(audit.content_snippet) == 200

    def test_multiple_matches_per_category(self):
        """finditer should capture ALL matches, not just the first."""
        mgr = _make_manager()
        content = "LGTM at line 10. Also APPROVED at line 50."
        outcome, audit = mgr._parse_review_outcome_audited(content)

        approve_matches = [p for p in audit.matched_patterns if p.category == "approve"]
        assert len(approve_matches) >= 2

    def test_delegates_to_parse_review_outcome(self):
        """parse_review_outcome() should return same outcome as audited version."""
        mgr = _make_manager()
        content = "APPROVED. No issues."
        outcome_plain = mgr.parse_review_outcome(content)
        outcome_audited, _ = mgr._parse_review_outcome_audited(content)

        assert outcome_plain.approved == outcome_audited.approved
        assert outcome_plain.has_critical_issues == outcome_audited.has_critical_issues
        assert outcome_plain.has_test_failures == outcome_audited.has_test_failures
        assert outcome_plain.has_change_requests == outcome_audited.has_change_requests
        assert outcome_plain.has_major_issues == outcome_audited.has_major_issues
