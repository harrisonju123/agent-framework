"""Tests for VerdictAudit and PatternMatch dataclasses."""

from agent_framework.core.verdict_audit import PatternMatch, VerdictAudit


class TestPatternMatch:
    def test_basic_fields(self):
        pm = PatternMatch(
            category="approve",
            pattern=r'\bAPPROVE[D]?\b',
            matched_text="APPROVED",
            position=42,
            suppressed_by_negation=False,
        )
        assert pm.category == "approve"
        assert pm.pattern == r'\bAPPROVE[D]?\b'
        assert pm.matched_text == "APPROVED"
        assert pm.position == 42
        assert pm.suppressed_by_negation is False

    def test_suppressed_match(self):
        pm = PatternMatch(
            category="critical_issues",
            pattern=r'\bCRITICAL\b.*?:',
            matched_text="CRITICAL:",
            position=10,
            suppressed_by_negation=True,
        )
        assert pm.suppressed_by_negation is True


class TestVerdictAudit:
    def test_to_dict_round_trip(self):
        matches = [
            PatternMatch("approve", r'\bAPPROVE[D]?\b', "APPROVED", 0, False),
        ]
        suppressed = [
            PatternMatch("critical_issues", r'\bCRITICAL\b.*?:', "CRITICAL:", 50, True),
        ]

        audit = VerdictAudit(
            method="review_outcome",
            value="approved",
            agent_id="qa-1",
            workflow_step="qa_review",
            task_id="chain-abc-qa_review-d3",
            outcome_flags={"approved": True, "has_critical": False},
            matched_patterns=matches,
            negation_suppressed=suppressed,
            override_applied=False,
            severity_tag_default_deny=False,
            no_changes_marker_found=False,
            content_snippet="APPROVED. No issues found.",
        )

        d = audit.to_dict()

        assert d["method"] == "review_outcome"
        assert d["value"] == "approved"
        assert d["agent_id"] == "qa-1"
        assert d["workflow_step"] == "qa_review"
        assert d["task_id"] == "chain-abc-qa_review-d3"
        assert d["outcome_flags"]["approved"] is True
        assert len(d["matched_patterns"]) == 1
        assert d["matched_patterns"][0]["category"] == "approve"
        assert len(d["negation_suppressed"]) == 1
        assert d["negation_suppressed"][0]["suppressed_by_negation"] is True
        assert d["override_applied"] is False
        assert d["content_snippet"] == "APPROVED. No issues found."

    def test_to_dict_defaults(self):
        audit = VerdictAudit(
            method="ambiguous_halt",
            value=None,
            agent_id="architect",
            workflow_step="code_review",
            task_id="task-1",
        )
        d = audit.to_dict()

        assert d["value"] is None
        assert d["outcome_flags"] is None
        assert d["matched_patterns"] == []
        assert d["negation_suppressed"] == []
        assert d["override_applied"] is False
        assert d["severity_tag_default_deny"] is False
        assert d["no_changes_marker_found"] is False
        assert d["content_snippet"] == ""

    def test_to_dict_with_override(self):
        audit = VerdictAudit(
            method="review_outcome",
            value="needs_fix",
            agent_id="qa",
            workflow_step="qa_review",
            task_id="task-2",
            override_applied=True,
            severity_tag_default_deny=False,
        )
        d = audit.to_dict()
        assert d["override_applied"] is True

    def test_to_dict_with_no_changes_marker(self):
        audit = VerdictAudit(
            method="no_changes_marker",
            value="no_changes",
            agent_id="architect",
            workflow_step="plan",
            task_id="task-3",
            no_changes_marker_found=True,
        )
        d = audit.to_dict()
        assert d["no_changes_marker_found"] is True
        assert d["method"] == "no_changes_marker"
