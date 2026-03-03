"""Tests for WorkflowStepConstants — verifies each constant and frozenset grouping."""

from agent_framework.workflow.constants import WorkflowStepConstants as Steps


class TestStepIDConstants:
    """Each constant matches its expected string value."""

    def test_plan(self):
        assert Steps.PLAN == "plan"

    def test_implement(self):
        assert Steps.IMPLEMENT == "implement"

    def test_code_review(self):
        assert Steps.CODE_REVIEW == "code_review"

    def test_qa_review(self):
        assert Steps.QA_REVIEW == "qa_review"

    def test_create_pr(self):
        assert Steps.CREATE_PR == "create_pr"

    def test_preview(self):
        assert Steps.PREVIEW == "preview"

    def test_preview_review(self):
        assert Steps.PREVIEW_REVIEW == "preview_review"

    def test_fix_alias(self):
        assert Steps.FIX == "fix"

    def test_implementation_alias(self):
        assert Steps.IMPLEMENTATION == "implementation"


class TestFrozensetGroupings:
    """Convenience frozensets contain the right members."""

    def test_implementation_steps(self):
        assert Steps.IMPLEMENTATION_STEPS == frozenset({"implement", "implementation"})

    def test_non_code_steps(self):
        assert Steps.NON_CODE_STEPS == frozenset({
            "plan", "planning", "code_review", "qa_review",
            "create_pr", "preview_review", "preview",
        })

    def test_review_steps(self):
        assert Steps.REVIEW_STEPS == frozenset({
            "preview_review", "code_review", "qa_review",
        })

    def test_preview_review_steps(self):
        assert Steps.PREVIEW_REVIEW_STEPS == frozenset({"preview_review"})

    def test_solo_workflow_steps(self):
        assert Steps.SOLO_WORKFLOW_STEPS == frozenset({
            "plan", "code_review", "preview_review", "create_pr",
        })

    def test_test_suppressed_steps(self):
        assert Steps.TEST_SUPPRESSED_STEPS == frozenset({
            "code_review", "preview_review", "create_pr",
        })

    def test_groupings_use_constants(self):
        """Frozensets are built from the class constants, not independent strings."""
        assert Steps.IMPLEMENT in Steps.IMPLEMENTATION_STEPS
        assert Steps.CODE_REVIEW in Steps.REVIEW_STEPS
        assert Steps.QA_REVIEW in Steps.REVIEW_STEPS
        assert Steps.PREVIEW_REVIEW in Steps.PREVIEW_REVIEW_STEPS
        assert Steps.PLAN in Steps.SOLO_WORKFLOW_STEPS
        assert Steps.CREATE_PR in Steps.TEST_SUPPRESSED_STEPS
