"""Centralized workflow step ID constants.

Single source of truth for all workflow step identifiers used across
the agent framework. Import from here instead of hardcoding strings.

This module is intentionally dependency-free (no intra-package imports)
to avoid circular import issues.
"""


class WorkflowStepConstants:
    """Workflow step ID string constants and convenience groupings."""

    # -- Core step IDs --
    PLAN = "plan"
    IMPLEMENT = "implement"
    CODE_REVIEW = "code_review"
    QA_REVIEW = "qa_review"
    CREATE_PR = "create_pr"
    PREVIEW = "preview"
    PREVIEW_REVIEW = "preview_review"

    # -- Aliases (used as step IDs in some contexts) --
    FIX = "fix"
    IMPLEMENTATION = "implementation"

    # -- Convenience groupings --

    # Steps that produce code changes
    IMPLEMENTATION_STEPS: frozenset[str] = frozenset({IMPLEMENT, IMPLEMENTATION})

    # Steps that produce prose, not code
    NON_CODE_STEPS: frozenset[str] = frozenset({
        PLAN, "planning", CODE_REVIEW, QA_REVIEW,
        CREATE_PR, PREVIEW_REVIEW, PREVIEW,
    })

    # Steps that perform review
    REVIEW_STEPS: frozenset[str] = frozenset({PREVIEW_REVIEW, CODE_REVIEW, QA_REVIEW})

    # Preview review steps (subset for special handling)
    PREVIEW_REVIEW_STEPS: frozenset[str] = frozenset({PREVIEW_REVIEW})

    # Steps where agent works alone (no MCP tools needed)
    SOLO_WORKFLOW_STEPS: frozenset[str] = frozenset({PLAN, CODE_REVIEW, PREVIEW_REVIEW, CREATE_PR})

    # Steps where testing injections are suppressed in prompts
    TEST_SUPPRESSED_STEPS: frozenset[str] = frozenset({CODE_REVIEW, PREVIEW_REVIEW, CREATE_PR})
