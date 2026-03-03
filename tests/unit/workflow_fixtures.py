"""Shared workflow definitions for unit tests.

These are simplified DAG workflows used across multiple test files.
They mirror the canonical workflow definitions in config/agent-framework.yaml
but may omit early steps (e.g. plan) that are irrelevant to the routing
behaviour under test.

TestPreviewWorkflowSync in test_preview_enforcement.py verifies that
PREVIEW_WORKFLOW's structure stays consistent with the live YAML.
"""

from agent_framework.core.config import WorkflowDefinition, WorkflowStepDefinition
from agent_framework.workflow.constants import WorkflowStepConstants as Steps

# Subset of the `preview` workflow starting at the preview step.
# The real workflow begins with a `plan` step (architect → architect), but
# preview-routing tests only need the preview → preview_review → implement
# portion, so the plan step is omitted here to keep fixtures minimal.
# Mirrors the review portion of the real default workflow: conditional edges
# at code_review and qa_review with NO "always" fallback.  When verdict is
# absent (ambiguous review output), no edge matches and execute_step returns
# False — which is the scenario Bug 1 and Bug 2 are designed to catch.
REVIEW_WORKFLOW = WorkflowDefinition(
    description="Workflow with conditional review steps",
    start_step=Steps.IMPLEMENT,
    pr_creator="architect",
    steps={
        Steps.IMPLEMENT: WorkflowStepDefinition(
            agent="engineer",
            next=[{"target": Steps.CODE_REVIEW}],
        ),
        Steps.CODE_REVIEW: WorkflowStepDefinition(
            agent="architect",
            next=[
                {"target": Steps.QA_REVIEW, "condition": "approved", "priority": 10},
                {"target": Steps.IMPLEMENT, "condition": "needs_fix", "priority": 5},
            ],
        ),
        Steps.QA_REVIEW: WorkflowStepDefinition(
            agent="qa",
            next=[
                {"target": Steps.CREATE_PR, "condition": "approved", "priority": 10},
                {"target": Steps.IMPLEMENT, "condition": "needs_fix", "priority": 5},
            ],
        ),
        Steps.CREATE_PR: WorkflowStepDefinition(agent="architect"),
    },
)

# Full pipeline workflow including the plan step. Used by pipeline E2E tests
# that exercise the complete chain: plan → implement → code_review → qa_review → create_pr.
PIPELINE_WORKFLOW = WorkflowDefinition(
    description="Full pipeline with planning step",
    start_step=Steps.PLAN,
    pr_creator="architect",
    steps={
        Steps.PLAN: WorkflowStepDefinition(
            agent="architect",
            next=[{"target": Steps.IMPLEMENT}],
        ),
        Steps.IMPLEMENT: WorkflowStepDefinition(
            agent="engineer",
            next=[{"target": Steps.CODE_REVIEW}],
        ),
        Steps.CODE_REVIEW: WorkflowStepDefinition(
            agent="architect",
            next=[
                {"target": Steps.QA_REVIEW, "condition": "approved", "priority": 10},
                {"target": Steps.IMPLEMENT, "condition": "needs_fix", "priority": 5},
            ],
        ),
        Steps.QA_REVIEW: WorkflowStepDefinition(
            agent="qa",
            next=[
                {"target": Steps.CREATE_PR, "condition": "approved", "priority": 10},
                {"target": Steps.IMPLEMENT, "condition": "needs_fix", "priority": 5},
            ],
        ),
        Steps.CREATE_PR: WorkflowStepDefinition(agent="architect"),
    },
)

PREVIEW_WORKFLOW = WorkflowDefinition(
    description="Read-only preview before implementation",
    start_step=Steps.PREVIEW,
    pr_creator="architect",
    steps={
        Steps.PREVIEW: WorkflowStepDefinition(
            agent="engineer",
            task_type="preview",
            next=[{"target": Steps.PREVIEW_REVIEW}],
        ),
        Steps.PREVIEW_REVIEW: WorkflowStepDefinition(
            agent="architect",
            next=[
                {"target": Steps.IMPLEMENT, "condition": "preview_approved", "priority": 10},
                {"target": Steps.PREVIEW, "condition": "needs_fix", "priority": 5},
                {"target": Steps.IMPLEMENT, "condition": "always", "priority": 0},
            ],
        ),
        Steps.IMPLEMENT: WorkflowStepDefinition(
            agent="engineer",
            next=[{"target": Steps.CREATE_PR}],
        ),
        Steps.CREATE_PR: WorkflowStepDefinition(agent="architect"),
    },
)
