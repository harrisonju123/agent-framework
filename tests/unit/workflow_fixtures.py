"""Shared workflow definitions for unit tests.

These are simplified DAG workflows used across multiple test files.
They mirror the canonical workflow definitions in config/agent-framework.yaml
but may omit early steps (e.g. plan) that are irrelevant to the routing
behaviour under test.

TestPreviewWorkflowSync in test_preview_enforcement.py verifies that
PREVIEW_WORKFLOW's structure stays consistent with the live YAML.
"""

from agent_framework.core.config import WorkflowDefinition, WorkflowStepDefinition

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
    start_step="implement",
    pr_creator="architect",
    steps={
        "implement": WorkflowStepDefinition(
            agent="engineer",
            next=[{"target": "code_review"}],
        ),
        "code_review": WorkflowStepDefinition(
            agent="architect",
            next=[
                {"target": "qa_review", "condition": "approved", "priority": 10},
                {"target": "implement", "condition": "needs_fix", "priority": 5},
            ],
        ),
        "qa_review": WorkflowStepDefinition(
            agent="qa",
            next=[
                {"target": "create_pr", "condition": "approved", "priority": 10},
                {"target": "implement", "condition": "needs_fix", "priority": 5},
            ],
        ),
        "create_pr": WorkflowStepDefinition(agent="architect"),
    },
)

PREVIEW_WORKFLOW = WorkflowDefinition(
    description="Read-only preview before implementation",
    start_step="preview",
    pr_creator="architect",
    steps={
        "preview": WorkflowStepDefinition(
            agent="engineer",
            task_type="preview",
            next=[{"target": "preview_review"}],
        ),
        "preview_review": WorkflowStepDefinition(
            agent="architect",
            next=[
                {"target": "implement", "condition": "preview_approved", "priority": 10},
                {"target": "preview", "condition": "needs_fix", "priority": 5},
                {"target": "implement", "condition": "always", "priority": 0},
            ],
        ),
        "implement": WorkflowStepDefinition(
            agent="engineer",
            next=[{"target": "create_pr"}],
        ),
        "create_pr": WorkflowStepDefinition(agent="architect"),
    },
)
