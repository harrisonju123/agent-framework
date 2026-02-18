"""Shared test fixtures and constants for unit tests."""

from agent_framework.core.config import WorkflowDefinition, WorkflowStepDefinition

# DAG workflow used by preview-related tests across multiple test files.
# Mirrors the `preview` entry in config/agent-framework.yaml.
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
            ],
        ),
        "implement": WorkflowStepDefinition(
            agent="engineer",
            next=[{"target": "create_pr"}],
        ),
        "create_pr": WorkflowStepDefinition(agent="architect"),
    },
)
