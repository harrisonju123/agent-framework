"""Tests for verdict_audit persistence through chain state save/load."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from agent_framework.core.chain_state import (
    ChainState,
    StepRecord,
    append_step,
    load_chain_state,
    save_chain_state,
)


def _make_task(**ctx_overrides):
    task = MagicMock()
    task.id = ctx_overrides.pop("task_id", "chain-root-implement-d1")
    task.description = "Test task"
    task.plan = None
    task.root_id = "root-task-123"
    task.context = {
        "workflow_step": "implement",
        "_root_task_id": "root-task-123",
        "user_goal": "Add verdict audit trail",
        "workflow": "default",
    }
    task.context.update(ctx_overrides)
    return task


class TestChainStateVerdictAudit:
    def test_verdict_audit_in_step_record(self):
        """StepRecord can store verdict_audit dict."""
        audit_dict = {
            "method": "review_outcome",
            "value": "approved",
            "agent_id": "qa",
            "workflow_step": "qa_review",
            "task_id": "chain-abc-qa-d3",
            "matched_patterns": [{"category": "approve", "pattern": "\\bAPPROVE[D]?\\b"}],
            "negation_suppressed": [],
            "override_applied": False,
        }

        record = StepRecord(
            step_id="qa_review",
            agent_id="qa",
            task_id="chain-abc-qa-d3",
            completed_at="2026-02-20T12:00:00+00:00",
            summary="QA approved",
            verdict="approved",
            verdict_audit=audit_dict,
        )

        assert record.verdict_audit is not None
        assert record.verdict_audit["method"] == "review_outcome"

    def test_save_load_round_trip(self):
        """verdict_audit dict persists through chain state save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            audit_dict = {
                "method": "review_outcome",
                "value": "needs_fix",
                "agent_id": "architect",
                "workflow_step": "code_review",
                "task_id": "chain-root-code_review-d2",
                "outcome_flags": {"approved": False, "has_critical": True},
                "matched_patterns": [
                    {"category": "critical_issues", "pattern": "\\bCRITICAL\\b.*?:"}
                ],
                "negation_suppressed": [],
                "override_applied": True,
                "severity_tag_default_deny": False,
                "no_changes_marker_found": False,
                "content_snippet": "CRITICAL: Missing null check",
            }

            state = ChainState(
                root_task_id="root-123",
                user_goal="Add feature",
                workflow="default",
            )
            state.steps.append(StepRecord(
                step_id="code_review",
                agent_id="architect",
                task_id="chain-root-code_review-d2",
                completed_at="2026-02-20T12:00:00+00:00",
                summary="Found critical issue",
                verdict="needs_fix",
                verdict_audit=audit_dict,
            ))

            save_chain_state(workspace, state)
            loaded = load_chain_state(workspace, "root-123")

            assert loaded is not None
            assert len(loaded.steps) == 1
            step = loaded.steps[0]
            assert step.verdict_audit is not None
            assert step.verdict_audit["method"] == "review_outcome"
            assert step.verdict_audit["override_applied"] is True
            assert step.verdict_audit["outcome_flags"]["has_critical"] is True

    def test_append_step_populates_verdict_audit(self):
        """append_step() reads verdict_audit from task.context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            audit_dict = {
                "method": "ambiguous_halt",
                "value": None,
                "agent_id": "qa",
                "workflow_step": "qa_review",
                "task_id": "chain-root-qa_review-d3",
            }

            task = _make_task(
                task_id="chain-root-qa_review-d3",
                workflow_step="qa_review",
                verdict_audit=audit_dict,
            )

            state = append_step(
                workspace, task, "qa", "Ambiguous review output",
            )

            assert len(state.steps) == 1
            assert state.steps[0].verdict_audit is not None
            assert state.steps[0].verdict_audit["method"] == "ambiguous_halt"

    def test_backward_compat_without_verdict_audit(self):
        """Old chain state files without verdict_audit field still load fine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            state_dir = workspace / ".agent-communication" / "chain-state"
            state_dir.mkdir(parents=True)

            # Write a chain state without verdict_audit field in steps
            data = {
                "root_task_id": "old-root",
                "user_goal": "Old task",
                "workflow": "default",
                "steps": [{
                    "step_id": "plan",
                    "agent_id": "architect",
                    "task_id": "chain-old-root-plan-d1",
                    "completed_at": "2026-02-19T12:00:00+00:00",
                    "summary": "Plan created",
                    "verdict": "approved",
                }],
            }
            (state_dir / "old-root.json").write_text(json.dumps(data))

            loaded = load_chain_state(workspace, "old-root")
            assert loaded is not None
            assert len(loaded.steps) == 1
            # verdict_audit defaults to None for old records
            assert loaded.steps[0].verdict_audit is None
