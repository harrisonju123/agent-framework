"""Tests for the QA → Engineer review feedback loop."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_framework.core.agent import (
    Agent,
    AgentConfig,
    MAX_REVIEW_CYCLES,
    QAFinding,
    ReviewOutcome,
)
from agent_framework.core.task import Task, TaskStatus, TaskType


# -- Fixtures --

def _make_task(
    task_type=TaskType.REVIEW,
    assigned_to="qa",
    task_id="review-task-abc123",
    **ctx_overrides,
):
    context = {
        "jira_key": "PROJ-42",
        "pr_url": "https://github.com/org/repo/pull/99",
        "pr_number": 99,
        "github_repo": "org/repo",
        "workflow": "default",
        **ctx_overrides,
    }
    return Task(
        id=task_id,
        type=task_type,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="engineer",
        assigned_to=assigned_to,
        created_at=datetime.now(timezone.utc),
        title="Review PR #99",
        description="Review the PR.",
        context=context,
    )


def _make_response(content="APPROVE - looks good"):
    return SimpleNamespace(
        content=content,
        error=None,
        input_tokens=100,
        output_tokens=50,
        model_used="sonnet",
        latency_ms=1000,
        finish_reason="end_turn",
    )


@pytest.fixture
def queue(tmp_path):
    q = MagicMock()
    q.queue_dir = tmp_path / "queues"
    q.queue_dir.mkdir()
    return q


@pytest.fixture
def qa_agent(queue):
    """QA agent for testing review fix logic."""
    from agent_framework.core.review_cycle import ReviewCycleManager
    config = AgentConfig(
        id="qa",
        name="QA Engineer",
        queue="qa",
        prompt="You are QA.",
    )
    a = Agent.__new__(Agent)
    a.config = config
    a.queue = queue
    a.logger = MagicMock()
    a.jira_client = None
    a._agent_definition = None
    a._review_cycle = ReviewCycleManager(
        config=config,
        queue=queue,
        logger=a.logger,
        agent_definition=None,
        session_logger=MagicMock(),
        activity_manager=MagicMock(),
    )
    return a


@pytest.fixture
def engineer_agent(queue):
    """Engineer agent — should never trigger review fix logic."""
    from agent_framework.core.review_cycle import ReviewCycleManager
    config = AgentConfig(
        id="engineer",
        name="Engineer",
        queue="engineer",
        prompt="You are an engineer.",
    )
    a = Agent.__new__(Agent)
    a.config = config
    a.queue = queue
    a.logger = MagicMock()
    a._review_cycle = ReviewCycleManager(
        config=config,
        queue=queue,
        logger=a.logger,
        agent_definition=None,
        session_logger=MagicMock(),
        activity_manager=MagicMock(),
    )
    return a


# -- ReviewOutcome dataclass --

class TestReviewOutcome:
    def test_needs_fix_when_critical(self):
        o = ReviewOutcome(
            approved=False, has_critical_issues=True,
            has_test_failures=False, has_change_requests=False,
            findings_summary="CRITICAL: sql injection",
        )
        assert o.needs_fix is True

    def test_needs_fix_when_test_failures(self):
        o = ReviewOutcome(
            approved=False, has_critical_issues=False,
            has_test_failures=True, has_change_requests=False,
            findings_summary="3 failed",
        )
        assert o.needs_fix is True

    def test_needs_fix_when_changes_requested(self):
        o = ReviewOutcome(
            approved=False, has_critical_issues=False,
            has_test_failures=False, has_change_requests=True,
            findings_summary="REQUEST_CHANGES",
        )
        assert o.needs_fix is True

    def test_no_fix_when_approved(self):
        o = ReviewOutcome(
            approved=True, has_critical_issues=False,
            has_test_failures=False, has_change_requests=False,
            findings_summary="LGTM",
        )
        assert o.needs_fix is False

    def test_needs_fix_when_major_issues(self):
        o = ReviewOutcome(
            approved=False, has_critical_issues=False,
            has_test_failures=False, has_change_requests=False,
            findings_summary="MAJOR: missing input validation",
            has_major_issues=True,
        )
        assert o.needs_fix is True


# -- _parse_review_outcome --

class TestParseReviewOutcome:
    def test_detects_approve(self, qa_agent):
        outcome = qa_agent._parse_review_outcome("APPROVE - all checks pass")
        assert outcome.approved is True
        assert outcome.needs_fix is False

    def test_detects_lgtm(self, qa_agent):
        outcome = qa_agent._parse_review_outcome("LGTM, ship it")
        assert outcome.approved is True

    def test_detects_request_changes(self, qa_agent):
        outcome = qa_agent._parse_review_outcome("Overall: REQUEST_CHANGES\nFix the bug.")
        assert outcome.has_change_requests is True
        assert outcome.approved is False
        assert outcome.needs_fix is True

    def test_detects_critical_issues(self, qa_agent):
        outcome = qa_agent._parse_review_outcome("CRITICAL: SQL injection in user input handler")
        assert outcome.has_critical_issues is True
        assert outcome.needs_fix is True

    def test_detects_test_failures(self, qa_agent):
        outcome = qa_agent._parse_review_outcome("Results: 12 passed, 3 failed")
        assert outcome.has_test_failures is True
        assert outcome.needs_fix is True

    def test_approval_overridden_by_issues(self, qa_agent):
        """If both APPROVE and issues are present, issues win."""
        outcome = qa_agent._parse_review_outcome(
            "APPROVE with reservations\nCRITICAL: missing auth check"
        )
        assert outcome.approved is False
        assert outcome.needs_fix is True

    def test_zero_failed_does_not_trigger(self, qa_agent):
        """'0 failed' should not be detected as a test failure."""
        outcome = qa_agent._parse_review_outcome("Results: 12 passed, 0 failed")
        assert outcome.has_test_failures is False

    def test_empty_content(self, qa_agent):
        outcome = qa_agent._parse_review_outcome("")
        assert outcome.approved is False
        assert outcome.needs_fix is False

    def test_detects_major_issues(self, qa_agent):
        """Uppercase MAJOR: triggers has_major_issues."""
        outcome = qa_agent._parse_review_outcome("MAJOR: missing input validation on /api/users")
        assert outcome.has_major_issues is True
        assert outcome.needs_fix is True

    def test_detects_high_severity(self, qa_agent):
        """Uppercase HIGH: triggers has_major_issues."""
        outcome = qa_agent._parse_review_outcome("HIGH: race condition in payment handler")
        assert outcome.has_major_issues is True
        assert outcome.needs_fix is True

    def test_lowercase_major_does_not_trigger(self, qa_agent):
        """Prose like 'not a major concern:' should NOT trigger major_issues."""
        outcome = qa_agent._parse_review_outcome("APPROVE\nThis is not a major concern: just cosmetic")
        assert outcome.has_major_issues is False
        assert outcome.approved is True

    def test_severity_tags_without_approve_trigger_fix(self, qa_agent):
        """Severity-tagged findings without explicit APPROVE → needs_fix."""
        outcome = qa_agent._parse_review_outcome(
            "MINOR: unused import\nSUGGESTION: rename variable for clarity"
        )
        assert outcome.approved is False
        assert outcome.has_major_issues is True
        assert outcome.needs_fix is True

    def test_minor_suggestions_with_approve_pass(self, qa_agent):
        """MINOR/SUGGESTION + explicit APPROVE → approved, no fix needed."""
        outcome = qa_agent._parse_review_outcome(
            "APPROVE\nMINOR: unused import\nSUGGESTION: rename variable"
        )
        assert outcome.approved is True
        assert outcome.needs_fix is False

    def test_ambiguous_response_with_findings_triggers_fix(self, qa_agent):
        """No verdict keyword but severity tags present → needs_fix."""
        outcome = qa_agent._parse_review_outcome(
            "Found some issues:\nMAJOR: SQL injection in login\nMINOR: typo in docs"
        )
        assert outcome.approved is False
        assert outcome.needs_fix is True

    def test_soft_approval_without_keyword_does_not_approve(self, qa_agent):
        """'looks good, no issues' without APPROVE/LGTM → no severity tags, ambiguous at parse level."""
        outcome = qa_agent._parse_review_outcome("looks good, no issues found")
        assert outcome.approved is False
        # No severity tags → parse returns needs_fix=False; queue guard catches the ambiguity
        assert outcome.needs_fix is False

    def test_major_overrides_approve(self, qa_agent):
        """MAJOR findings override explicit APPROVE (same as CRITICAL)."""
        outcome = qa_agent._parse_review_outcome(
            "APPROVE with reservations\nMAJOR: missing rate limiting"
        )
        assert outcome.approved is False
        assert outcome.has_major_issues is True
        assert outcome.needs_fix is True


# -- _extract_review_findings --

class TestExtractFindingsCaseSensitivity:
    def test_ignores_lowercase_severity_tags(self, qa_agent):
        """Severity tags must be uppercase to be captured."""
        content = "suggestion: maybe rename this variable"
        findings_summary, structured_findings = qa_agent._extract_review_findings(content)
        # Falls back to truncated content since no uppercase tags found
        assert findings_summary == content[:500]
        assert structured_findings == []

class TestExtractReviewFindings:
    def test_extracts_severity_lines(self, qa_agent):
        content = """Review summary:

CRITICAL: SQL injection in login handler
HIGH: Missing rate limiting on /api/users
MINOR: Unused import in utils.py

Overall: REQUEST_CHANGES"""
        findings_summary, structured_findings = qa_agent._extract_review_findings(content)
        assert "CRITICAL: SQL injection" in findings_summary
        assert "HIGH: Missing rate limiting" in findings_summary
        assert "MINOR: Unused import" in findings_summary

    def test_falls_back_to_truncated_content(self, qa_agent):
        content = "No tagged lines here, just general feedback about the code."
        findings_summary, structured_findings = qa_agent._extract_review_findings(content)
        assert findings_summary == content[:500]
        assert structured_findings == []


# -- Guard conditions --

class TestQueueReviewFixGuards:
    def test_skips_for_non_qa_agent(self, engineer_agent):
        task = _make_task()
        response = _make_response("REQUEST_CHANGES")
        engineer_agent._queue_review_fix_if_needed(task, response)
        engineer_agent.queue.push.assert_not_called()

    def test_skips_for_non_review_task(self, qa_agent):
        task = _make_task(task_type=TaskType.IMPLEMENTATION)
        response = _make_response("REQUEST_CHANGES")
        qa_agent._queue_review_fix_if_needed(task, response)
        qa_agent.queue.push.assert_not_called()

    def test_skips_when_approved(self, qa_agent):
        task = _make_task()
        response = _make_response("APPROVE - all good")
        qa_agent._queue_review_fix_if_needed(task, response)
        qa_agent.queue.push.assert_not_called()

    def test_ambiguous_response_queues_fix(self, qa_agent, queue):
        """Ambiguous response (no approve, no issues) → treated as needs_fix."""
        task = _make_task()
        response = _make_response("I found some concerns but overall okay")
        qa_agent._queue_review_fix_if_needed(task, response)
        queue.push.assert_called_once()
        fix_task = queue.push.call_args[0][0]
        assert fix_task.type == TaskType.FIX

    def test_no_severity_no_approve_queues_fix(self, qa_agent, queue):
        """'12 passed, 0 failed' with no APPROVE → ambiguous guard queues fix."""
        task = _make_task()
        response = _make_response("Results: 12 passed, 0 failed")
        qa_agent._queue_review_fix_if_needed(task, response)
        queue.push.assert_called_once()
        fix_task = queue.push.call_args[0][0]
        assert fix_task.type == TaskType.FIX


# -- Happy path --

class TestQueueReviewFix:
    def test_queues_fix_task_on_request_changes(self, qa_agent, queue):
        task = _make_task()
        response = _make_response("REQUEST_CHANGES\nCRITICAL: missing validation")

        qa_agent._queue_review_fix_if_needed(task, response)

        queue.push.assert_called_once()
        fix_task = queue.push.call_args[0][0]
        target_queue = queue.push.call_args[0][1]

        assert target_queue == "engineer"
        assert fix_task.type == TaskType.FIX
        assert fix_task.context["_review_cycle_count"] == 1
        assert fix_task.context["pr_url"] == "https://github.com/org/repo/pull/99"
        assert "PROJ-42" in fix_task.title

    def test_increments_cycle_count(self, qa_agent, queue):
        task = _make_task(_review_cycle_count=2)
        response = _make_response("REQUEST_CHANGES\nFix the tests")

        qa_agent._queue_review_fix_if_needed(task, response)

        fix_task = queue.push.call_args[0][0]
        assert fix_task.context["_review_cycle_count"] == 3

    def test_escalates_after_max_cycles(self, qa_agent, queue):
        task = _make_task(_review_cycle_count=MAX_REVIEW_CYCLES)
        response = _make_response("REQUEST_CHANGES\nStill broken")

        qa_agent._queue_review_fix_if_needed(task, response)

        queue.push.assert_called_once()
        escalation_task = queue.push.call_args[0][0]
        target_queue = queue.push.call_args[0][1]

        assert target_queue == "architect"
        assert escalation_task.type == TaskType.ESCALATION
        assert "escalation_reason" in escalation_task.context

    def test_queue_push_error_is_caught(self, qa_agent, queue):
        task = _make_task()
        response = _make_response("REQUEST_CHANGES")
        queue.push.side_effect = OSError("disk full")

        # Should not raise
        qa_agent._queue_review_fix_if_needed(task, response)
        qa_agent.logger.error.assert_called()

    def test_skips_duplicate_fix_task(self, qa_agent, queue):
        """If the fix task file already exists, don't queue again."""
        task = _make_task()
        response = _make_response("REQUEST_CHANGES\nFix it")

        # Pre-create the fix task file — ID uses task.root_id (falls back to task.id)
        fix_id = f"review-fix-{task.root_id[:12]}-c1"
        engineer_dir = queue.queue_dir / "engineer"
        engineer_dir.mkdir()
        (engineer_dir / f"{fix_id}.json").write_text("{}")

        qa_agent._queue_review_fix_if_needed(task, response)

        queue.push.assert_not_called()


# -- _build_review_fix_task --

class TestBuildReviewFixTask:
    def test_task_fields(self, qa_agent):
        task = _make_task()
        outcome = ReviewOutcome(
            approved=False, has_critical_issues=True,
            has_test_failures=False, has_change_requests=True,
            findings_summary="CRITICAL: SQL injection",
        )
        fix_task = qa_agent._build_review_fix_task(task, outcome, cycle_count=1)

        assert fix_task.type == TaskType.FIX
        assert fix_task.assigned_to == "engineer"
        assert fix_task.context["_review_cycle_count"] == 1
        assert "SQL injection" in fix_task.description
        assert fix_task.context.get("review_mode") is None

    def test_strips_review_prefixed_context_keys(self, qa_agent):
        """All review_* keys should be stripped from fix task context."""
        task = _make_task(review_mode=True, review_comments="some comments")
        outcome = ReviewOutcome(
            approved=False, has_critical_issues=True,
            has_test_failures=False, has_change_requests=False,
            findings_summary="CRITICAL: issue",
        )
        fix_task = qa_agent._build_review_fix_task(task, outcome, cycle_count=1)

        assert "review_mode" not in fix_task.context
        assert "review_comments" not in fix_task.context

    def test_carries_forward_pr_context(self, qa_agent):
        task = _make_task()
        outcome = ReviewOutcome(
            approved=False, has_critical_issues=False,
            has_test_failures=True, has_change_requests=False,
            findings_summary="2 tests failed",
        )
        fix_task = qa_agent._build_review_fix_task(task, outcome, cycle_count=2)

        assert fix_task.context["pr_url"] == "https://github.com/org/repo/pull/99"
        assert fix_task.context["github_repo"] == "org/repo"
        assert fix_task.context["jira_key"] == "PROJ-42"


# -- Cycle count propagation in _build_review_task --

class TestReviewTaskCycleCountPropagation:
    def test_review_task_carries_cycle_count(self, engineer_agent):
        """When engineer creates a review task, _review_cycle_count flows through."""
        task = _make_task(
            task_type=TaskType.FIX,
            assigned_to="engineer",
            _review_cycle_count=2,
        )
        pr_info = {
            "pr_url": "https://github.com/org/repo/pull/99",
            "pr_number": 99,
            "owner": "org",
            "repo": "repo",
            "github_repo": "org/repo",
        }
        review_task = engineer_agent._build_review_task(task, pr_info)
        assert review_task.context["_review_cycle_count"] == 2

    def test_review_task_defaults_cycle_count_to_zero(self, engineer_agent):
        """First review has no cycle count yet — defaults to 0."""
        task = _make_task(task_type=TaskType.IMPLEMENTATION, assigned_to="engineer")
        pr_info = {
            "pr_url": "https://github.com/org/repo/pull/99",
            "pr_number": 99,
            "owner": "org",
            "repo": "repo",
            "github_repo": "org/repo",
        }
        review_task = engineer_agent._build_review_task(task, pr_info)
        assert review_task.context["_review_cycle_count"] == 0


# -- QA replica agent --

class TestQaReplicaAgent:
    def test_qa_replica_fires_review_fix(self, queue):
        """qa-2 (replica) should also trigger review fix logic."""
        from agent_framework.core.review_cycle import ReviewCycleManager
        config = AgentConfig(
            id="qa-2",
            name="QA Replica",
            queue="qa",
            prompt="You are QA.",
        )
        a = Agent.__new__(Agent)
        a.config = config
        a.queue = queue
        a.logger = MagicMock()
        a._review_cycle = ReviewCycleManager(
            config=config,
            queue=queue,
            logger=a.logger,
            agent_definition=None,
            session_logger=MagicMock(),
            activity_manager=MagicMock(),
        )

        task = _make_task()
        response = _make_response("REQUEST_CHANGES\nFix the bug")

        a._queue_review_fix_if_needed(task, response)

        queue.push.assert_called_once()
        assert queue.push.call_args[0][1] == "engineer"

    def test_qa_replica_skips_code_review_queue(self, queue):
        """qa-2 (replica) should skip _queue_code_review_if_needed just like qa."""
        from agent_framework.core.review_cycle import ReviewCycleManager
        config = AgentConfig(
            id="qa-2",
            name="QA Replica",
            queue="qa",
            prompt="You are QA.",
        )
        a = Agent.__new__(Agent)
        a.config = config
        a.queue = queue
        a.logger = MagicMock()
        a._review_cycle = ReviewCycleManager(
            config=config,
            queue=queue,
            logger=a.logger,
            agent_definition=None,
            session_logger=MagicMock(),
            activity_manager=MagicMock(),
        )

        task = _make_task(task_type=TaskType.IMPLEMENTATION, assigned_to="qa-2")
        response = _make_response(
            "Created PR: https://github.com/org/repo/pull/99"
        )

        a._queue_code_review_if_needed(task, response)

        # qa-2 should be treated as QA and not queue a review to itself
        queue.push.assert_not_called()


# -- Negation-aware pattern matching --

class TestNegatedPatterns:
    """Negated phrases like 'No test failures' must not trigger needs_fix."""

    @pytest.mark.parametrize("phrase", [
        "No test failures found.",
        "All tests passed, no test failures.",
        "0 test failures detected.",
        "Completed without test failures.",
        "There are not test failures in this run.",
        "Zero test failures.",
    ])
    def test_negated_test_failures_do_not_trigger(self, qa_agent, phrase):
        outcome = qa_agent._parse_review_outcome(f"APPROVE\n{phrase}")
        assert outcome.has_test_failures is False, f"False positive for: {phrase}"
        assert outcome.needs_fix is False

    @pytest.mark.parametrize("phrase", [
        "3 tests failed",
        "test failure in auth module",
        "Tests fail on CI",
    ])
    def test_real_test_failures_still_trigger(self, qa_agent, phrase):
        outcome = qa_agent._parse_review_outcome(phrase)
        assert outcome.has_test_failures is True, f"Missed real failure: {phrase}"

    @pytest.mark.parametrize("phrase", [
        "No CRITICAL issues found.",
        "Zero CRITICAL: problems detected.",
    ])
    def test_negated_critical_issues_do_not_trigger(self, qa_agent, phrase):
        outcome = qa_agent._parse_review_outcome(f"APPROVE\n{phrase}")
        assert outcome.has_critical_issues is False, f"False positive for: {phrase}"

    @pytest.mark.parametrize("phrase", [
        "No MAJOR issues found.",
        "Zero HIGH: problems detected.",
    ])
    def test_negated_major_issues_do_not_trigger(self, qa_agent, phrase):
        outcome = qa_agent._parse_review_outcome(f"APPROVE\n{phrase}")
        assert outcome.has_major_issues is False, f"False positive for: {phrase}"


# -- ESCALATION guard in _queue_code_review_if_needed --

class TestEscalationGuard:
    """Architect completing an ESCALATION task must not trigger a new QA review."""

    @pytest.fixture
    def architect_agent(self, queue):
        from agent_framework.core.review_cycle import ReviewCycleManager
        config = AgentConfig(
            id="architect",
            name="Architect",
            queue="architect",
            prompt="You are the architect.",
        )
        a = Agent.__new__(Agent)
        a.config = config
        a.queue = queue
        a.logger = MagicMock()
        a._review_cycle = ReviewCycleManager(
            config=config,
            queue=queue,
            logger=a.logger,
            agent_definition=None,
            session_logger=MagicMock(),
            activity_manager=MagicMock(),
        )
        return a

    def test_escalation_task_skips_code_review_queue(self, architect_agent):
        task = _make_task(
            task_type=TaskType.ESCALATION,
            assigned_to="architect",
        )
        response = _make_response(
            "Replanned implementation. Created PR: https://github.com/org/repo/pull/100"
        )

        architect_agent._queue_code_review_if_needed(task, response)

        architect_agent.queue.push.assert_not_called()

    def test_review_task_still_skips_code_review_queue(self, architect_agent):
        """Existing REVIEW guard still works alongside ESCALATION guard."""
        task = _make_task(task_type=TaskType.REVIEW, assigned_to="architect")
        response = _make_response("APPROVE")

        architect_agent._queue_code_review_if_needed(task, response)

        architect_agent.queue.push.assert_not_called()

    def test_fix_task_at_max_cycles_skips_review(self, engineer_agent):
        """FIX task that already hit the escalation cap must not spawn another review."""
        task = _make_task(
            task_type=TaskType.FIX,
            assigned_to="engineer",
            _review_cycle_count=MAX_REVIEW_CYCLES,
        )
        response = _make_response(
            "Fixed. Created PR: https://github.com/org/repo/pull/99"
        )

        engineer_agent._queue_code_review_if_needed(task, response)

        engineer_agent.queue.push.assert_not_called()

    def test_fix_task_below_max_cycles_queues_review(self, engineer_agent, queue):
        """FIX task below the cap should still get its review queued."""
        task = _make_task(
            task_type=TaskType.FIX,
            assigned_to="engineer",
            _review_cycle_count=1,
        )
        response = _make_response(
            "Fixed. Created PR: https://github.com/org/repo/pull/99"
        )

        engineer_agent._queue_code_review_if_needed(task, response)

        queue.push.assert_called_once()
        review_task = queue.push.call_args[0][0]
        assert queue.push.call_args[0][1] == "qa"
        assert review_task.type == TaskType.REVIEW


# -- Dedup guard in _queue_code_review_if_needed --

class TestCodeReviewDedup:
    """Duplicate review tasks must not be queued."""

    def test_duplicate_review_task_not_queued(self, engineer_agent, queue):
        """Pre-existing review task file prevents a second push."""
        task = _make_task(
            task_type=TaskType.FIX,
            assigned_to="engineer",
            _review_cycle_count=1,
        )
        response = _make_response(
            "Fixed. Created PR: https://github.com/org/repo/pull/99"
        )

        # Pre-create the review task file so the dedup check fires
        review_id = f"review-{task.id}-99"
        qa_dir = queue.queue_dir / "qa"
        qa_dir.mkdir(exist_ok=True)
        (qa_dir / f"{review_id}.json").write_text("{}")

        engineer_agent._queue_code_review_if_needed(task, response)

        queue.push.assert_not_called()


# -- Startup purge of orphaned review-chain tasks --

class TestPurgeOrphanedReviewTasks:
    """_purge_orphaned_review_tasks removes REVIEW/FIX tasks for escalated PRs."""

    @staticmethod
    def _write_task(path, task_type, pr_url):
        """Write a minimal task JSON file."""
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "type": task_type,
            "context": {"pr_url": pr_url},
        }))

    @pytest.fixture
    def purge_agent(self, tmp_path):
        """Agent wired to a real tmp directory structure."""
        q = MagicMock()
        q.queue_dir = tmp_path / "queues"
        q.queue_dir.mkdir()
        q.completed_dir = tmp_path / "completed"
        q.completed_dir.mkdir()

        config = AgentConfig(
            id="engineer",
            name="Engineer",
            queue="engineer",
            prompt="You are an engineer.",
        )
        a = Agent.__new__(Agent)
        a.config = config
        a.queue = q
        a.logger = MagicMock()
        from agent_framework.core.review_cycle import ReviewCycleManager
        a._review_cycle = ReviewCycleManager(
            config=config,
            queue=q,
            logger=a.logger,
            agent_definition=None,
            session_logger=MagicMock(),
            activity_manager=MagicMock(),
        )
        return a

    def test_purges_review_task_for_escalated_pr(self, purge_agent):
        pr = "https://github.com/org/repo/pull/99"
        # Escalation in architect queue
        self._write_task(
            purge_agent.queue.queue_dir / "architect" / "esc-1.json",
            "escalation", pr,
        )
        # Orphaned REVIEW in qa queue
        review_file = purge_agent.queue.queue_dir / "qa" / "review-1.json"
        self._write_task(review_file, "review", pr)

        purge_agent._purge_orphaned_review_tasks()

        assert not review_file.exists()

    def test_purges_fix_task_for_escalated_pr(self, purge_agent):
        pr = "https://github.com/org/repo/pull/99"
        # Escalation in completed dir
        self._write_task(
            purge_agent.queue.completed_dir / "esc-2.json",
            "escalation", pr,
        )
        # Orphaned FIX in engineer queue
        fix_file = purge_agent.queue.queue_dir / "engineer" / "fix-1.json"
        self._write_task(fix_file, "fix", pr)

        purge_agent._purge_orphaned_review_tasks()

        assert not fix_file.exists()

    def test_keeps_tasks_for_non_escalated_pr(self, purge_agent):
        escalated_pr = "https://github.com/org/repo/pull/99"
        other_pr = "https://github.com/org/repo/pull/100"
        # Escalation only for PR 99
        self._write_task(
            purge_agent.queue.queue_dir / "architect" / "esc-1.json",
            "escalation", escalated_pr,
        )
        # REVIEW for a different PR — should survive
        review_file = purge_agent.queue.queue_dir / "qa" / "review-other.json"
        self._write_task(review_file, "review", other_pr)

        purge_agent._purge_orphaned_review_tasks()

        assert review_file.exists()

    def test_noop_when_no_escalations(self, purge_agent):
        # REVIEW task with no matching escalation — untouched
        review_file = purge_agent.queue.queue_dir / "qa" / "review-1.json"
        self._write_task(review_file, "review", "https://github.com/org/repo/pull/99")

        purge_agent._purge_orphaned_review_tasks()

        assert review_file.exists()
        purge_agent.logger.info.assert_not_called()


# -- Structured JSON findings --

class TestStructuredJsonFindings:
    """Test JSON parsing of structured QA findings."""

    def test_parse_structured_json_findings(self, qa_agent):
        """Parse JSON block with valid structured findings."""
        content = """REQUEST_CHANGES

```json
[
  {
    "file": "src/api/auth.ts",
    "line_number": 45,
    "severity": "CRITICAL",
    "description": "SQL injection vulnerability in login handler",
    "suggested_fix": "Use parameterized queries with prepared statements",
    "category": "security"
  },
  {
    "file": "src/db/query.ts",
    "line_number": 89,
    "severity": "HIGH",
    "description": "N+1 query detected in user data fetch",
    "suggested_fix": "Add eager loading or batch query",
    "category": "performance"
  }
]
```"""
        outcome = qa_agent._parse_review_outcome(content)

        assert outcome.has_change_requests is True
        assert outcome.needs_fix is True
        assert len(outcome.structured_findings) == 2

        # Check first finding
        finding1 = outcome.structured_findings[0]
        assert finding1.file == "src/api/auth.ts"
        assert finding1.line_number == 45
        assert finding1.severity == "CRITICAL"
        assert finding1.description == "SQL injection vulnerability in login handler"
        assert finding1.suggested_fix == "Use parameterized queries with prepared statements"
        assert finding1.category == "security"

        # Check second finding
        finding2 = outcome.structured_findings[1]
        assert finding2.file == "src/db/query.ts"
        assert finding2.line_number == 89
        assert finding2.severity == "HIGH"
        assert finding2.category == "performance"

    def test_json_parsing_fallback_to_regex(self, qa_agent):
        """When JSON parsing fails, fall back to regex extraction."""
        content = """REQUEST_CHANGES

CRITICAL: SQL injection in src/api/auth.ts:45
HIGH: N+1 query in src/db/query.ts:89
"""
        outcome = qa_agent._parse_review_outcome(content)

        assert outcome.has_critical_issues is True
        assert outcome.needs_fix is True
        # Regex fallback doesn't produce structured findings
        assert len(outcome.structured_findings) == 0
        # But findings_summary should have text
        assert "CRITICAL: SQL injection" in outcome.findings_summary
        assert "HIGH: N+1 query" in outcome.findings_summary

    def test_json_with_optional_fields_missing(self, qa_agent):
        """Handle JSON findings with optional fields (line_number, suggested_fix) missing."""
        content = """REQUEST_CHANGES

```json
[
  {
    "file": "src/utils.py",
    "severity": "MINOR",
    "description": "Unused import",
    "category": "readability"
  }
]
```"""
        outcome = qa_agent._parse_review_outcome(content)

        assert len(outcome.structured_findings) == 1
        finding = outcome.structured_findings[0]
        assert finding.file == "src/utils.py"
        assert finding.line_number is None
        assert finding.suggested_fix is None
        assert finding.severity == "MINOR"

    def test_malformed_json_does_not_crash(self, qa_agent):
        """Malformed JSON should be caught and fallback to regex."""
        content = """REQUEST_CHANGES

```json
{
  "invalid": "not an array"
}
```

CRITICAL: Some issue
"""
        outcome = qa_agent._parse_review_outcome(content)

        # Should not crash, falls back to regex
        assert outcome.has_critical_issues is True
        assert outcome.needs_fix is True
        # No structured findings because JSON was invalid
        assert len(outcome.structured_findings) == 0

    def test_empty_json_array(self, qa_agent):
        """Empty JSON array with APPROVE should pass."""
        content = """APPROVE

```json
[]
```"""
        outcome = qa_agent._parse_review_outcome(content)

        assert outcome.approved is True
        assert outcome.needs_fix is False
        assert len(outcome.structured_findings) == 0


# -- Numbered checklist generation --

class TestNumberedChecklistFormatting:
    """Test numbered checklist generation in fix task description."""

    def test_numbered_checklist_formatting(self, qa_agent):
        """Fix task should format structured findings as numbered checklist."""
        task = _make_task()
        findings = [
            QAFinding(
                file="src/api/auth.ts",
                line_number=45,
                severity="CRITICAL",
                description="SQL injection vulnerability in login handler",
                suggested_fix="Use parameterized queries with prepared statements",
                category="security"
            ),
            QAFinding(
                file="src/db/query.ts",
                line_number=89,
                severity="HIGH",
                description="N+1 query detected in user data fetch",
                suggested_fix="Add eager loading or batch query",
                category="performance"
            ),
        ]
        outcome = ReviewOutcome(
            approved=False, has_critical_issues=True,
            has_test_failures=False, has_change_requests=True,
            findings_summary="See structured findings",
            structured_findings=findings,
        )

        fix_task = qa_agent._build_review_fix_task(task, outcome, cycle_count=1)

        # Check numbered checklist format (using new formatter with emoji/markdown)
        assert "### 1. 🔴 CRITICAL: Security (src/api/auth.ts:45)" in fix_task.description
        assert "**Issue**: SQL injection vulnerability in login handler" in fix_task.description
        assert "**Suggested Fix**: Use parameterized queries with prepared statements" in fix_task.description

        assert "### 2. 🟠 HIGH: Performance (src/db/query.ts:89)" in fix_task.description
        assert "**Issue**: N+1 query detected in user data fetch" in fix_task.description
        assert "**Suggested Fix**: Add eager loading or batch query" in fix_task.description

    def test_checklist_with_optional_fields(self, qa_agent):
        """Checklist should handle optional fields gracefully."""
        task = _make_task()
        findings = [
            QAFinding(
                file="src/utils.py",
                line_number=None,  # No line number
                severity="MINOR",
                description="Unused import",
                suggested_fix=None,  # No suggested fix
                category="readability"
            ),
        ]
        outcome = ReviewOutcome(
            approved=False, has_critical_issues=False,
            has_test_failures=False, has_change_requests=True,
            findings_summary="See structured findings",
            structured_findings=findings,
        )

        fix_task = qa_agent._build_review_fix_task(task, outcome, cycle_count=1)

        # Check format without line number (using new formatter)
        assert "### 1. ⚪ MINOR: Readability (src/utils.py)" in fix_task.description
        assert "**Issue**: Unused import" in fix_task.description
        # Should not have Suggested Fix: line since suggested_fix is None
        assert "**Suggested Fix**:" not in fix_task.description

    def test_backward_compatibility_regex_only(self, qa_agent):
        """When no structured findings, fall back to legacy text format."""
        task = _make_task()
        outcome = ReviewOutcome(
            approved=False, has_critical_issues=True,
            has_test_failures=False, has_change_requests=False,
            findings_summary="CRITICAL: SQL injection in src/api/auth.ts:45\nHIGH: N+1 query",
            structured_findings=[],  # Empty structured findings
        )

        fix_task = qa_agent._build_review_fix_task(task, outcome, cycle_count=1)

        # Should use legacy findings_summary
        assert "CRITICAL: SQL injection in src/api/auth.ts:45" in fix_task.description
        assert "HIGH: N+1 query" in fix_task.description
        # Should NOT have numbered checklist format (no emoji headers)
        assert "###" not in fix_task.description
        assert "🔴" not in fix_task.description


# -- QAFinding dataclass --

class TestQAFindingDataclass:
    """Test QAFinding dataclass structure."""

    def test_create_qa_finding_with_all_fields(self):
        """Create QAFinding with all fields populated."""
        finding = QAFinding(
            file="src/api/auth.ts",
            line_number=45,
            severity="CRITICAL",
            description="SQL injection vulnerability",
            suggested_fix="Use parameterized queries",
            category="security"
        )

        assert finding.file == "src/api/auth.ts"
        assert finding.line_number == 45
        assert finding.severity == "CRITICAL"
        assert finding.description == "SQL injection vulnerability"
        assert finding.suggested_fix == "Use parameterized queries"
        assert finding.category == "security"

    def test_create_qa_finding_with_optional_fields_none(self):
        """Create QAFinding with optional fields as None."""
        finding = QAFinding(
            file="src/utils.py",
            line_number=None,
            severity="MINOR",
            description="Unused import",
            suggested_fix=None,
            category="readability"
        )

        assert finding.file == "src/utils.py"
        assert finding.line_number is None
        assert finding.suggested_fix is None


class TestEscalationCarriesStructuredFindings:
    """Verify escalation to architect preserves structured findings."""

    def test_escalation_carries_structured_findings(self, qa_agent, queue):
        from agent_framework.core.review_cycle import ReviewOutcome

        findings_data = {
            "findings": [
                {"file": "src/x.py", "line_number": 1, "severity": "CRITICAL",
                 "description": "bug", "suggested_fix": None, "category": "correctness"}
            ],
            "total_count": 1,
            "critical_count": 1,
        }
        task = _make_task(
            _review_cycle_count=3,
            structured_findings=findings_data,
        )
        outcome = ReviewOutcome(
            approved=False, has_critical_issues=True,
            has_test_failures=False, has_change_requests=False,
            findings_summary="CRITICAL: bug",
        )

        qa_agent._review_cycle.escalate_review_to_architect(task, outcome, cycle_count=4)

        pushed_task = queue.push.call_args[0][0]
        assert "structured_findings" in pushed_task.context
        assert pushed_task.context["structured_findings"] == findings_data

    def test_escalation_without_structured_findings(self, qa_agent, queue):
        from agent_framework.core.review_cycle import ReviewOutcome

        task = _make_task(_review_cycle_count=3)
        outcome = ReviewOutcome(
            approved=False, has_critical_issues=True,
            has_test_failures=False, has_change_requests=False,
            findings_summary="CRITICAL: bug",
        )

        qa_agent._review_cycle.escalate_review_to_architect(task, outcome, cycle_count=4)

        pushed_task = queue.push.call_args[0][0]
        assert "structured_findings" not in pushed_task.context


# -- Chain ID stability for review and fix task IDs --

class TestChainIdStability:
    """Verify that review/fix task IDs don't nest 'chain-' on each cycle."""

    def test_review_task_id_uses_root_task_id(self, engineer_agent):
        """build_review_task uses _root_task_id so the review ID stays flat."""
        task = _make_task(
            task_type=TaskType.IMPLEMENTATION,
            task_id="chain-chain-original-engineer-1",
        )
        task.context["_root_task_id"] = "original"
        pr_info = {
            "pr_number": 42,
            "pr_url": "https://github.com/org/repo/pull/42",
            "owner": "org",
            "repo": "repo",
            "github_repo": "org/repo",
        }

        review_task = engineer_agent._build_review_task(task, pr_info)

        assert "chain-" not in review_task.id
        assert review_task.id.startswith("review-original-")

    def test_review_task_title_strips_chain_prefix(self, engineer_agent):
        """build_review_task strips [chain] markers from the title."""
        task = _make_task(
            task_type=TaskType.IMPLEMENTATION,
            task_id="chain-chain-original-engineer-1",
        )
        task.context["_root_task_id"] = "original"
        task.title = "[chain] [chain] Fix auth bug"
        pr_info = {
            "pr_number": 7,
            "pr_url": "https://github.com/org/repo/pull/7",
            "owner": "org",
            "repo": "repo",
            "github_repo": "org/repo",
        }

        review_task = engineer_agent._build_review_task(task, pr_info)

        assert "[chain]" not in review_task.title

    def test_review_fix_id_uses_root_task_id(self, qa_agent):
        """build_review_fix_task uses _root_task_id so the fix ID stays flat."""
        from agent_framework.core.review_cycle import ReviewOutcome

        task = _make_task(
            task_id="chain-chain-rootid-qa-1",
        )
        task.context["_root_task_id"] = "rootid"
        outcome = ReviewOutcome(
            approved=False, has_critical_issues=True,
            has_test_failures=False, has_change_requests=True,
            findings_summary="CRITICAL: bug",
        )

        fix_task = qa_agent._build_review_fix_task(task, outcome, cycle_count=1)

        # ID is anchored on "rootid", not the nested chain task ID
        assert fix_task.id.startswith("review-fix-rootid")
        assert "chain" not in fix_task.id
