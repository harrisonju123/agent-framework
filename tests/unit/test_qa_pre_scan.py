"""Tests for QA pre-scan feature: parallel lightweight QA during code review."""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace, MappingProxyType
from unittest.mock import MagicMock, patch

import pytest

from agent_framework.core.agent import Agent, AgentConfig
from agent_framework.core.prompt_builder import PromptBuilder, PromptContext
from agent_framework.core.task import Task, TaskStatus, TaskType
from agent_framework.workflow.executor import WorkflowExecutor


# -- Helpers --

def _make_task(task_id="task-abc123def456", workflow="default", **ctx_overrides):
    context = {"workflow": workflow, **ctx_overrides}
    return Task(
        id=task_id,
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.COMPLETED,
        priority=1,
        created_by="architect",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Implement feature X",
        description="Build the thing.",
        context=context,
    )


def _make_response(content="Done."):
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
    q.completed_dir = tmp_path / "completed"
    q.completed_dir.mkdir()
    return q


@pytest.fixture
def executor(queue, tmp_path):
    return WorkflowExecutor(queue, queue.queue_dir)


@pytest.fixture
def agent(queue, tmp_path):
    """Minimal Agent for testing post-completion flow."""
    config = AgentConfig(
        id="qa",
        name="QA Engineer",
        queue="qa",
        prompt="You are qa.",
    )
    a = Agent.__new__(Agent)
    a.config = config
    a.queue = queue
    a.workspace = tmp_path
    a.logger = MagicMock()
    a._session_logger = MagicMock()
    a._memory_enabled = False
    a._memory_store = MagicMock()
    a._tool_tips_enabled = False
    a._tool_pattern_store = MagicMock()
    a._optimization_config = MappingProxyType({"enable_effort_budget_ceilings": False})
    a._budget = MagicMock()
    a._budget.estimate_cost.return_value = 0.0

    from agent_framework.core.review_cycle import ReviewCycleManager
    a._review_cycle = ReviewCycleManager(
        config=config, queue=queue, logger=a.logger,
        agent_definition=None, session_logger=a._session_logger,
        activity_manager=MagicMock(),
    )

    from agent_framework.core.git_operations import GitOperationsManager
    a._git_ops = GitOperationsManager(
        config=config, workspace=tmp_path, queue=queue,
        logger=a.logger, session_logger=a._session_logger,
    )

    from agent_framework.core.workflow_router import WorkflowRouter
    a._workflows_config = {}
    a._workflow_executor = WorkflowExecutor(queue, queue.queue_dir)
    a._workflow_router = WorkflowRouter(
        config=config, queue=queue, workspace=tmp_path,
        logger=a.logger, session_logger=a._session_logger,
        workflows_config={}, workflow_executor=a._workflow_executor,
        agents_config=[], multi_repo_manager=None,
    )
    return a


# -- TestQueueQAPreScan --

class TestQueueQAPreScan:
    """Tests for _queue_qa_pre_scan() in WorkflowExecutor."""

    def test_prescan_queued_on_implement_to_code_review(self, executor, queue, tmp_path):
        """Pre-scan task queued when implement→code_review transition fires."""
        task = _make_task(
            workflow_step="implement",
            implementation_branch="feature/test-branch",
            _root_task_id="root-123",
        )

        executor._queue_qa_pre_scan(task)

        queue.push.assert_called_once()
        pushed_task = queue.push.call_args[0][0]
        pushed_agent = queue.push.call_args[0][1]

        assert pushed_agent == "qa"
        assert pushed_task.id == "prescan-root-123"
        assert pushed_task.type == TaskType.QA_VERIFICATION
        assert pushed_task.context["pre_scan"] is True
        assert "[pre-scan]" in pushed_task.title

    def test_prescan_skipped_without_implementation_branch(self, executor, queue, tmp_path):
        """No pre-scan when implementation_branch is absent."""
        task = _make_task(
            workflow_step="implement",
            _root_task_id="root-123",
        )

        executor._queue_qa_pre_scan(task)

        queue.push.assert_not_called()

    def test_prescan_dedup_by_queue_file(self, executor, queue, tmp_path):
        """Dedup: skip if prescan-{root_task_id}.json already in QA queue."""
        task = _make_task(
            workflow_step="implement",
            implementation_branch="feature/branch",
            _root_task_id="root-dup",
        )
        # Create existing queue file
        qa_dir = queue.queue_dir / "qa"
        qa_dir.mkdir()
        (qa_dir / "prescan-root-dup.json").write_text("{}")

        executor._queue_qa_pre_scan(task)

        queue.push.assert_not_called()

    def test_prescan_dedup_by_completed_file(self, executor, queue, tmp_path):
        """Dedup: skip if prescan already completed."""
        task = _make_task(
            workflow_step="implement",
            implementation_branch="feature/branch",
            _root_task_id="root-done",
        )
        (queue.completed_dir / "prescan-root-done.json").write_text("{}")

        executor._queue_qa_pre_scan(task)

        queue.push.assert_not_called()

    def test_prescan_queue_failure_nonfatal(self, executor, queue, tmp_path):
        """Queue push failure is logged as warning, does not raise."""
        task = _make_task(
            workflow_step="implement",
            implementation_branch="feature/branch",
            _root_task_id="root-fail",
        )
        queue.push.side_effect = RuntimeError("queue full")

        # Should not raise — failure is non-fatal
        executor._queue_qa_pre_scan(task)
        # Verify push was attempted and failed gracefully
        queue.push.assert_called_once()

    def test_prescan_context_has_correct_fields(self, executor, queue, tmp_path):
        """Pre-scan task context includes essential fields and pre_scan flag."""
        task = _make_task(
            workflow_step="implement",
            implementation_branch="feature/test",
            _root_task_id="root-ctx",
            github_repo="org/repo",
            jira_key="PROJ-42",
        )

        executor._queue_qa_pre_scan(task)

        pushed_task = queue.push.call_args[0][0]
        ctx = pushed_task.context
        assert ctx["pre_scan"] is True
        assert ctx["_root_task_id"] == "root-ctx"
        assert ctx["github_repo"] == "org/repo"
        assert ctx["implementation_branch"] == "feature/test"


# -- TestPreScanPostCompletion --

class TestPreScanPostCompletion:
    """Tests for pre-scan early return in _run_post_completion_flow()."""

    def test_prescan_early_return_skips_workflow_chain(self, agent):
        """Pre-scan tasks skip enforce_chain, legacy review, and PR creation."""
        task = _make_task(pre_scan=True, _root_task_id="root-prescan")
        task.status = TaskStatus.COMPLETED
        response = _make_response("APPROVE\n```json\n{\"findings\": []}\n```")

        agent._run_post_completion_flow = Agent._run_post_completion_flow.__get__(agent)
        agent._save_pre_scan_findings = MagicMock()
        agent._extract_and_store_memories = MagicMock()
        agent._analyze_tool_patterns = MagicMock()
        agent._log_task_completion_metrics = MagicMock()

        agent._run_post_completion_flow(task, response, None, datetime.now(timezone.utc))

        agent._save_pre_scan_findings.assert_called_once_with(task, response)
        agent._extract_and_store_memories.assert_called_once()
        agent._analyze_tool_patterns.assert_called_once()
        agent._log_task_completion_metrics.assert_called_once()

    def test_prescan_does_not_trigger_fan_in(self, agent):
        """Pre-scan tasks must not trigger fan-in check."""
        task = _make_task(pre_scan=True, _root_task_id="root-nofanin")
        task.status = TaskStatus.COMPLETED
        response = _make_response()

        agent._run_post_completion_flow = Agent._run_post_completion_flow.__get__(agent)
        agent._save_pre_scan_findings = MagicMock()
        agent._extract_and_store_memories = MagicMock()
        agent._analyze_tool_patterns = MagicMock()
        agent._log_task_completion_metrics = MagicMock()
        agent._workflow_router = MagicMock()

        agent._run_post_completion_flow(task, response, None, datetime.now(timezone.utc))

        agent._workflow_router.check_and_create_fan_in_task.assert_not_called()


# -- TestPreScanFindingsPersistence --

class TestPreScanFindingsPersistence:
    """Tests for _save_pre_scan_findings() and _extract_structured_findings_from_content()."""

    def test_findings_saved_to_correct_path(self, agent, tmp_path):
        """Findings file created at .agent-communication/pre-scans/{root_task_id}.json."""
        agent._save_pre_scan_findings = Agent._save_pre_scan_findings.__get__(agent)
        agent._extract_structured_findings_from_content = Agent._extract_structured_findings_from_content

        task = _make_task(pre_scan=True, _root_task_id="root-save")
        response = _make_response("All good\n```json\n{\"findings\": []}\n```")

        agent._save_pre_scan_findings(task, response)

        findings_file = tmp_path / ".agent-communication" / "pre-scans" / "root-save.json"
        assert findings_file.exists()

        data = json.loads(findings_file.read_text())
        assert data["root_task_id"] == "root-save"
        assert "timestamp" in data
        assert "structured_findings" in data

    def test_findings_parsed_from_json_block(self, agent, tmp_path):
        """Structured findings extracted from ```json block in response."""
        agent._save_pre_scan_findings = Agent._save_pre_scan_findings.__get__(agent)
        agent._extract_structured_findings_from_content = Agent._extract_structured_findings_from_content

        findings_json = json.dumps({
            "findings": [
                {"severity": "HIGH", "file": "src/auth.py", "line": 42,
                 "description": "SQL injection", "suggested_fix": "Use params"}
            ],
            "summary": "1 issue found"
        })
        content = f"Review done.\n```json\n{findings_json}\n```\nDone."
        task = _make_task(pre_scan=True, _root_task_id="root-parse")
        response = _make_response(content)

        agent._save_pre_scan_findings(task, response)

        findings_file = tmp_path / ".agent-communication" / "pre-scans" / "root-parse.json"
        data = json.loads(findings_file.read_text())
        sf = data["structured_findings"]
        assert len(sf["findings"]) == 1
        assert sf["findings"][0]["severity"] == "HIGH"

    def test_findings_raw_list_parsed(self):
        """Raw JSON list (without wrapping dict) is normalized to findings format."""
        content = '```json\n[{"severity": "LOW", "description": "unused import"}]\n```'
        result = Agent._extract_structured_findings_from_content(content)
        assert result["findings"][0]["severity"] == "LOW"

    def test_empty_content_returns_empty(self):
        """Empty or missing content returns empty dict."""
        assert Agent._extract_structured_findings_from_content("") == {}
        assert Agent._extract_structured_findings_from_content("no json here") == {}

    def test_invalid_json_returns_empty(self):
        """Malformed JSON block returns empty dict."""
        content = '```json\n{invalid json}\n```'
        assert Agent._extract_structured_findings_from_content(content) == {}

    def test_save_failure_nonfatal(self, agent, tmp_path):
        """Failure to write findings file is logged as warning, doesn't raise."""
        agent._save_pre_scan_findings = Agent._save_pre_scan_findings.__get__(agent)
        agent._extract_structured_findings_from_content = Agent._extract_structured_findings_from_content

        task = _make_task(pre_scan=True, _root_task_id="root-fail")
        response = _make_response("done")

        # Make the directory creation fail
        with patch.object(Path, "mkdir", side_effect=OSError("permission denied")):
            agent._save_pre_scan_findings(task, response)

        agent.logger.warning.assert_called()

    def test_raw_summary_truncated(self, agent, tmp_path):
        """Raw summary is truncated to 4000 chars."""
        agent._save_pre_scan_findings = Agent._save_pre_scan_findings.__get__(agent)
        agent._extract_structured_findings_from_content = Agent._extract_structured_findings_from_content

        task = _make_task(pre_scan=True, _root_task_id="root-trunc")
        long_content = "x" * 10000
        response = _make_response(long_content)

        agent._save_pre_scan_findings(task, response)

        findings_file = tmp_path / ".agent-communication" / "pre-scans" / "root-trunc.json"
        data = json.loads(findings_file.read_text())
        assert len(data["raw_summary"]) == 4000


# -- TestPreScanPromptInjection --

class TestPreScanPromptInjection:
    """Tests for _load_pre_scan_findings() in PromptBuilder."""

    def _make_builder(self, tmp_path, agent_id="engineer"):
        config = AgentConfig(
            id=agent_id, name=agent_id.title(),
            queue=agent_id, prompt=f"You are {agent_id}.",
        )
        ctx = PromptContext(
            config=config, workspace=tmp_path,
            mcp_enabled=False, optimization_config={},
        )
        return PromptBuilder(ctx)

    def _write_prescan(self, tmp_path, root_task_id, findings=None, raw_summary=""):
        pre_scans_dir = tmp_path / ".agent-communication" / "pre-scans"
        pre_scans_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "root_task_id": root_task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_summary": raw_summary,
            "structured_findings": {"findings": findings or [], "summary": ""},
        }
        (pre_scans_dir / f"{root_task_id}.json").write_text(json.dumps(data))

    def test_engineer_sees_prescan_findings(self, tmp_path):
        """Engineer prompt includes pre-scan findings."""
        builder = self._make_builder(tmp_path, "engineer")
        self._write_prescan(tmp_path, "root-eng", findings=[
            {"severity": "HIGH", "file": "src/auth.py", "line": 10,
             "description": "SQL injection"}
        ])

        task = _make_task(
            workflow_step="implement", _root_task_id="root-eng",
        )
        result = builder._load_pre_scan_findings(task)

        assert "PRE-SCAN FINDINGS" in result
        assert "SQL injection" in result
        assert "src/auth.py:10" in result

    def test_qa_sees_prescan_results(self, tmp_path):
        """QA prompt includes pre-scan results with staleness note."""
        builder = self._make_builder(tmp_path, "qa")
        self._write_prescan(tmp_path, "root-qa", findings=[
            {"severity": "LOW", "file": "src/main.py",
             "description": "unused import"}
        ])

        task = _make_task(
            workflow_step="qa_review", _root_task_id="root-qa",
        )
        result = builder._load_pre_scan_findings(task)

        assert "PRE-SCAN RESULTS" in result
        assert "stale" in result
        assert "unused import" in result

    def test_architect_does_not_see_prescan(self, tmp_path):
        """Architect prompt does not include pre-scan findings."""
        builder = self._make_builder(tmp_path, "architect")
        self._write_prescan(tmp_path, "root-arch", findings=[
            {"severity": "HIGH", "file": "f.py", "description": "issue"}
        ])

        task = _make_task(
            workflow_step="code_review", _root_task_id="root-arch",
        )
        result = builder._load_pre_scan_findings(task)

        assert result == ""

    def test_missing_file_returns_empty(self, tmp_path):
        """Missing pre-scan file returns empty string."""
        builder = self._make_builder(tmp_path, "engineer")
        task = _make_task(
            workflow_step="implement", _root_task_id="root-missing",
        )
        result = builder._load_pre_scan_findings(task)
        assert result == ""

    def test_corrupt_json_returns_empty(self, tmp_path):
        """Corrupt pre-scan JSON file returns empty string."""
        builder = self._make_builder(tmp_path, "engineer")
        pre_scans_dir = tmp_path / ".agent-communication" / "pre-scans"
        pre_scans_dir.mkdir(parents=True, exist_ok=True)
        (pre_scans_dir / "root-corrupt.json").write_text("{invalid")

        task = _make_task(
            workflow_step="implement", _root_task_id="root-corrupt",
        )
        result = builder._load_pre_scan_findings(task)
        assert result == ""

    def test_no_workflow_step_returns_empty(self, tmp_path):
        """Tasks without workflow_step don't get pre-scan findings."""
        builder = self._make_builder(tmp_path, "engineer")
        self._write_prescan(tmp_path, "root-nowf", findings=[
            {"severity": "HIGH", "file": "f.py", "description": "issue"}
        ])

        task = _make_task(_root_task_id="root-nowf")
        # Remove workflow_step
        task.context.pop("workflow_step", None)
        result = builder._load_pre_scan_findings(task)
        assert result == ""

    def test_empty_findings_with_raw_summary_uses_summary(self, tmp_path):
        """When findings list is empty but raw_summary exists, it's included."""
        builder = self._make_builder(tmp_path, "engineer")
        self._write_prescan(
            tmp_path, "root-raw",
            findings=[],
            raw_summary="All tests passed. 2 lint warnings found.",
        )

        task = _make_task(
            workflow_step="implement", _root_task_id="root-raw",
        )
        result = builder._load_pre_scan_findings(task)

        assert "All tests passed" in result

    def test_prescan_constraint_in_qa_agent_prompt(self, tmp_path):
        """Pre-scan mode instructions present in QA agent prompt (agents.yaml)."""
        # The pre-scan constraint lives in the QA agent's base prompt (agents.yaml),
        # not in dynamic prompt injection, to avoid duplicate tokens.
        qa_prompt = "PRE-SCAN MODE:\nWhen context.pre_scan is true:\n- Run ONLY lightweight checks"
        builder = self._make_builder(tmp_path, "qa")
        builder.ctx.config.prompt = qa_prompt

        task = _make_task(pre_scan=True, workflow_step="qa_review")
        prompt = builder._build_prompt_legacy(task)
        assert "PRE-SCAN MODE" in prompt
        assert "lightweight checks" in prompt


# -- TestIsPrescanAlreadyQueued --

class TestIsPrescanAlreadyQueued:
    """Tests for _is_prescan_already_queued()."""

    def test_not_queued(self, executor):
        assert not executor._is_prescan_already_queued("root-new")

    def test_queued_in_qa(self, executor, tmp_path):
        qa_dir = executor.queue_dir / "qa"
        qa_dir.mkdir()
        (qa_dir / "prescan-root-q.json").write_text("{}")
        assert executor._is_prescan_already_queued("root-q")

    def test_completed(self, executor, queue):
        (queue.completed_dir / "prescan-root-c.json").write_text("{}")
        assert executor._is_prescan_already_queued("root-c")


# -- TestRouteToStepTrigger --

class TestRouteToStepTrigger:
    """Tests that _route_to_step triggers pre-scan at the right transition."""

    def test_prescan_triggered_on_implement_to_code_review(self, executor, queue, tmp_path):
        """Verify _route_to_step calls _queue_qa_pre_scan for implement→code_review."""
        from agent_framework.workflow.dag import WorkflowStep, WorkflowDAG

        task = _make_task(
            workflow_step="implement",
            implementation_branch="feature/x",
            _root_task_id="root-trigger",
            _chain_depth=1,
        )

        target_step = WorkflowStep(
            id="code_review", agent="architect",
        )
        workflow = MagicMock(spec=WorkflowDAG)
        workflow.steps = {"code_review": target_step}

        executor._queue_qa_pre_scan = MagicMock()

        executor._route_to_step(task, target_step, workflow, "engineer", None)

        executor._queue_qa_pre_scan.assert_called_once_with(task)

    def test_prescan_not_triggered_on_other_transitions(self, executor, queue, tmp_path):
        """Pre-scan NOT triggered for non-implement→code_review transitions."""
        from agent_framework.workflow.dag import WorkflowStep, WorkflowDAG

        task = _make_task(
            workflow_step="code_review",
            _root_task_id="root-other",
            _chain_depth=1,
        )

        target_step = WorkflowStep(
            id="qa_review", agent="qa",
        )
        workflow = MagicMock(spec=WorkflowDAG)
        workflow.steps = {"qa_review": target_step}

        executor._queue_qa_pre_scan = MagicMock()

        executor._route_to_step(task, target_step, workflow, "architect", None)

        executor._queue_qa_pre_scan.assert_not_called()
