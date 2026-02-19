"""Tests for per-task budget ceiling feature."""

from datetime import datetime, timezone
from types import MappingProxyType
from unittest.mock import MagicMock

import pytest

from agent_framework.core.budget_manager import BudgetManager
from agent_framework.core.config import OptimizationConfig
from agent_framework.core.task import Task, TaskStatus, TaskType, PlanDocument
from agent_framework.workflow.executor import WorkflowExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(
    *,
    task_id="task-1",
    estimated_effort=None,
    context=None,
    plan=None,
):
    return Task(
        id=task_id,
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.IN_PROGRESS,
        priority=3,
        created_by="test",
        assigned_to="engineer",
        created_at=datetime.now(timezone.utc),
        title="Test task",
        description="A test task",
        estimated_effort=estimated_effort,
        context=context or {},
        plan=plan,
    )


_DEFAULT_CEILINGS = {"XS": 3.0, "S": 5.0, "M": 15.0, "L": 30.0, "XL": 50.0}


def _make_budget_manager(enable_ceilings=True, ceilings=None):
    opt_config = {
        "enable_effort_budget_ceilings": enable_ceilings,
        "effort_budget_ceilings": _DEFAULT_CEILINGS if ceilings is None else ceilings,
    }
    return BudgetManager(
        agent_id="test-agent",
        optimization_config=opt_config,
        logger=MagicMock(),
        session_logger=MagicMock(),
        llm=MagicMock(),
        workspace=MagicMock(),
        activity_manager=MagicMock(),
    )


def _make_executor():
    return WorkflowExecutor(
        queue=MagicMock(),
        queue_dir=MagicMock(),
        agent_logger=MagicMock(),
    )


def _make_plan(file_count, step_count=0):
    plan = MagicMock(spec=PlanDocument)
    plan.files_to_modify = [f"file_{i}.py" for i in range(file_count)]
    plan.approach = [f"step_{i}" for i in range(step_count)]
    return plan


# ---------------------------------------------------------------------------
# derive_effort_from_plan
# ---------------------------------------------------------------------------

class TestDeriveEffortFromPlan:
    def test_no_plan_defaults_to_m(self):
        bm = _make_budget_manager()
        assert bm.derive_effort_from_plan(None) == "M"

    def test_no_files_defaults_to_m(self):
        bm = _make_budget_manager()
        plan = MagicMock(spec=PlanDocument)
        plan.files_to_modify = []
        plan.approach = []
        assert bm.derive_effort_from_plan(plan) == "M"

    def test_steps_only_sizes_correctly(self):
        """Plans with approach steps but no files still get sized."""
        bm = _make_budget_manager()
        # 0 files + 8 steps = 200 → S
        assert bm.derive_effort_from_plan(_make_plan(0, step_count=8)) == "S"

    def test_xs_boundary(self):
        """< 150 lines → XS. 2 files + 1 step = 125."""
        bm = _make_budget_manager()
        assert bm.derive_effort_from_plan(_make_plan(2, step_count=1)) == "XS"  # 125

    def test_xs_to_s_exact_boundary(self):
        """Exactly 150 → S (not XS)."""
        bm = _make_budget_manager()
        # 3 files + 0 steps = 150
        assert bm.derive_effort_from_plan(_make_plan(3, step_count=0)) == "S"

    def test_s_boundary(self):
        """150–349 lines → S."""
        bm = _make_budget_manager()
        # 3 files + 1 step = 175
        assert bm.derive_effort_from_plan(_make_plan(3, step_count=1)) == "S"
        # 5 files + 3 steps = 325
        assert bm.derive_effort_from_plan(_make_plan(5, step_count=3)) == "S"

    def test_s_to_m_exact_boundary(self):
        """Exactly 350 → M (not S)."""
        bm = _make_budget_manager()
        # 7 files + 0 steps = 350
        assert bm.derive_effort_from_plan(_make_plan(7, step_count=0)) == "M"

    def test_m_boundary(self):
        """350–599 lines → M."""
        bm = _make_budget_manager()
        # 10 files + 3 steps = 575
        assert bm.derive_effort_from_plan(_make_plan(10, step_count=3)) == "M"

    def test_m_to_l_exact_boundary(self):
        """Exactly 600 → L (not M)."""
        bm = _make_budget_manager()
        # 12 files + 0 steps = 600
        assert bm.derive_effort_from_plan(_make_plan(12, step_count=0)) == "L"

    def test_l_boundary(self):
        """600–999 lines → L."""
        bm = _make_budget_manager()
        # 18 files + 3 steps = 975
        assert bm.derive_effort_from_plan(_make_plan(18, step_count=3)) == "L"

    def test_l_to_xl_exact_boundary(self):
        """Exactly 1000 → XL (not L)."""
        bm = _make_budget_manager()
        # 20 files + 0 steps = 1000
        assert bm.derive_effort_from_plan(_make_plan(20, step_count=0)) == "XL"

    def test_xl_boundary(self):
        """≥ 1000 lines → XL."""
        bm = _make_budget_manager()
        # 18 files + 5 steps = 1025
        assert bm.derive_effort_from_plan(_make_plan(18, step_count=5)) == "XL"


# ---------------------------------------------------------------------------
# get_effort_ceiling
# ---------------------------------------------------------------------------

class TestGetEffortCeiling:
    def test_returns_ceiling_for_known_size(self):
        bm = _make_budget_manager()
        assert bm.get_effort_ceiling("S") == 5.0
        assert bm.get_effort_ceiling("XL") == 50.0

    def test_returns_none_for_unknown_size(self):
        bm = _make_budget_manager()
        assert bm.get_effort_ceiling("XXL") is None

    def test_returns_none_when_ceilings_empty(self):
        bm = _make_budget_manager(ceilings={})
        assert bm.get_effort_ceiling("M") is None


# ---------------------------------------------------------------------------
# _check_budget_ceiling (WorkflowExecutor)
# ---------------------------------------------------------------------------

class TestCheckBudgetCeiling:
    def test_ok_when_no_ceiling(self):
        executor = _make_executor()
        task = _make_task(context={})
        assert executor._check_budget_ceiling(task) == "ok"

    def test_ok_when_under_80_percent(self):
        executor = _make_executor()
        task = _make_task(context={
            "_budget_ceiling": 10.0,
            "_cumulative_cost": 5.0,
        })
        assert executor._check_budget_ceiling(task) == "ok"

    def test_warn_at_80_percent(self):
        executor = _make_executor()
        task = _make_task(context={
            "_budget_ceiling": 10.0,
            "_cumulative_cost": 8.0,
        })
        assert executor._check_budget_ceiling(task) == "warn"

    def test_halt_at_ceiling(self):
        executor = _make_executor()
        task = _make_task(context={
            "_budget_ceiling": 10.0,
            "_cumulative_cost": 10.0,
        })
        assert executor._check_budget_ceiling(task) == "halt"

    def test_halt_above_ceiling(self):
        executor = _make_executor()
        task = _make_task(context={
            "_budget_ceiling": 5.0,
            "_cumulative_cost": 7.5,
        })
        assert executor._check_budget_ceiling(task) == "halt"

    def test_warn_boundary_exact(self):
        executor = _make_executor()
        task = _make_task(context={
            "_budget_ceiling": 10.0,
            "_cumulative_cost": 8.0,  # exactly 80%
        })
        assert executor._check_budget_ceiling(task) == "warn"


# ---------------------------------------------------------------------------
# Cost accumulation in _run_post_completion_flow
# ---------------------------------------------------------------------------

class TestCostAccumulation:
    def test_cumulative_cost_increments(self):
        """Verify _cumulative_cost increments in context across chain hops."""
        from agent_framework.core.agent import Agent

        agent = MagicMock()
        agent._budget = MagicMock()
        agent._budget.estimate_cost.return_value = 2.50
        agent._optimization_config = MappingProxyType({
            "enable_effort_budget_ceilings": False,
        })
        agent._workflow_router = MagicMock()
        agent._review_cycle = MagicMock()
        agent._git_ops = MagicMock()
        agent._memory_retriever = MagicMock()
        agent._tool_pattern_analyzer = MagicMock()
        agent.config = MagicMock()
        agent.config.base_id = "engineer"
        agent.logger = MagicMock()

        agent._run_post_completion_flow = Agent._run_post_completion_flow.__get__(agent)
        agent._resolve_budget_ceiling = Agent._resolve_budget_ceiling.__get__(agent)
        agent._extract_and_store_memories = MagicMock()
        agent._analyze_tool_patterns = MagicMock()
        agent._log_task_completion_metrics = MagicMock()

        task = _make_task(context={"workflow": "default", "_cumulative_cost": 1.0})
        response = MagicMock()
        task_start_time = datetime.now(timezone.utc)

        agent._run_post_completion_flow(task, response, None, task_start_time)

        assert task.context["_cumulative_cost"] == pytest.approx(3.5)

    def test_cumulative_cost_starts_at_zero(self):
        """First hop: no prior cost, starts from 0."""
        from agent_framework.core.agent import Agent

        agent = MagicMock()
        agent._budget = MagicMock()
        agent._budget.estimate_cost.return_value = 1.25
        agent._optimization_config = MappingProxyType({
            "enable_effort_budget_ceilings": False,
        })
        agent._workflow_router = MagicMock()
        agent._review_cycle = MagicMock()
        agent._git_ops = MagicMock()
        agent._memory_retriever = MagicMock()
        agent._tool_pattern_analyzer = MagicMock()
        agent.config = MagicMock()
        agent.config.base_id = "architect"
        agent.logger = MagicMock()

        agent._run_post_completion_flow = Agent._run_post_completion_flow.__get__(agent)
        agent._resolve_budget_ceiling = Agent._resolve_budget_ceiling.__get__(agent)
        agent._extract_and_store_memories = MagicMock()
        agent._analyze_tool_patterns = MagicMock()
        agent._log_task_completion_metrics = MagicMock()

        task = _make_task(context={"workflow": "default"})
        response = MagicMock()

        agent._run_post_completion_flow(task, response, None, datetime.now(timezone.utc))

        assert task.context["_cumulative_cost"] == pytest.approx(1.25)

    def test_no_cost_when_response_is_none(self):
        """No cost accumulated when response is None (e.g. pre-scan routing)."""
        from agent_framework.core.agent import Agent

        agent = MagicMock()
        agent._budget = MagicMock()
        agent._optimization_config = MappingProxyType({
            "enable_effort_budget_ceilings": False,
        })
        agent._workflow_router = MagicMock()
        agent._review_cycle = MagicMock()
        agent._git_ops = MagicMock()
        agent._memory_retriever = MagicMock()
        agent._tool_pattern_analyzer = MagicMock()
        agent.config = MagicMock()
        agent.config.base_id = "engineer"
        agent.logger = MagicMock()

        agent._run_post_completion_flow = Agent._run_post_completion_flow.__get__(agent)
        agent._resolve_budget_ceiling = Agent._resolve_budget_ceiling.__get__(agent)
        agent._extract_and_store_memories = MagicMock()
        agent._analyze_tool_patterns = MagicMock()
        agent._log_task_completion_metrics = MagicMock()

        task = _make_task(context={"workflow": "default"})

        agent._run_post_completion_flow(task, None, None, datetime.now(timezone.utc))

        assert "_cumulative_cost" not in task.context
        agent._budget.estimate_cost.assert_not_called()


# ---------------------------------------------------------------------------
# Ceiling stamping
# ---------------------------------------------------------------------------

class TestCeilingStamping:
    def test_ceiling_set_from_estimated_effort(self):
        from agent_framework.core.agent import Agent

        agent = MagicMock()
        agent._budget = MagicMock()
        agent._budget.estimate_cost.return_value = 1.0
        agent._budget.get_effort_ceiling.return_value = 5.0
        agent._optimization_config = MappingProxyType({
            "enable_effort_budget_ceilings": True,
            "effort_budget_ceilings": {"S": 5.0},
        })
        agent._workflow_router = MagicMock()
        agent._review_cycle = MagicMock()
        agent._git_ops = MagicMock()
        agent._memory_retriever = MagicMock()
        agent._tool_pattern_analyzer = MagicMock()
        agent.config = MagicMock()
        agent.config.base_id = "architect"
        agent.logger = MagicMock()

        agent._run_post_completion_flow = Agent._run_post_completion_flow.__get__(agent)
        agent._resolve_budget_ceiling = Agent._resolve_budget_ceiling.__get__(agent)
        agent._extract_and_store_memories = MagicMock()
        agent._analyze_tool_patterns = MagicMock()
        agent._log_task_completion_metrics = MagicMock()

        task = _make_task(estimated_effort="S", context={"workflow": "default"})
        response = MagicMock()

        agent._run_post_completion_flow(task, response, None, datetime.now(timezone.utc))

        assert task.context["_budget_ceiling"] == 5.0

    def test_ceiling_not_overwritten_on_later_hops(self):
        """Once stamped, _budget_ceiling must not change."""
        from agent_framework.core.agent import Agent

        agent = MagicMock()
        agent._budget = MagicMock()
        agent._budget.estimate_cost.return_value = 1.0
        agent._optimization_config = MappingProxyType({
            "enable_effort_budget_ceilings": True,
        })
        agent._workflow_router = MagicMock()
        agent._review_cycle = MagicMock()
        agent._git_ops = MagicMock()
        agent._memory_retriever = MagicMock()
        agent._tool_pattern_analyzer = MagicMock()
        agent.config = MagicMock()
        agent.config.base_id = "engineer"
        agent.logger = MagicMock()

        agent._run_post_completion_flow = Agent._run_post_completion_flow.__get__(agent)
        agent._resolve_budget_ceiling = Agent._resolve_budget_ceiling.__get__(agent)
        agent._extract_and_store_memories = MagicMock()
        agent._analyze_tool_patterns = MagicMock()
        agent._log_task_completion_metrics = MagicMock()

        task = _make_task(context={
            "workflow": "default",
            "_budget_ceiling": 15.0,
        })
        response = MagicMock()

        agent._run_post_completion_flow(task, response, None, datetime.now(timezone.utc))

        # Should remain 15.0, not re-derived
        assert task.context["_budget_ceiling"] == 15.0

    def test_no_ceiling_when_feature_disabled(self):
        from agent_framework.core.agent import Agent

        agent = MagicMock()
        agent._budget = MagicMock()
        agent._budget.estimate_cost.return_value = 1.0
        agent._optimization_config = MappingProxyType({
            "enable_effort_budget_ceilings": False,
        })
        agent._workflow_router = MagicMock()
        agent._review_cycle = MagicMock()
        agent._git_ops = MagicMock()
        agent._memory_retriever = MagicMock()
        agent._tool_pattern_analyzer = MagicMock()
        agent.config = MagicMock()
        agent.config.base_id = "engineer"
        agent.logger = MagicMock()

        agent._run_post_completion_flow = Agent._run_post_completion_flow.__get__(agent)
        agent._resolve_budget_ceiling = Agent._resolve_budget_ceiling.__get__(agent)
        agent._extract_and_store_memories = MagicMock()
        agent._analyze_tool_patterns = MagicMock()
        agent._log_task_completion_metrics = MagicMock()

        task = _make_task(context={"workflow": "default"})
        response = MagicMock()

        agent._run_post_completion_flow(task, response, None, datetime.now(timezone.utc))

        assert "_budget_ceiling" not in task.context

    def test_ceiling_derived_from_plan_when_no_estimated_effort(self):
        from agent_framework.core.agent import Agent

        agent = MagicMock()
        agent._budget = MagicMock()
        agent._budget.estimate_cost.return_value = 0.5
        agent._budget.derive_effort_from_plan.return_value = "L"
        agent._budget.get_effort_ceiling.return_value = 30.0
        agent._optimization_config = MappingProxyType({
            "enable_effort_budget_ceilings": True,
        })
        agent._workflow_router = MagicMock()
        agent._review_cycle = MagicMock()
        agent._git_ops = MagicMock()
        agent._memory_retriever = MagicMock()
        agent._tool_pattern_analyzer = MagicMock()
        agent.config = MagicMock()
        agent.config.base_id = "architect"
        agent.logger = MagicMock()

        agent._run_post_completion_flow = Agent._run_post_completion_flow.__get__(agent)
        agent._resolve_budget_ceiling = Agent._resolve_budget_ceiling.__get__(agent)
        agent._extract_and_store_memories = MagicMock()
        agent._analyze_tool_patterns = MagicMock()
        agent._log_task_completion_metrics = MagicMock()

        task = _make_task(context={"workflow": "default"})  # no estimated_effort
        response = MagicMock()

        agent._run_post_completion_flow(task, response, None, datetime.now(timezone.utc))

        assert task.context["_budget_ceiling"] == 30.0
        agent._budget.derive_effort_from_plan.assert_called_once()


# ---------------------------------------------------------------------------
# Clean halt on budget breach (no chain task queued)
# ---------------------------------------------------------------------------

class TestBudgetHalt:
    def test_halt_terminates_chain_without_create_pr(self):
        """When ceiling breached, _route_to_step returns without queuing any chain task."""
        from agent_framework.workflow.dag import WorkflowDAG, WorkflowStep, WorkflowEdge, EdgeCondition, EdgeConditionType

        executor = _make_executor()

        implement_step = WorkflowStep(id="implement", agent="engineer")
        create_pr_step = WorkflowStep(id="create_pr", agent="architect")
        qa_step = WorkflowStep(
            id="qa_review",
            agent="qa",
            next=[WorkflowEdge(target="create_pr", condition=EdgeCondition(EdgeConditionType.ALWAYS))],
        )

        workflow = WorkflowDAG(
            name="default",
            description="test",
            steps={
                "implement": implement_step,
                "qa_review": qa_step,
                "create_pr": create_pr_step,
            },
            start_step="implement",
        )

        task = _make_task(context={
            "_budget_ceiling": 5.0,
            "_cumulative_cost": 6.0,
            "workflow": "default",
            "workflow_step": "qa_review",
            "implementation_branch": "feature/xyz",
        })

        executor._build_chain_task = MagicMock(return_value=_make_task(task_id="chain-1"))
        executor._is_chain_task_already_queued = MagicMock(return_value=False)

        executor._route_to_step(task, qa_step, workflow, "qa", None)

        # No chain task should be built — clean halt
        executor._build_chain_task.assert_not_called()

    def test_budget_halted_flag_set_in_context(self):
        """Halt sets budget_halted=True for observability."""
        from agent_framework.workflow.dag import WorkflowDAG, WorkflowStep

        executor = _make_executor()

        implement_step = WorkflowStep(id="implement", agent="engineer")
        qa_step = WorkflowStep(id="qa_review", agent="qa")

        workflow = WorkflowDAG(
            name="default",
            description="test",
            steps={"implement": implement_step, "qa_review": qa_step},
            start_step="implement",
        )

        task = _make_task(context={
            "_budget_ceiling": 5.0,
            "_cumulative_cost": 7.0,
            "workflow": "default",
        })

        executor._build_chain_task = MagicMock()
        executor._is_chain_task_already_queued = MagicMock(return_value=False)

        executor._route_to_step(task, qa_step, workflow, "engineer", None)

        assert task.context["budget_halted"] is True

    def test_no_halt_when_under_ceiling(self):
        """Normal routing when cost is under ceiling."""
        from agent_framework.workflow.dag import WorkflowDAG, WorkflowStep

        executor = _make_executor()

        implement_step = WorkflowStep(id="implement", agent="engineer")
        qa_step = WorkflowStep(id="qa_review", agent="qa")

        workflow = WorkflowDAG(
            name="default",
            description="test",
            steps={"implement": implement_step, "qa_review": qa_step},
            start_step="implement",
        )

        task = _make_task(context={
            "_budget_ceiling": 15.0,
            "_cumulative_cost": 3.0,
            "workflow": "default",
        })

        executor._build_chain_task = MagicMock(return_value=_make_task(task_id="chain-1"))
        executor._is_chain_task_already_queued = MagicMock(return_value=False)

        executor._route_to_step(task, qa_step, workflow, "engineer", None)

        # Should route to qa_step normally
        call_args = executor._build_chain_task.call_args
        assert call_args[0][1] == qa_step
        assert "budget_halted" not in task.context


# ---------------------------------------------------------------------------
# Fan-in cost aggregation
# ---------------------------------------------------------------------------

class TestFanInCostAggregation:
    def test_subtask_costs_aggregated(self):
        """Subtask own costs (cumulative - parent baseline) summed correctly."""
        from agent_framework.queue.file_queue import FileQueue

        queue = FileQueue.__new__(FileQueue)
        queue.create_fan_in_task = FileQueue.create_fan_in_task.__get__(queue)

        parent = _make_task(task_id="parent-1", context={
            "_cumulative_cost": 2.0,
            "_budget_ceiling": 30.0,
        })

        # Each subtask inherits parent baseline (2.0) then adds own cost
        sub1 = _make_task(task_id="sub-1", context={"_cumulative_cost": 5.0})  # own: 3.0
        sub1.result_summary = "Done sub1"
        sub2 = _make_task(task_id="sub-2", context={"_cumulative_cost": 6.5})  # own: 4.5
        sub2.result_summary = "Done sub2"

        fan_in = queue.create_fan_in_task(parent, [sub1, sub2])

        # parent baseline (2.0) + sub1 own (3.0) + sub2 own (4.5) = 9.5
        assert fan_in.context["_cumulative_cost"] == pytest.approx(9.5)
        # Ceiling propagated from parent
        assert fan_in.context["_budget_ceiling"] == 30.0

    def test_subtask_costs_default_to_zero(self):
        from agent_framework.queue.file_queue import FileQueue

        queue = FileQueue.__new__(FileQueue)
        queue.create_fan_in_task = FileQueue.create_fan_in_task.__get__(queue)

        parent = _make_task(task_id="parent-1", context={})
        sub1 = _make_task(task_id="sub-1", context={})
        sub1.result_summary = "Done"

        fan_in = queue.create_fan_in_task(parent, [sub1])

        assert fan_in.context["_cumulative_cost"] == 0.0

    def test_subtask_with_no_own_cost(self):
        """Subtask that only inherited parent cost contributes 0 own cost."""
        from agent_framework.queue.file_queue import FileQueue

        queue = FileQueue.__new__(FileQueue)
        queue.create_fan_in_task = FileQueue.create_fan_in_task.__get__(queue)

        parent = _make_task(task_id="parent-1", context={"_cumulative_cost": 2.0})
        # Subtask inherited 2.0 but had no LLM call (no own cost)
        sub1 = _make_task(task_id="sub-1", context={"_cumulative_cost": 2.0})
        sub1.result_summary = "No-op"

        fan_in = queue.create_fan_in_task(parent, [sub1])

        assert fan_in.context["_cumulative_cost"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestOptimizationConfigValidation:
    def test_default_ceilings_valid(self):
        config = OptimizationConfig()
        assert config.effort_budget_ceilings["M"] == 15.0

    def test_negative_ceiling_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            OptimizationConfig(effort_budget_ceilings={"S": -1.0})

    def test_zero_ceiling_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            OptimizationConfig(effort_budget_ceilings={"M": 0.0})

    def test_custom_ceilings_accepted(self):
        config = OptimizationConfig(effort_budget_ceilings={"TINY": 1.0, "HUGE": 100.0})
        assert config.effort_budget_ceilings["TINY"] == 1.0
        assert config.effort_budget_ceilings["HUGE"] == 100.0

    def test_absolute_ceiling_must_be_positive(self):
        with pytest.raises(ValueError, match="must be positive"):
            OptimizationConfig(absolute_budget_ceiling_usd=0.0)
        with pytest.raises(ValueError, match="must be positive"):
            OptimizationConfig(absolute_budget_ceiling_usd=-5.0)

    def test_absolute_ceiling_none_by_default(self):
        config = OptimizationConfig()
        assert config.absolute_budget_ceiling_usd is None

    def test_absolute_ceiling_accepted(self):
        config = OptimizationConfig(absolute_budget_ceiling_usd=25.0)
        assert config.absolute_budget_ceiling_usd == 25.0


# ---------------------------------------------------------------------------
# _resolve_budget_ceiling with absolute ceiling
# ---------------------------------------------------------------------------

class TestResolveBudgetCeilingWithAbsolute:
    """Test min(effort, absolute) logic in _resolve_budget_ceiling."""

    def _make_agent(self, *, enable_ceilings=True, absolute=None, effort_ceiling_return=None):
        from agent_framework.core.agent import Agent

        agent = MagicMock()
        agent._budget = MagicMock()
        agent._budget.get_effort_ceiling.return_value = effort_ceiling_return
        agent._budget.derive_effort_from_plan.return_value = "M"
        agent._optimization_config = MappingProxyType({
            "enable_effort_budget_ceilings": enable_ceilings,
            "absolute_budget_ceiling_usd": absolute,
        })
        agent._resolve_budget_ceiling = Agent._resolve_budget_ceiling.__get__(agent)
        return agent

    def test_absolute_overrides_effort_ceiling_when_lower(self):
        """absolute=$10 < M=$15 → use $10."""
        agent = self._make_agent(absolute=10.0, effort_ceiling_return=15.0)
        task = _make_task(estimated_effort="M")
        assert agent._resolve_budget_ceiling(task) == 10.0

    def test_effort_ceiling_wins_when_lower_than_absolute(self):
        """M=$15 < absolute=$25 → use $15."""
        agent = self._make_agent(absolute=25.0, effort_ceiling_return=15.0)
        task = _make_task(estimated_effort="M")
        assert agent._resolve_budget_ceiling(task) == 15.0

    def test_absolute_alone_when_effort_not_in_config(self):
        """No per-effort ceiling configured, absolute=$25 used as fallback."""
        agent = self._make_agent(enable_ceilings=True, absolute=25.0, effort_ceiling_return=None)
        task = _make_task(estimated_effort="M")
        assert agent._resolve_budget_ceiling(task) == 25.0

    def test_absolute_alone_when_ceilings_disabled(self):
        """effort ceilings disabled but absolute set → absolute used."""
        agent = self._make_agent(enable_ceilings=False, absolute=25.0)
        task = _make_task(estimated_effort="M")
        assert agent._resolve_budget_ceiling(task) == 25.0

    def test_none_when_both_absent(self):
        """Neither ceiling configured → None."""
        agent = self._make_agent(enable_ceilings=False, absolute=None)
        task = _make_task(estimated_effort="M")
        assert agent._resolve_budget_ceiling(task) is None
