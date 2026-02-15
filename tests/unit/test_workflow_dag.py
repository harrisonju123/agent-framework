"""Tests for workflow DAG structure and validation."""

import pytest

from agent_framework.workflow.dag import (
    WorkflowDAG,
    WorkflowStep,
    WorkflowEdge,
    EdgeCondition,
    EdgeConditionType,
)


class TestWorkflowDAG:
    def test_simple_linear_dag(self):
        """Create a simple linear workflow DAG."""
        steps = {
            "plan": WorkflowStep(
                id="plan",
                agent="architect",
                next=[WorkflowEdge(target="implement")]
            ),
            "implement": WorkflowStep(
                id="implement",
                agent="engineer",
                next=[WorkflowEdge(target="review")]
            ),
            "review": WorkflowStep(
                id="review",
                agent="qa",
                next=[]
            ),
        }

        dag = WorkflowDAG(
            name="simple",
            description="Simple linear workflow",
            steps=steps,
            start_step="plan"
        )

        assert dag.name == "simple"
        assert dag.start_step == "plan"
        assert len(dag.steps) == 3
        assert dag.get_agents_in_order() == ["architect", "engineer", "qa"]

    def test_from_linear_chain(self):
        """Convert legacy linear chain to DAG."""
        agents = ["architect", "engineer", "qa"]
        dag = WorkflowDAG.from_linear_chain("legacy", agents)

        assert dag.name == "legacy"
        assert dag.start_step == "architect"
        assert len(dag.steps) == 3
        assert dag.get_agents_in_order() == agents
        assert dag.metadata.get("legacy_linear") is True

    def test_dag_with_conditional_branches(self):
        """Create DAG with conditional branches."""
        steps = {
            "implement": WorkflowStep(
                id="implement",
                agent="engineer",
                next=[WorkflowEdge(target="qa_review")]
            ),
            "qa_review": WorkflowStep(
                id="qa_review",
                agent="qa",
                next=[
                    WorkflowEdge(
                        target="create_pr",
                        condition=EdgeCondition(EdgeConditionType.APPROVED),
                        priority=10
                    ),
                    WorkflowEdge(
                        target="implement",
                        condition=EdgeCondition(EdgeConditionType.NEEDS_FIX),
                        priority=5
                    ),
                ]
            ),
            "create_pr": WorkflowStep(
                id="create_pr",
                agent="architect",
                next=[]
            ),
        }

        dag = WorkflowDAG(
            name="conditional",
            description="Workflow with conditional branches",
            steps=steps,
            start_step="implement"
        )

        # Check QA step has two edges
        qa_edges = dag.get_next_steps("qa_review")
        assert len(qa_edges) == 2
        assert qa_edges[0].condition.type == EdgeConditionType.APPROVED
        assert qa_edges[1].condition.type == EdgeConditionType.NEEDS_FIX

    def test_unconditional_cycle_detection(self):
        """Workflow should reject unconditional cycles (infinite loops)."""
        # Create an unconditional cycle: A -> B -> C -> A (all ALWAYS edges)
        steps = {
            "a": WorkflowStep(
                id="a",
                agent="agent_a",
                next=[WorkflowEdge(target="b")]  # ALWAYS edge (default)
            ),
            "b": WorkflowStep(
                id="b",
                agent="agent_b",
                next=[WorkflowEdge(target="c")]  # ALWAYS edge (default)
            ),
            "c": WorkflowStep(
                id="c",
                agent="agent_c",
                next=[WorkflowEdge(target="a")]  # ALWAYS edge - creates cycle!
            ),
        }

        with pytest.raises(ValueError, match="unconditional cycle"):
            WorkflowDAG(
                name="cyclic",
                description="Invalid cyclic workflow",
                steps=steps,
                start_step="a"
            )

    def test_conditional_cycle_allowed(self):
        """Conditional cycles (feedback loops) should be allowed."""
        # QA can send back to Engineer on failure - this is valid
        steps = {
            "implement": WorkflowStep(
                id="implement",
                agent="engineer",
                next=[WorkflowEdge(target="qa")]
            ),
            "qa": WorkflowStep(
                id="qa",
                agent="qa",
                next=[
                    WorkflowEdge(
                        target="done",
                        condition=EdgeCondition(EdgeConditionType.APPROVED)
                    ),
                    WorkflowEdge(
                        target="implement",  # Cycle back to engineer
                        condition=EdgeCondition(EdgeConditionType.NEEDS_FIX)
                    ),
                ]
            ),
            "done": WorkflowStep(
                id="done",
                agent="architect",
                next=[]
            ),
        }

        # Should not raise - conditional cycles are OK
        dag = WorkflowDAG(
            name="feedback_loop",
            description="Valid workflow with feedback loop",
            steps=steps,
            start_step="implement"
        )
        assert dag is not None

    def test_invalid_start_step(self):
        """Validate start_step exists in steps."""
        steps = {
            "plan": WorkflowStep(id="plan", agent="architect", next=[])
        }

        with pytest.raises(ValueError, match="Start step 'nonexistent' not found"):
            WorkflowDAG(
                name="invalid",
                description="Invalid start step",
                steps=steps,
                start_step="nonexistent"
            )

    def test_invalid_edge_target(self):
        """Validate edge targets exist in steps."""
        steps = {
            "plan": WorkflowStep(
                id="plan",
                agent="architect",
                next=[WorkflowEdge(target="nonexistent")]
            )
        }

        with pytest.raises(ValueError, match="edge to non-existent step"):
            WorkflowDAG(
                name="invalid",
                description="Invalid edge target",
                steps=steps,
                start_step="plan"
            )

    def test_terminal_step_detection(self):
        """Check if a step is terminal (no outgoing edges)."""
        steps = {
            "start": WorkflowStep(
                id="start",
                agent="architect",
                next=[WorkflowEdge(target="end")]
            ),
            "end": WorkflowStep(
                id="end",
                agent="engineer",
                next=[]
            ),
        }

        dag = WorkflowDAG(
            name="terminal_test",
            description="Test terminal detection",
            steps=steps,
            start_step="start"
        )

        assert not dag.is_terminal_step("start")
        assert dag.is_terminal_step("end")

    def test_edge_priority_sorting(self):
        """Edges should be sorted by priority (highest first)."""
        edges = [
            WorkflowEdge(target="low", priority=1),
            WorkflowEdge(target="high", priority=10),
            WorkflowEdge(target="medium", priority=5),
        ]

        step = WorkflowStep(id="test", agent="test_agent", next=edges)

        # After __post_init__, edges should be sorted by priority
        assert step.next[0].target == "high"
        assert step.next[1].target == "medium"
        assert step.next[2].target == "low"

    def test_get_all_agents(self):
        """Get unique list of all agents in workflow."""
        steps = {
            "step1": WorkflowStep(id="step1", agent="architect", next=[]),
            "step2": WorkflowStep(id="step2", agent="engineer", next=[]),
            "step3": WorkflowStep(id="step3", agent="engineer", next=[]),  # Duplicate
            "step4": WorkflowStep(id="step4", agent="qa", next=[]),
        }

        dag = WorkflowDAG(
            name="multi_agent",
            description="Multiple agents",
            steps=steps,
            start_step="step1"
        )

        agents = dag.get_all_agents()
        assert set(agents) == {"architect", "engineer", "qa"}
        assert len(agents) == 3  # No duplicates

    def test_self_loop_detection(self):
        """Detect self-loops (step pointing to itself)."""
        steps = {
            "self_loop": WorkflowStep(
                id="self_loop",
                agent="architect",
                next=[WorkflowEdge(target="self_loop")]  # Self-loop
            )
        }

        with pytest.raises(ValueError, match="cycle"):
            WorkflowDAG(
                name="self_loop",
                description="Self-loop workflow",
                steps=steps,
                start_step="self_loop"
            )


class TestEdgeConditions:
    def test_always_condition(self):
        """ALWAYS condition requires no parameters."""
        condition = EdgeCondition(EdgeConditionType.ALWAYS)
        assert condition.type == EdgeConditionType.ALWAYS
        assert condition.params == {}

    def test_files_match_requires_pattern(self):
        """FILES_MATCH condition requires pattern parameter."""
        with pytest.raises(ValueError, match="pattern"):
            EdgeCondition(EdgeConditionType.FILES_MATCH)

        # Should work with pattern
        condition = EdgeCondition(
            EdgeConditionType.FILES_MATCH,
            params={"pattern": "*.md"}
        )
        assert condition.params["pattern"] == "*.md"

    def test_pr_size_requires_max_files(self):
        """PR_SIZE_UNDER condition requires max_files parameter."""
        with pytest.raises(ValueError, match="max_files"):
            EdgeCondition(EdgeConditionType.PR_SIZE_UNDER)

        # Should work with max_files
        condition = EdgeCondition(
            EdgeConditionType.PR_SIZE_UNDER,
            params={"max_files": 5}
        )
        assert condition.params["max_files"] == 5

    def test_signal_target_requires_target(self):
        """SIGNAL_TARGET condition requires target parameter."""
        with pytest.raises(ValueError, match="target"):
            EdgeCondition(EdgeConditionType.SIGNAL_TARGET)

        # Should work with target
        condition = EdgeCondition(
            EdgeConditionType.SIGNAL_TARGET,
            params={"target": "qa"}
        )
        assert condition.params["target"] == "qa"


class TestWorkflowStep:
    def test_task_type_override(self):
        """Steps can override default task type."""
        step = WorkflowStep(
            id="custom",
            agent="engineer",
            task_type_override="TESTING"
        )

        assert step.task_type_override == "TESTING"

    def test_metadata(self):
        """Steps can have arbitrary metadata."""
        step = WorkflowStep(
            id="meta",
            agent="qa",
            metadata={"timeout": 1800, "requires_approval": True}
        )

        assert step.metadata["timeout"] == 1800
        assert step.metadata["requires_approval"] is True
