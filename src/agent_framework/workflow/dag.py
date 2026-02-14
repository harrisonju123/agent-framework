"""Workflow DAG (Directed Acyclic Graph) representation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


@dataclass
class CheckpointConfig:
    """Configuration for a workflow checkpoint (human approval gate).

    Checkpoints pause workflow execution until a human approves continuation.
    """
    message: str  # Displayed when checkpoint is reached
    reason: Optional[str] = None  # Audit trail context for why approval is needed


class EdgeConditionType(str, Enum):
    """Types of conditions for workflow edges."""
    ALWAYS = "always"  # Unconditional transition (default for backward compatibility)
    PR_CREATED = "pr_created"
    NO_PR = "no_pr"
    APPROVED = "approved"
    NEEDS_FIX = "needs_fix"
    TEST_PASSED = "test_passed"
    TEST_FAILED = "test_failed"
    FILES_MATCH = "files_match"
    PR_SIZE_UNDER = "pr_size_under"
    SIGNAL_TARGET = "signal_target"  # Use routing signal if present


@dataclass
class EdgeCondition:
    """Condition that must be met for an edge to be traversed."""
    type: EdgeConditionType
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate condition parameters."""
        if self.type == EdgeConditionType.FILES_MATCH and "pattern" not in self.params:
            raise ValueError("files_match condition requires 'pattern' parameter")
        if self.type == EdgeConditionType.PR_SIZE_UNDER and "max_files" not in self.params:
            raise ValueError("pr_size_under condition requires 'max_files' parameter")
        if self.type == EdgeConditionType.SIGNAL_TARGET and "target" not in self.params:
            raise ValueError("signal_target condition requires 'target' parameter")


@dataclass
class WorkflowEdge:
    """Directed edge in the workflow graph."""
    target: str  # Target step ID
    condition: EdgeCondition = field(default_factory=lambda: EdgeCondition(EdgeConditionType.ALWAYS))
    priority: int = 0  # Higher priority edges evaluated first


@dataclass
class WorkflowStep:
    """A step in the workflow (node in the DAG)."""
    id: str  # Unique step identifier
    agent: str  # Agent responsible for this step
    next: List[WorkflowEdge] = field(default_factory=list)
    task_type_override: Optional[str] = None  # Override default task type for this agent
    metadata: Dict[str, Any] = field(default_factory=dict)
    checkpoint: Optional[CheckpointConfig] = None

    def __post_init__(self):
        """Sort edges by priority (highest first)."""
        self.next.sort(key=lambda e: e.priority, reverse=True)


@dataclass
class WorkflowDAG:
    """Directed graph representing a workflow (allows conditional cycles).

    While traditionally a DAG (Directed Acyclic Graph), we allow controlled cycles
    for feedback loops (e.g., QA sends code back to Engineer for fixes).
    """
    name: str
    description: str
    steps: Dict[str, WorkflowStep]  # step_id -> WorkflowStep
    start_step: str  # Entry point step ID
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the workflow structure."""
        if self.start_step not in self.steps:
            raise ValueError(f"Start step '{self.start_step}' not found in steps")

        # Validate all edge targets exist
        for step_id, step in self.steps.items():
            for edge in step.next:
                if edge.target not in self.steps:
                    raise ValueError(
                        f"Step '{step_id}' has edge to non-existent step '{edge.target}'"
                    )

        # Check for unconditional cycles (problematic - infinite loops)
        # Conditional cycles are OK (e.g., QA -> Engineer on failure)
        if self._has_unconditional_cycle():
            raise ValueError("Workflow contains an unconditional cycle (infinite loop)")

    def _has_unconditional_cycle(self) -> bool:
        """Detect unconditional cycles (problematic infinite loops).

        Conditional cycles (like QA -> Engineer on failure) are allowed.
        Only ALWAYS edges are considered for cycle detection.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {step_id: WHITE for step_id in self.steps}

        def visit(node: str) -> bool:
            """Returns True if unconditional cycle detected."""
            if color[node] == GRAY:
                return True  # Back edge found (cycle)
            if color[node] == BLACK:
                return False  # Already processed

            color[node] = GRAY
            step = self.steps[node]
            # Only follow ALWAYS edges (unconditional transitions)
            for edge in step.next:
                if edge.condition.type == EdgeConditionType.ALWAYS:
                    if visit(edge.target):
                        return True
            color[node] = BLACK
            return False

        for step_id in self.steps:
            if color[step_id] == WHITE:
                if visit(step_id):
                    return True
        return False

    def get_next_steps(self, current_step_id: str) -> List[WorkflowEdge]:
        """Get all possible next steps from current step."""
        if current_step_id not in self.steps:
            return []
        return self.steps[current_step_id].next

    def get_agents_in_order(self) -> List[str]:
        """Get list of agents in topological order (for backward compatibility).

        Returns agents in the order they would be executed in a linear chain.
        This is used to support the legacy linear workflow pattern.
        """
        visited = set()
        result = []

        def dfs(step_id: str):
            if step_id in visited:
                return
            visited.add(step_id)
            step = self.steps[step_id]
            if step.agent not in result:
                result.append(step.agent)
            for edge in step.next:
                dfs(edge.target)

        dfs(self.start_step)
        return result

    @classmethod
    def from_linear_chain(cls, name: str, agents: List[str], description: str = "") -> "WorkflowDAG":
        """Create a simple linear DAG from a list of agents (backward compatible).

        Converts the legacy linear workflow format into a DAG with unconditional edges.
        Example: [architect, engineer, qa] becomes architect→engineer→qa
        """
        if not agents:
            raise ValueError("Cannot create DAG from empty agent list")

        steps = {}
        for i, agent in enumerate(agents):
            step_id = agent  # Use agent name as step ID for simplicity
            next_edges = []
            if i < len(agents) - 1:
                # Add edge to next agent in chain
                next_edges.append(WorkflowEdge(
                    target=agents[i + 1],
                    condition=EdgeCondition(EdgeConditionType.ALWAYS)
                ))

            steps[step_id] = WorkflowStep(
                id=step_id,
                agent=agent,
                next=next_edges
            )

        return cls(
            name=name,
            description=description or f"Linear workflow: {' → '.join(agents)}",
            steps=steps,
            start_step=agents[0],
            metadata={"legacy_linear": True}
        )

    def is_terminal_step(self, step_id: str) -> bool:
        """Check if a step has no outgoing edges (terminal node)."""
        if step_id not in self.steps:
            return True
        return len(self.steps[step_id].next) == 0

    def get_all_agents(self) -> List[str]:
        """Get unique list of all agents in the workflow."""
        agents = set()
        for step in self.steps.values():
            agents.add(step.agent)
        return sorted(agents)
