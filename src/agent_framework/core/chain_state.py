"""Chain state file — single source of truth for workflow chain context.

Each workflow chain gets an append-only JSON file at
`.agent-communication/chain-state/{root_task_id}.json` that accumulates
structured data (plan, files modified, verdicts, findings) across all steps.

This replaces the fragile upstream_summary mechanism where raw LLM prose
was truncated and passed between agents. Instead, each step writes a
StepRecord with concrete data, and the prompt builder renders step-appropriate
context for the consuming agent.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.atomic_io import atomic_write_text
from ..utils.subprocess_utils import run_git_command

logger = logging.getLogger(__name__)

# Budget for rendered chain state context in prompts
CHAIN_STATE_MAX_PROMPT_CHARS = 12000


@dataclass
class StepRecord:
    """A single completed step in a workflow chain."""

    step_id: str                      # "plan", "implement", "code_review", etc.
    agent_id: str                     # "architect", "engineer", "qa"
    task_id: str                      # chain task ID for traceability
    completed_at: str                 # ISO timestamp
    summary: str                      # 1-2 paragraph structured summary (2KB max)
    verdict: Optional[str] = None     # "approved" / "needs_fix" / "no_changes"
    plan: Optional[Dict[str, Any]] = None  # PlanDocument serialized (plan step only)
    files_modified: List[str] = field(default_factory=list)  # git diff --name-only
    commit_shas: List[str] = field(default_factory=list)     # commits made
    findings: Optional[List[Dict[str, Any]]] = None  # structured findings (review steps)
    tool_stats: Optional[Dict[str, Any]] = None       # quantitative tool usage stats
    error: Optional[str] = None       # if step failed


@dataclass
class ChainState:
    """Accumulated state for an entire workflow chain."""

    root_task_id: str
    user_goal: str                    # original task description (stable)
    workflow: str                     # workflow name ("default", "preview", etc.)
    implementation_branch: Optional[str] = None
    steps: List[StepRecord] = field(default_factory=list)
    current_step: Optional[str] = None
    attempt: int = 1                  # retry counter


def _chain_state_dir(workspace: Path) -> Path:
    return workspace / ".agent-communication" / "chain-state"


def _chain_state_path(workspace: Path, root_task_id: str) -> Path:
    return _chain_state_dir(workspace) / f"{root_task_id}.json"


def load_chain_state(workspace: Path, root_task_id: str) -> Optional[ChainState]:
    """Load chain state from disk. Returns None if not found or corrupt."""
    path = _chain_state_path(workspace, root_task_id)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load chain state for {root_task_id}: {e}")
        return None

    if not isinstance(data, dict) or "root_task_id" not in data:
        logger.warning(f"Malformed chain state for {root_task_id}: missing root_task_id")
        return None

    # Filter to known fields to tolerate schema evolution (future fields don't crash old code)
    known_fields = set(StepRecord.__dataclass_fields__)
    steps = [
        StepRecord(**{k: v for k, v in s.items() if k in known_fields})
        for s in data.get("steps", [])
    ]

    return ChainState(
        root_task_id=data["root_task_id"],
        user_goal=data.get("user_goal", ""),
        workflow=data.get("workflow", "default"),
        implementation_branch=data.get("implementation_branch"),
        steps=steps,
        current_step=data.get("current_step"),
        attempt=data.get("attempt", 1),
    )


def save_chain_state(workspace: Path, state: ChainState) -> None:
    """Atomically write chain state to disk."""
    state_dir = _chain_state_dir(workspace)
    state_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "root_task_id": state.root_task_id,
        "user_goal": state.user_goal,
        "workflow": state.workflow,
        "implementation_branch": state.implementation_branch,
        "steps": [asdict(s) for s in state.steps],
        "current_step": state.current_step,
        "attempt": state.attempt,
    }

    path = _chain_state_path(workspace, state.root_task_id)
    atomic_write_text(path, json.dumps(data, indent=2, default=str))


def append_step(
    workspace: Path,
    task: "Task",
    agent_id: str,
    response_content: str,
    working_dir: Optional[Path] = None,
    tool_stats: Optional[Dict[str, Any]] = None,
) -> ChainState:
    """Append a completed step record to the chain state.

    Called from agent._handle_successful_response AFTER plan extraction
    and verdict setting, so all structured data is available.

    Note: read-modify-write is safe here because workflow chain steps are
    sequential — the next agent doesn't start until the current one completes
    and queues the continuation task. Pre-scan tasks skip this path entirely.
    """
    root_task_id = task.root_id

    # Load or create chain state
    state = load_chain_state(workspace, root_task_id)
    if state is None:
        state = ChainState(
            root_task_id=root_task_id,
            user_goal=task.context.get("user_goal", task.description),
            workflow=task.context.get("workflow", "default"),
        )

    # Update implementation branch if set
    impl_branch = task.context.get("implementation_branch")
    if impl_branch:
        state.implementation_branch = impl_branch

    step_id = task.context.get("workflow_step", "unknown")

    # Build a structured summary (not raw LLM output)
    summary = _build_step_summary(task, response_content, step_id)

    # Serialize plan if present
    plan_dict = None
    if task.plan is not None:
        try:
            plan_dict = task.plan.model_dump()
        except Exception:
            plan_dict = None

    # Collect git evidence
    files_modified = _collect_files_modified(working_dir)
    commit_shas = _collect_commit_shas(working_dir)

    # Structured findings from QA/review
    findings = None
    structured = task.context.get("structured_findings")
    if structured and isinstance(structured, dict):
        findings = structured.get("findings")

    record = StepRecord(
        step_id=step_id,
        agent_id=agent_id,
        task_id=task.id,
        completed_at=datetime.now(timezone.utc).isoformat(),
        summary=summary[:2048],
        verdict=task.context.get("verdict"),
        plan=plan_dict,
        files_modified=files_modified,
        commit_shas=commit_shas,
        findings=findings,
        tool_stats=tool_stats,
    )

    state.steps.append(record)
    state.current_step = None
    save_chain_state(workspace, state)

    return state


def _build_step_summary(task: "Task", response_content: str, step_id: str) -> str:
    """Build a structured summary from task data rather than raw LLM output.

    Prefers structured data (plan objectives, verdicts, findings) over
    raw response text. Only falls back to response truncation when no
    structured data exists.
    """
    parts = []

    if step_id in ("plan", "planning") and task.plan:
        parts.append("Plan objectives: " + "; ".join(task.plan.objectives[:5]))
        if task.plan.files_to_modify:
            parts.append("Files to modify: " + ", ".join(task.plan.files_to_modify[:10]))
        if task.plan.approach:
            parts.append("Approach: " + "; ".join(task.plan.approach[:5]))

    verdict = task.context.get("verdict")
    if verdict:
        parts.append(f"Verdict: {verdict}")

    findings = task.context.get("structured_findings")
    if findings and isinstance(findings, dict):
        finding_list = findings.get("findings", [])
        if finding_list:
            count = len(finding_list)
            parts.append(f"Findings: {count} issue(s) identified")

    if parts:
        return "\n".join(parts)

    # Fallback: extract first substantive paragraph from response
    if response_content:
        lines = [l.strip() for l in response_content.split("\n") if l.strip()]
        # Skip tool call noise — take first lines that look like prose
        prose_lines = []
        for line in lines:
            if line.startswith(("```", "{", "[", "Tool", "Reading", "Searching")):
                continue
            prose_lines.append(line)
            if len("\n".join(prose_lines)) > 500:
                break
        if prose_lines:
            return "\n".join(prose_lines)[:1024]

    return f"Step {step_id} completed by {task.context.get('source_agent', 'unknown')}"


def _collect_files_modified(working_dir: Optional[Path]) -> List[str]:
    """Run git diff --name-only to collect files modified in the working directory.

    Tries uncommitted changes first (most common), then staged-only, then
    last commit. Stops after the first strategy that finds results.
    """
    if not working_dir or not working_dir.exists():
        return []

    strategies = [
        ["diff", "--name-only", "HEAD"],      # uncommitted (staged + unstaged vs HEAD)
        ["diff", "--name-only", "--staged"],   # fresh repo with no HEAD yet
        ["diff", "--name-only", "HEAD~1"],     # agent already committed
    ]

    for args in strategies:
        try:
            result = run_git_command(args, cwd=working_dir, check=False, timeout=10)
            output = (result.stdout or "").strip()
            if output:
                return sorted(set(output.split("\n")))
        except Exception:
            continue

    return []


def _collect_commit_shas(working_dir: Optional[Path]) -> List[str]:
    """Collect recent commit SHAs from the working directory."""
    if not working_dir or not working_dir.exists():
        return []

    try:
        result = run_git_command(
            ["log", "-5", "--format=%h"],
            cwd=working_dir, check=False, timeout=10,
        )
        output = (result.stdout or "").strip()
        if output:
            return output.split("\n")
    except Exception:
        pass
    return []


# --- Prompt rendering ---

def render_for_step(state: ChainState, consumer_step: str) -> str:
    """Render chain state context appropriate for the consuming step.

    Different agents need different views of the chain history:
    - implement: needs the plan + requirements
    - implement (fix): needs review findings + files to fix
    - code_review: needs plan summary + files changed + commits
    - qa_review: needs plan + files changed + code review result
    - create_pr: needs combined summary for PR body
    """
    if not state.steps:
        return ""

    # Check if this is a fix cycle (review→engineer loop)
    is_fix_cycle = _is_fix_cycle(state, consumer_step)

    if consumer_step == "implement" and is_fix_cycle:
        return _render_for_fix(state)
    elif consumer_step == "implement":
        return _render_for_implement(state)
    elif consumer_step == "code_review":
        return _render_for_code_review(state)
    elif consumer_step in ("qa_review",):
        return _render_for_qa_review(state)
    elif consumer_step == "create_pr":
        return _render_for_create_pr(state)
    else:
        return _render_generic(state)


def _is_fix_cycle(state: ChainState, consumer_step: str) -> bool:
    """Detect if engineer is receiving work back from a review step."""
    if consumer_step != "implement":
        return False
    # If the last step was a review with needs_fix verdict
    for step in reversed(state.steps):
        if step.step_id in ("code_review", "qa_review"):
            return step.verdict == "needs_fix"
    return False


def _render_for_implement(state: ChainState) -> str:
    """Render context for the engineer's implementation step."""
    plan_step = _find_step(state, "plan")
    if not plan_step or not plan_step.plan:
        # No structured plan — fall through to legacy upstream_summary
        return ""

    lines = ["\n## CHAIN STATE — IMPLEMENTATION CONTEXT\n"]
    plan = plan_step.plan
    lines.append("### PLAN")
    if plan.get("objectives"):
        lines.append("**Objectives:**")
        for obj in plan["objectives"]:
            lines.append(f"- {obj}")
    if plan.get("approach"):
        lines.append("\n**Approach:**")
        for i, step in enumerate(plan["approach"], 1):
            lines.append(f"{i}. {step}")
    if plan.get("files_to_modify"):
        lines.append(f"\n**Files to modify:** {', '.join(plan['files_to_modify'])}")
    if plan.get("risks"):
        lines.append("\n**Risks:**")
        for risk in plan["risks"]:
            lines.append(f"- {risk}")
    lines.append("")

    return _truncate("\n".join(lines))


def _render_for_fix(state: ChainState) -> str:
    """Render context for the engineer's fix cycle (after review rejection)."""
    lines = ["\n## CHAIN STATE — FIX CYCLE\n"]

    # Find the most recent review step with findings
    for step in reversed(state.steps):
        if step.step_id in ("code_review", "qa_review") and step.verdict == "needs_fix":
            lines.append(f"### REVIEW FINDINGS ({step.step_id})")
            if step.findings:
                for i, f in enumerate(step.findings, 1):
                    severity = f.get("severity", "UNKNOWN")
                    desc = f.get("description", "")
                    file_path = f.get("file", "")
                    line_no = f.get("line_number")
                    location = f"{file_path}:{line_no}" if line_no else file_path
                    lines.append(f"{i}. [{severity}] {location} — {desc}")
                    suggested = f.get("suggested_fix")
                    if suggested:
                        lines.append(f"   Fix: {suggested}")
            elif step.summary:
                lines.append(step.summary)
            lines.append("")

            # Show files from the implement step (what needs fixing), not the review step
            impl_step = _find_step(state, "implement")
            if impl_step and impl_step.files_modified:
                lines.append(f"### FILES TO FIX\n{', '.join(impl_step.files_modified)}\n")
            break

    return _truncate("\n".join(lines))


def _render_for_code_review(state: ChainState) -> str:
    """Render context for the architect's code review step."""
    lines = ["\n## CHAIN STATE — CODE REVIEW CONTEXT\n"]

    # Plan objectives as review anchor
    plan_step = _find_step(state, "plan")
    if plan_step and plan_step.plan:
        objectives = plan_step.plan.get("objectives", [])
        if objectives:
            lines.append("### PLAN OBJECTIVES")
            for obj in objectives:
                lines.append(f"- {obj}")
            lines.append("")

    # Files changed by engineer
    impl_step = _find_step(state, "implement")
    if impl_step:
        if impl_step.files_modified:
            lines.append("### FILES CHANGED")
            for f in impl_step.files_modified:
                lines.append(f"- {f}")
            lines.append("")
        if impl_step.commit_shas:
            lines.append("### COMMIT LOG")
            lines.append(", ".join(impl_step.commit_shas))
            lines.append("")
        lines.extend(_render_tool_stats(impl_step))

    return _truncate("\n".join(lines))


def _render_for_qa_review(state: ChainState) -> str:
    """Render context for the QA review step."""
    lines = ["\n## CHAIN STATE — QA REVIEW CONTEXT\n"]

    # Plan for acceptance criteria reference
    plan_step = _find_step(state, "plan")
    if plan_step and plan_step.plan:
        plan = plan_step.plan
        if plan.get("success_criteria"):
            lines.append("### SUCCESS CRITERIA (from plan)")
            for criterion in plan["success_criteria"]:
                lines.append(f"- {criterion}")
            lines.append("")

    # Files changed by engineer
    impl_step = _find_step(state, "implement")
    if impl_step and impl_step.files_modified:
        lines.append("### FILES CHANGED")
        for f in impl_step.files_modified:
            lines.append(f"- {f}")
        lines.append("")

    # Tool usage from implementation
    if impl_step:
        lines.extend(_render_tool_stats(impl_step))

    # Code review result
    review_step = _find_step(state, "code_review")
    if review_step:
        lines.append("### CODE REVIEW RESULT")
        lines.append(f"Verdict: {review_step.verdict or 'unknown'}")
        if review_step.summary:
            lines.append(review_step.summary[:500])
        lines.append("")

    return _truncate("\n".join(lines))


def _render_for_create_pr(state: ChainState) -> str:
    """Render a combined summary for the PR creation step."""
    lines = ["\n## CHAIN STATE — PR SUMMARY\n"]

    # User goal
    if state.user_goal:
        lines.append(f"**Goal:** {state.user_goal[:500]}\n")

    # Plan summary
    plan_step = _find_step(state, "plan")
    if plan_step and plan_step.plan:
        objectives = plan_step.plan.get("objectives", [])
        if objectives:
            lines.append("### What was planned")
            for obj in objectives:
                lines.append(f"- {obj}")
            lines.append("")

    # Files changed
    all_files = set()
    for step in state.steps:
        all_files.update(step.files_modified)
    if all_files:
        lines.append("### Files changed")
        for f in sorted(all_files):
            lines.append(f"- {f}")
        lines.append("")

    # Review verdicts
    review_verdicts = []
    for step in state.steps:
        if step.verdict and step.step_id in ("code_review", "qa_review"):
            review_verdicts.append(f"- {step.step_id}: {step.verdict}")
    if review_verdicts:
        lines.append("### Review results")
        lines.extend(review_verdicts)
        lines.append("")

    return _truncate("\n".join(lines))


def _render_generic(state: ChainState) -> str:
    """Fallback: render a timeline of all completed steps."""
    lines = ["\n## CHAIN STATE\n"]

    for step in state.steps:
        verdict_str = f" [{step.verdict}]" if step.verdict else ""
        files_str = f" ({len(step.files_modified)} files)" if step.files_modified else ""
        lines.append(f"- {step.step_id} ({step.agent_id}){verdict_str}{files_str}")
        if step.summary:
            # Indent and truncate summary
            for summary_line in step.summary.split("\n")[:3]:
                lines.append(f"  {summary_line}")

    lines.append("")
    return _truncate("\n".join(lines))


def _render_tool_stats(step: StepRecord) -> List[str]:
    """Render tool stats block for review contexts."""
    if not step.tool_stats:
        return []

    stats = step.tool_stats
    total = stats.get("total_calls", 0)
    if total == 0:
        return []

    lines = ["### TOOL USAGE"]

    # Distribution summary — top tools sorted by count
    dist = stats.get("tool_distribution", {})
    if dist:
        parts = [f"{t}: {c}" for t, c in sorted(dist.items(), key=lambda x: -x[1])]
        lines.append(f"{total} calls ({', '.join(parts)})")

    # Duplicate reads
    dupes = stats.get("duplicate_reads", {})
    if dupes:
        dupe_parts = [f"{f}: {c}x" for f, c in sorted(dupes.items(), key=lambda x: -x[1])]
        lines.append(f"{len(dupes)} duplicate reads ({', '.join(dupe_parts)})")

    # Edit density
    density = stats.get("edit_density")
    if density is not None:
        lines.append(f"Edit density: {density * 100:.1f}%")

    lines.append("")
    return lines


def _find_step(state: ChainState, step_id: str) -> Optional[StepRecord]:
    """Find the most recent step with the given ID."""
    for step in reversed(state.steps):
        if step.step_id == step_id:
            return step
    return None


def _truncate(text: str) -> str:
    """Enforce prompt budget on rendered chain state."""
    if len(text) > CHAIN_STATE_MAX_PROMPT_CHARS:
        return text[:CHAIN_STATE_MAX_PROMPT_CHARS] + "\n[chain state truncated]\n"
    return text
