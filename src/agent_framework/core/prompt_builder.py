"""Prompt building module for Agent framework.

Extracts prompt construction logic from the monolithic Agent class.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..workflow.executor import PREVIEW_REVIEW_STEPS

if TYPE_CHECKING:
    from .config import AgentConfig, AgentDefinition, WorkflowDefinition
    from .context_window_manager import ContextWindowManager
    from ..memory.memory_retriever import MemoryRetriever
    from ..memory.tool_pattern_store import ToolPatternStore
    from .session_logger import SessionLogger
    from ..llm.base import LLMBackend
    from ..queue.file_queue import FileQueue
    from ..indexing.query import IndexQuery

from .task import Task, TaskType
from ..utils.type_helpers import get_type_str


@dataclass
class PromptContext:
    """Configuration needed for prompt building.

    Consolidates all dependencies required by PromptBuilder.
    """
    # Core configuration
    config: "AgentConfig"
    workspace: Path

    # Feature toggles
    mcp_enabled: bool

    # Integration configs
    jira_config: Optional[Any] = None
    github_config: Optional[Any] = None

    # Agent definition
    agent_definition: Optional["AgentDefinition"] = None

    # Optimization and memory
    optimization_config: Optional[Dict[str, Any]] = None
    memory_retriever: Optional["MemoryRetriever"] = None
    tool_pattern_store: Optional["ToolPatternStore"] = None
    context_window_manager: Optional["ContextWindowManager"] = None

    # Logging
    session_logger: Optional["SessionLogger"] = None
    logger: Optional[logging.Logger] = None

    # LLM backend (for specialization detection)
    llm: Optional["LLMBackend"] = None

    # Queue (for loading dependency tasks in optimized mode)
    queue: Optional["FileQueue"] = None

    # Agent reference (for callbacks like metrics recording)
    agent: Optional[Any] = None

    # Workflow definitions (for terminal step detection)
    workflows_config: Optional[Dict[str, "WorkflowDefinition"]] = None

    # Codebase indexing (structural code context injection)
    code_index_query: Optional["IndexQuery"] = None
    code_indexing_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.optimization_config is None:
            self.optimization_config = {}


class PromptBuilder:
    """Builds prompts for agent tasks.

    Handles all prompt construction variants:
    - Legacy format (full task JSON)
    - Optimized format (minimal fields)
    - Shadow mode (both for comparison)
    - Memory injection
    - Tool tips injection
    - Replan context injection
    - Human guidance injection
    """

    # Steps where test execution should be suppressed in the LLM prompt.
    # QA handles testing at qa_review; running tests at review/PR steps wastes budget.
    _TEST_SUPPRESSED_STEPS = frozenset({"code_review", "preview_review", "create_pr"})

    # Steps where the LLM must review only — no file writes or implementation.
    # Injects _build_review_only_guidance() (hard "REVIEWER" constraints + VERDICT format).
    _REVIEW_ONLY_STEPS = frozenset({"code_review"})

    # Steps where the architect evaluates a preview plan rather than live code.
    # Injects _build_preview_review_guidance() (lighter constraints, same VERDICT format).
    # Kept separate from _REVIEW_ONLY_STEPS because the guidance text differs —
    # preview_review evaluates a plan, code_review evaluates a diff.
    # Canonical source: workflow.executor.PREVIEW_REVIEW_STEPS (shared with Agent and WorkflowExecutor).
    _PREVIEW_REVIEW_STEPS = PREVIEW_REVIEW_STEPS

    def __init__(self, context: PromptContext):
        """Initialize prompt builder with context.

        Args:
            context: PromptContext containing all dependencies
        """
        self.ctx = context
        self.logger = context.logger or logging.getLogger(__name__)

        # Caches for guidance sections (rebuilt identically per task)
        self._error_handling_guidance: Optional[str] = None
        self._guidance_cache: Dict[str, str] = {}

        # State tracking for specialization (set during build, returned to agent)
        self._current_specialization = None
        self._current_file_count = 0

    def build(self, task: Task) -> str:
        """Build prompt from task.

        Main entry point for prompt construction. Handles all optimization
        strategies and feature flags.

        Returns:
            Fully constructed prompt string ready for LLM
        """
        shadow_mode = self.ctx.optimization_config.get("shadow_mode", False)
        use_optimizations = self._should_use_optimization(task)

        # Detect specialization once — reused for both prompt and team composition.
        # Returns (profile, files) so we avoid calling detect_file_patterns twice.
        from .engineer_specialization import apply_specialization_to_prompt
        profile, files = self._detect_engineer_specialization(task)
        self._current_specialization = profile
        self._current_file_count = len(files)
        prompt_text = apply_specialization_to_prompt(self.ctx.config.prompt, profile)

        # Note: Activity manager update moved to Agent class after build()

        # Determine which prompt to use
        if shadow_mode:
            prompt = self._handle_shadow_mode_comparison(task, prompt_override=prompt_text)
        elif use_optimizations:
            prompt = self._build_prompt_optimized(task, prompt_override=prompt_text)
        else:
            prompt = self._build_prompt_legacy(task, prompt_override=prompt_text)

        # Inject requirements checklist so engineer knows exactly what to deliver
        prompt = self._inject_requirements_checklist(prompt, task)

        # Inject preview mode constraints when task is a preview
        if task.type == TaskType.PREVIEW:
            prompt = self._inject_preview_mode(prompt, task)

        # Log prompt preview for debugging (sanitized)
        if self.logger.isEnabledFor(logging.DEBUG):
            prompt_preview = prompt[:500].replace(task.id, "TASK_ID")
            if hasattr(task, 'context') and task.context.get('jira_key'):
                prompt_preview = prompt_preview.replace(task.context['jira_key'], "JIRA-XXX")
            self.logger.debug(f"Built prompt preview (first 500 chars): {prompt_preview}...")

        # Append test failure context if present
        prompt = self._append_test_failure_context(prompt, task)

        # Inject relevant memories from previous tasks
        prompt = self._inject_memories(prompt, task)

        # Inject tool efficiency tips from session analysis
        prompt = self._inject_tool_tips(prompt, task)

        # Inject structural codebase overview and relevant symbols
        prompt = self._inject_codebase_index(prompt, task)

        # Inject read cache from previous chain steps
        prompt = self._inject_read_cache(prompt, task)

        # Inject self-eval critique if retrying after failed self-evaluation
        prompt = self._inject_self_eval_context(prompt, task)

        # Inject replan history if retrying with revised approach
        prompt = self._inject_replan_context(prompt, task)

        # Inject retry context: truncated error + partial progress from previous attempt
        prompt = self._inject_retry_context(prompt, task)

        # Inject human guidance if provided via `agent guide` command
        prompt = self._inject_human_guidance(prompt, task)

        # Session log: capture what was sent to the LLM
        if self.ctx.session_logger:
            prompt_hash = hashlib.md5(prompt.encode(), usedforsecurity=False).hexdigest()[:12]
            has_replan = "_revised_plan" in task.context
            has_self_eval = "_self_eval_critique" in task.context and "_revised_plan" not in task.context
            self.ctx.session_logger.log(
                "prompt_built",
                prompt_length=len(prompt),
                prompt_hash=prompt_hash,
                replan_injected=has_replan,
                self_eval_injected=has_self_eval,
                retry=task.retry_count,
            )
            self.ctx.session_logger.log_prompt(prompt)

        return prompt

    def get_current_specialization(self):
        """Get the specialization profile detected during last build() call.

        Returns:
            SpecializationProfile or None
        """
        return self._current_specialization

    def get_current_file_count(self) -> int:
        """Get the file count detected during last build() call.

        Returns:
            Number of files detected for specialization
        """
        return self._current_file_count

    def _should_use_optimization(self, task: Task) -> bool:
        """Check if optimization strategies should be used for this task.

        Uses canary rollout to gradually enable optimizations.
        Task-level overrides take precedence over canary selection.
        """
        # Check for task-level override first
        if hasattr(task, 'optimization_override') and task.optimization_override is not None:
            reason = getattr(task, 'optimization_override_reason', 'no reason given')
            self.logger.info(
                f"Task {task.id} optimization override: {task.optimization_override} ({reason})"
            )
            return task.optimization_override

        canary_pct = self.ctx.optimization_config.get("canary_percentage", 0)

        if canary_pct == 0:
            return False
        elif canary_pct >= 100:
            return True
        else:
            # Use deterministic hash for consistent selection
            task_hash = int(hashlib.md5(task.id.encode(), usedforsecurity=False).hexdigest()[:8], 16)
            return (task_hash % 100) < canary_pct

    def _build_prompt_legacy(self, task: Task, prompt_override: str = None) -> str:
        """Build prompt using legacy format (original implementation)."""
        task_json = task.model_dump_json(indent=2)
        agent_prompt = prompt_override or self.ctx.config.prompt

        # Extract integration context
        jira_key = task.context.get("jira_key")
        github_repo = task.context.get("github_repo")
        jira_project = task.context.get("jira_project")

        mcp_guidance = ""

        # Build MCP guidance sections
        if self.ctx.mcp_enabled:
            if jira_key or jira_project:
                mcp_guidance += self._build_jira_guidance(jira_key, jira_project)
            if github_repo:
                mcp_guidance += self._build_github_guidance(github_repo, jira_key)
            mcp_guidance += self._build_error_handling_guidance()

        # Intermediate chain steps must not create PRs — the terminal step handles that
        if task.context.get("chain_step") and not self._is_at_terminal_workflow_step(task):
            mcp_guidance += """
IMPORTANT: You are an intermediate step in the workflow chain.
Push your commits but do NOT create a pull request.
The PR will be created by a downstream agent after all steps complete.

"""

        # Subtasks must not create PRs — the fan-in task creates a single PR
        if task.parent_task_id is not None:
            mcp_guidance += """
IMPORTANT: You are a SUBTASK of a decomposed task.
Commit and push your changes, but do NOT create a pull request.
A fan-in task will aggregate all subtask results and create a single PR.

"""

        # Suppress redundant test execution at review/PR steps — QA handles testing
        workflow_step = task.context.get("workflow_step")
        if workflow_step in self._TEST_SUPPRESSED_STEPS:
            mcp_guidance += """
IMPORTANT: Do NOT run the test suite (pytest, go test, npm test, etc.) during this step.
Testing is handled by the QA agent at the qa_review step — running tests here wastes time and budget.
Focus on reviewing the code changes.

"""

        # Review-only steps must not modify code — only inspect the diff
        if workflow_step in self._REVIEW_ONLY_STEPS:
            mcp_guidance += self._build_review_only_guidance()
        elif workflow_step in self._PREVIEW_REVIEW_STEPS:
            mcp_guidance += self._build_preview_review_guidance()

        # Load upstream context from previous agent if available
        upstream_context = self._load_upstream_context(task)

        # Load pre-scan findings from parallel QA scan if available
        prescan_context = self._load_pre_scan_findings(task)

        # Render plan as human-readable section (empty string when no plan)
        plan_section = self._render_plan_section(task)

        # Per-step instructions override (e.g. create_pr gets PR-specific instructions)
        step_instructions = self._build_step_instructions_section(task)

        return f"""You are {self.ctx.config.id}.

TASK DETAILS:
{task_json}

{mcp_guidance}{upstream_context}{prescan_context}{plan_section}{step_instructions}
YOUR RESPONSIBILITIES:
{agent_prompt}

IMPORTANT:
- Complete the task described above
- This task will be automatically marked as completed when you're done
"""

    def _build_prompt_optimized(self, task: Task, prompt_override: str = None) -> str:
        """Build prompt using optimized format.

        Applies:
        - Strategy 1: Minimal task prompts (only essential fields)
        - Strategy 3: Context deduplication (no redundant info)
        - Strategy 4: Compact JSON (no whitespace)
        - Strategy 5: Result summarization (include dep summaries)
        """
        # Always use minimal fields in optimized prompts (Strategy 1)
        task_dict = self._get_minimal_task_dict(task)

        # Always use compact JSON in optimized prompts (Strategy 4)
        task_json = json.dumps(task_dict, separators=(',', ':'))

        # Extract integration context (only dynamic values, not boilerplate)
        jira_key = task.context.get("jira_key")
        github_repo = task.context.get("github_repo")
        jira_project = task.context.get("jira_project")

        # Build minimal MCP context (just dynamic values, no boilerplate)
        context_note = ""
        if self.ctx.mcp_enabled:
            if jira_key:
                context_note += f"JIRA Ticket: {jira_key}\n"
            if jira_project:
                context_note += f"JIRA Project: {jira_project}\n"
            if github_repo:
                context_note += f"GitHub Repository: {github_repo}\n"

        # Include dependency results (Strategy 5: Result Summarization)
        dep_context = ""
        enable_summarization = self.ctx.optimization_config.get("enable_result_summarization", False)
        if enable_summarization and task.depends_on and self.ctx.queue:
            dep_context = "\nPREVIOUS WORK:\n"
            for dep_id in task.depends_on:
                dep_task = self.ctx.queue.get_completed(dep_id)
                if dep_task and dep_task.result_summary:
                    dep_context += f"- {dep_task.title}: {dep_task.result_summary}\n"

        # Intermediate chain steps must not create PRs
        chain_note = ""
        if task.context.get("chain_step") and not self._is_at_terminal_workflow_step(task):
            chain_note += "\nIMPORTANT: You are an intermediate step in the workflow chain.\nPush your commits but do NOT create a pull request.\n"

        # Subtasks must not create PRs — fan-in handles it
        if task.parent_task_id is not None:
            chain_note += "\nIMPORTANT: You are a SUBTASK of a decomposed task.\nCommit and push your changes, but do NOT create a pull request.\nA fan-in task will aggregate all subtask results and create a single PR.\n"

        # Suppress redundant test execution at review/PR steps
        workflow_step = task.context.get("workflow_step")
        if workflow_step in self._TEST_SUPPRESSED_STEPS:
            chain_note += "\nIMPORTANT: Do NOT run the test suite (pytest, go test, npm test, etc.) during this step.\nTesting is handled by the QA agent at the qa_review step — running tests here wastes time and budget.\nFocus on reviewing the code changes.\n"

        # Review-only steps must not modify code — only inspect the diff
        if workflow_step in self._REVIEW_ONLY_STEPS:
            chain_note += "\n" + self._build_review_only_guidance()
        elif workflow_step in self._PREVIEW_REVIEW_STEPS:
            chain_note += "\n" + self._build_preview_review_guidance()

        # Load upstream context from previous agent if available
        upstream_context = self._load_upstream_context(task)

        # Load pre-scan findings from parallel QA scan if available
        prescan_context = self._load_pre_scan_findings(task)

        # Render plan as human-readable section (empty string when no plan)
        plan_section = self._render_plan_section(task)

        # Per-step instructions override (e.g. create_pr gets PR-specific instructions)
        step_instructions = self._build_step_instructions_section(task)

        # Build optimized prompt (shorter, focused on essentials)
        agent_prompt = prompt_override or self.ctx.config.prompt
        return f"""You are {self.ctx.config.id}.

TASK:
{task_json}

{context_note}{dep_context}{chain_note}{upstream_context}{prescan_context}{plan_section}{step_instructions}
{agent_prompt}

IMPORTANT:
- Complete the task described above
- This task will be automatically marked as completed when you're done
"""

    def _handle_shadow_mode_comparison(self, task: Task, prompt_override: str = None) -> str:
        """Generate and compare both prompts in shadow mode, return legacy prompt."""
        legacy_prompt = self._build_prompt_legacy(task, prompt_override=prompt_override)
        optimized_prompt = self._build_prompt_optimized(task, prompt_override=prompt_override)

        # Log comparison
        legacy_len = len(legacy_prompt)
        optimized_len = len(optimized_prompt)
        savings = legacy_len - optimized_len
        savings_pct = (savings / legacy_len * 100) if legacy_len > 0 else 0

        # Truncate task ID for security
        task_id_short = task.id[:8] + "..." if len(task.id) > 8 else task.id

        self.logger.debug(
            f"[SHADOW MODE] Task {task_id_short} prompt comparison: "
            f"legacy={legacy_len} chars, optimized={optimized_len} chars, "
            f"savings={savings} chars ({savings_pct:.1f}%)"
        )

        # Record metrics for analysis (if agent callback is available)
        if self.ctx.agent and hasattr(self.ctx.agent, '_record_optimization_metrics'):
            self.ctx.agent._record_optimization_metrics(task, legacy_len, optimized_len)

        # Return legacy prompt (no behavioral change in shadow mode)
        return legacy_prompt

    def _is_at_terminal_workflow_step(self, task: Task) -> bool:
        """Check if the current agent is at the last step in the workflow DAG.

        Returns True for standalone tasks (no workflow) to preserve backward
        compatibility — standalone agents should always be allowed to create PRs.
        """
        workflow_name = task.context.get("workflow")
        if not workflow_name or not self.ctx.workflows_config or workflow_name not in self.ctx.workflows_config:
            return True

        workflow_def = self.ctx.workflows_config[workflow_name]
        try:
            dag = workflow_def.to_dag(workflow_name)
        except Exception:
            return True

        # Prefer explicit workflow_step from chain context
        step_id = task.context.get("workflow_step")
        if step_id and step_id in dag.steps:
            return dag.is_terminal_step(step_id)

        # Fallback: find the step for this agent's base_id
        for step in dag.steps.values():
            if step.agent == self.ctx.config.base_id:
                return dag.is_terminal_step(step.id)

        # If we can't determine the step, assume terminal (safer default)
        return True

    def _get_minimal_task_dict(self, task: Task) -> Dict[str, Any]:
        """Extract only prompt-relevant task fields.

        Implements Strategy 1 (Minimal Task Prompts) from optimization plan.
        Omits metadata fields that don't contribute to task execution.
        """
        # Validate essential fields
        if not task.title or not task.description:
            self.logger.warning(
                f"Task {task.id} missing essential fields: "
                f"title={bool(task.title)}, description={bool(task.description)}. "
                f"Falling back to full task dict."
            )
            return task.model_dump()

        minimal = {
            "title": task.title.strip(),
            "description": task.description.strip(),
            "type": get_type_str(task.type),
        }

        # Include acceptance criteria and deliverables if present
        if task.acceptance_criteria:
            minimal["acceptance_criteria"] = task.acceptance_criteria
        if task.deliverables:
            minimal["deliverables"] = task.deliverables

        # Include notes if non-empty (can contain important context)
        if task.notes:
            minimal["notes"] = task.notes

        # Include only relevant context keys
        relevant_context = {}
        for key in ["jira_key", "jira_project", "github_repo", "mode", "user_goal", "repository_name", "epic_key"]:
            if key in task.context:
                relevant_context[key] = task.context[key]

        if relevant_context:
            minimal["context"] = relevant_context

        return minimal

    def _build_jira_guidance(self, jira_key: str, jira_project: str) -> str:
        """Build JIRA integration guidance for MCP."""
        can_create = (
            self.ctx.agent_definition is not None
            and self.ctx.agent_definition.jira_can_create_tickets
        )
        cache_key = f"jira:{jira_key}:{jira_project}:{can_create}"
        if cache_key in self._guidance_cache:
            return self._guidance_cache[cache_key]

        jira_server = self.ctx.jira_config.server if self.ctx.jira_config else "jira.example.com"

        tools = [
            f'- Search issues: jira_search_issues(jql="project = {jira_project or "PROJ"}")',
            f'- Get issue: jira_get_issue(issueKey="{jira_key or "PROJ-123"}")',
        ]

        if can_create:
            tools.extend([
                f'- Create ticket: jira_create_issue(project="{jira_project or "PROJ"}", summary="...", description="...", issueType="Story")',
                f'- Create epic: jira_create_epic(project="{jira_project or "PROJ"}", title="...", description="...")',
                f'- Create subtask: jira_create_subtask(parentKey="{jira_key or "PROJ-123"}", summary="...", description="...")',
            ])

        tools.extend([
            f'- Update status: jira_transition_issue(issueKey="{jira_key or "PROJ-123"}", transitionName="In Progress")',
            f'- Add comment: jira_add_comment(issueKey="{jira_key or "PROJ-123"}", comment="...")',
        ])

        tools_block = "\n".join(tools)

        restriction = ""
        if not can_create:
            restriction = (
                "\nIMPORTANT RESTRICTIONS:\n"
                "- Do NOT create JIRA tickets — the architect handles ticket creation\n"
                "- Do NOT use Bash, curl, or urllib to call the JIRA API directly\n"
            )

        result = f"""
JIRA INTEGRATION (via MCP):
You have access to JIRA via MCP tools:
{tools_block}

Current context:
- JIRA Server: {jira_server}
- Ticket: {jira_key or 'N/A'}
- Project: {jira_project or 'N/A'}
{restriction}
"""
        self._guidance_cache[cache_key] = result
        return result

    def _build_github_guidance(self, github_repo: str, jira_key: str) -> str:
        """Build GitHub integration guidance for MCP."""
        cache_key = f"github:{github_repo}:{jira_key}"
        if cache_key in self._guidance_cache:
            return self._guidance_cache[cache_key]

        owner, repo = github_repo.split("/")

        # Get formatting patterns from config
        branch_pattern = "{type}/{ticket_id}-{slug}"
        pr_title_pattern = "[{ticket_id}] {title}"
        if self.ctx.github_config:
            branch_pattern = self.ctx.github_config.branch_pattern
            pr_title_pattern = self.ctx.github_config.pr_title_pattern

        result = f"""
GITHUB INTEGRATION (via MCP):
Repository: {github_repo}
Branch naming: Use pattern "{branch_pattern}"
  Example: feature/{jira_key or 'PROJ-123'}-add-authentication
PR title: Use pattern "{pr_title_pattern}"
  Example: [{jira_key or 'PROJ-123'}] Add authentication feature

Available tools:
- github_create_pr(owner="{owner}", repo="{repo}",
                   title="[{jira_key or 'PROJ-123'}] Your Title",
                   body="...",
                   head="feature/{jira_key or 'PROJ-123'}-slug")
- github_add_pr_comment(owner="{owner}", repo="{repo}", prNumber=123, body="...")
- github_link_pr_to_jira(owner="{owner}", repo="{repo}", prNumber=123, jiraKey="{jira_key or 'PROJ-123'}")

NOTE: You are responsible for committing and pushing your changes.

Workflow coordination:
1. Make your code changes
2. Commit changes: git add <files> && git commit -m "[TICKET] description"
3. Push to feature branch: git push
4. Create a PR using github_create_pr (if your workflow requires it)
5. Update JIRA using jira_transition_issue and jira_add_comment

"""
        self._guidance_cache[cache_key] = result
        return result

    def _build_review_only_guidance(self) -> str:
        """Build constraints for review-only steps (e.g. code_review).

        Prevents the reviewer from re-implementing code instead of reviewing it.
        """
        return """IMPORTANT — CODE REVIEW CONSTRAINTS:
You are a REVIEWER, not an implementer. You must NOT modify any files.

- Do NOT use Write, Edit, or NotebookEdit tools
- Do NOT use Bash to create, modify, or delete files
- Do NOT spawn implementation subagents or delegate fixes
- Only explore files that appear in the diff (use Read, Grep, Glob, git diff)
- If changes are missing or incorrect, report them as findings — do NOT fix them yourself

Your output MUST end with exactly one of:
  VERDICT: APPROVE
  VERDICT: REQUEST_CHANGES

"""

    def _build_preview_review_guidance(self) -> str:
        """Build constraints for the preview_review step.

        At preview_review the architect evaluates the engineer's structured preview
        (files to modify, approach, risks) and decides whether to authorize
        proceeding to implementation. No code modifications — evaluate the plan only.
        """
        return """IMPORTANT — PREVIEW REVIEW CONSTRAINTS:
You are evaluating an EXECUTION PREVIEW produced by the engineer. Do NOT modify any files.

- Do NOT use Write, Edit, or NotebookEdit tools
- Do NOT use Bash to create, modify, or delete files
- Read source files only if you need context to evaluate the plan

Evaluate the engineer's preview for:
- Correctness: does the proposed approach actually solve the task?
- Completeness: are all necessary changes identified?
- Risk: are edge cases and backward compatibility concerns addressed?
- Scope: is anything added that wasn't requested?

Your output MUST end with exactly one of:
  VERDICT: APPROVE     (authorize the engineer to proceed with implementation)
  VERDICT: REQUEST_CHANGES  (send back with specific required changes)

"""

    def _build_step_instructions_section(self, task: Task) -> str:
        """Build a high-priority instructions section from per-step config.

        When a workflow step defines custom instructions (via YAML), they override
        the general agent prompt for that step. This prevents e.g. the create_pr
        step from receiving generic architect planning instructions.
        """
        instructions = task.context.get("_step_instructions")
        if instructions is None:
            return ""

        step_name = task.context.get("workflow_step", "unknown")
        return (
            f"## CURRENT STEP: {step_name}\n\n"
            f"## STEP INSTRUCTIONS\n"
            f"{instructions}\n\n"
            "You MUST follow the step instructions above. "
            "Ignore any conflicting guidance in the agent responsibilities section below.\n\n"
        )

    def _build_error_handling_guidance(self) -> str:
        """Build error handling guidance for MCP tools."""
        if self._error_handling_guidance is not None:
            return self._error_handling_guidance
        self._error_handling_guidance = """
ERROR HANDLING:
If a tool call fails:
1. Read the error message carefully
2. If rate limited, wait and retry
3. If authentication failed, report failure
4. If invalid input, correct and retry
5. For partial failures (e.g., PR created but JIRA update failed):
   - Retry the failed operation
   - If still fails, leave completed operations and report the failure
   - Do NOT try to undo successful operations

"""
        return self._error_handling_guidance

    def _load_upstream_context(self, task: Task) -> str:
        """Load upstream agent's findings from chain state, inline context, or disk.

        Priority cascade:
        1. Rejection feedback (human override)
        2. Chain state file (structured, step-aware rendering)
        3. Structured findings (QA checklist)
        4. Inline upstream_summary (raw text fallback)
        5. Disk file (last resort)
        """
        # Rejection feedback takes top priority — human said "redo this"
        rejection_feedback = task.context.get("rejection_feedback")
        if rejection_feedback and rejection_feedback.strip():
            return (
                "\n## HUMAN FEEDBACK — CHECKPOINT REJECTED\n"
                "Your previous output at this step was reviewed and rejected. "
                "Address the following feedback:\n\n"
                f"{rejection_feedback}\n"
            )

        # Chain state: structured, step-appropriate context rendering
        chain_context = self._load_chain_state_context(task)
        if chain_context:
            return chain_context

        # Structured findings take priority — they give the engineer precise, actionable items
        structured = task.context.get("structured_findings")
        if structured:
            formatted = self._format_structured_findings(structured)
            if formatted:
                return formatted

        # Prefer inline context — works across worktrees where file path may not resolve
        inline = task.context.get("upstream_summary")
        if inline:
            return f"\n## UPSTREAM AGENT FINDINGS\n{inline}\n"

        context_file = task.context.get("upstream_context_file")
        if not context_file:
            return ""

        try:
            context_path = Path(context_file).resolve()
            summaries_dir = (self.ctx.workspace / ".agent-context" / "summaries").resolve()

            # Only read files inside our summaries directory
            if not str(context_path).startswith(str(summaries_dir)):
                self.logger.warning(f"Upstream context path outside summaries dir: {context_file}")
                return ""

            if not context_path.exists():
                return ""

            content = context_path.read_text(encoding="utf-8")
            if not content.strip():
                return ""

            return f"\n## UPSTREAM AGENT FINDINGS\n{content}\n"
        except Exception as e:
            self.logger.debug(f"Failed to load upstream context: {e}")
            return ""

    def _load_chain_state_context(self, task: Task) -> str:
        """Load chain state and render step-appropriate context.

        Returns empty string if no chain state exists, letting the cascade
        fall through to legacy mechanisms.
        """
        # Only applies to workflow chain tasks
        if not (task.context.get("workflow") or task.context.get("chain_step")):
            return ""

        try:
            from .chain_state import load_chain_state, render_for_step

            root_task_id = task.root_id
            state = load_chain_state(self.ctx.workspace, root_task_id)
            if not state or not state.steps:
                return ""

            consumer_step = task.context.get("workflow_step", "")
            rendered = render_for_step(state, consumer_step)

            if rendered and rendered.strip():
                self.logger.debug(
                    f"Chain state: rendered {len(rendered)} chars for "
                    f"step {consumer_step!r} ({len(state.steps)} steps in chain)"
                )
                return rendered
        except Exception as e:
            self.logger.debug(f"Failed to load chain state context: {e}")

        return ""

    def _load_pre_scan_findings(self, task: Task) -> str:
        """Load QA pre-scan findings from disk for injection into prompts.

        Only injects for engineer and qa agents on chain tasks (has workflow_step).
        Formats differently per agent role.
        """
        # Only inject for engineer and qa on workflow chain tasks
        if self.ctx.config.base_id not in ("engineer", "qa"):
            return ""
        if not task.context.get("workflow_step"):
            return ""

        root_task_id = task.root_id
        findings_file = (
            self.ctx.workspace / ".agent-communication" / "pre-scans" / f"{root_task_id}.json"
        )

        if not findings_file.exists():
            return ""

        try:
            data = json.loads(findings_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            self.logger.debug(f"Failed to load pre-scan findings: {e}")
            return ""

        findings = data.get("structured_findings", {})
        findings_list = findings.get("findings", [])
        raw_summary = data.get("raw_summary", "")

        if not findings_list and not raw_summary:
            return ""

        if self.ctx.config.base_id == "engineer":
            return self._format_prescan_for_engineer(findings_list, raw_summary)
        else:
            return self._format_prescan_for_qa(findings_list, raw_summary)

    def _format_prescan_for_engineer(self, findings: list, raw_summary: str) -> str:
        """Format pre-scan findings as actionable items for the engineer."""
        lines = ["\n## QA PRE-SCAN FINDINGS (from parallel scan)\n"]
        lines.append("These findings were detected by an automated QA pre-scan running "
                      "in parallel with code review. Address them alongside architect feedback.\n")

        if findings:
            for i, f in enumerate(findings, 1):
                severity = f.get("severity", "UNKNOWN")
                desc = f.get("description", "")
                file_path = f.get("file", "")
                line_no = f.get("line_number") or f.get("line")
                location = f"{file_path}:{line_no}" if line_no else file_path
                lines.append(f"{i}. [{severity}] {location} — {desc}")
                suggested = f.get("suggested_fix")
                if suggested:
                    lines.append(f"   Fix: {suggested}")
        elif raw_summary:
            lines.append(raw_summary[:2000])

        lines.append("")
        return "\n".join(lines)

    def _format_prescan_for_qa(self, findings: list, raw_summary: str) -> str:
        """Format pre-scan results for the QA agent's full review."""
        lines = ["\n## PRE-SCAN RESULTS (your earlier parallel scan)\n"]
        lines.append("You ran a pre-scan earlier. These findings may be stale if "
                      "the engineer made fixes since then. Focus on deeper issues "
                      "and re-verify any pre-scan items that weren't addressed.\n")

        if findings:
            for i, f in enumerate(findings, 1):
                severity = f.get("severity", "UNKNOWN")
                desc = f.get("description", "")
                file_path = f.get("file", "")
                lines.append(f"{i}. [{severity}] {file_path} — {desc}")
        elif raw_summary:
            lines.append(raw_summary[:2000])

        lines.append("")
        return "\n".join(lines)

    def _format_structured_findings(self, structured: dict) -> str:
        """Format structured QA findings as a file-grouped actionable checklist.

        Args:
            structured: Dict with 'findings' list and optional summary/counts.

        Returns:
            Formatted prompt section string.
        """
        findings = structured.get("findings", [])
        if not findings:
            return ""

        # Group findings by file
        by_file: Dict[str, list] = {}
        for f in findings:
            key = f.get("file", "unknown")
            by_file.setdefault(key, []).append(f)

        lines = ["\n## QA FINDINGS — ACTION REQUIRED\n"]

        num = 1
        for filepath, file_findings in by_file.items():
            lines.append(f"### {filepath}")
            for f in file_findings:
                severity = f.get("severity", "UNKNOWN")
                desc = f.get("description", "")
                line_no = f.get("line_number")
                location = f":{line_no}" if line_no else ""
                lines.append(f"{num}. [{severity}] {filepath}{location} — {desc}")
                suggested = f.get("suggested_fix")
                if suggested:
                    lines.append(f"   Fix: {suggested}")
                num += 1
            lines.append("")

        lines.append("Address all findings above, then re-run tests.\n")
        return "\n".join(lines)

    def _render_plan_section(self, task: Task) -> str:
        """Render task.plan as a human-readable prompt section.

        Returns empty string when plan is None so callers can unconditionally
        include the result without conditional checks.
        """
        if task.plan is None:
            return ""

        plan = task.plan
        lines = ["IMPLEMENTATION PLAN:"]

        if plan.objectives:
            lines.append("Objectives:")
            for obj in plan.objectives:
                lines.append(f"- {obj}")

        if plan.approach:
            lines.append("Approach:")
            for i, step in enumerate(plan.approach, 1):
                lines.append(f"{i}. {step}")

        if plan.files_to_modify:
            lines.append(f"Files to modify: {', '.join(plan.files_to_modify)}")

        if plan.risks:
            lines.append("Risks:")
            for risk in plan.risks:
                lines.append(f"- {risk}")

        if plan.success_criteria:
            lines.append("Success criteria:")
            for criterion in plan.success_criteria:
                lines.append(f"- {criterion}")

        lines.append("")
        return "\n".join(lines)

    def _inject_requirements_checklist(self, prompt: str, task: Task) -> str:
        """Inject structured deliverables checklist from task context.

        All agents (engineer, QA, code review) see the same checklist —
        engineer uses it as a contract, reviewers use it for verification.
        """
        checklist = task.context.get("requirements_checklist")
        if not checklist:
            return prompt

        count = len(checklist)
        lines = [f"\n## REQUIRED DELIVERABLES ({count} items)"]
        lines.append(
            "You MUST implement ALL of the following. Commit after completing each one."
        )

        for item in checklist:
            item_id = item.get("id", "?")
            desc = item.get("description", "")
            files = item.get("files", [])
            file_hint = f" ({', '.join(files)})" if files else ""
            lines.append(f"{item_id}. [ ] {desc}{file_hint}")

        lines.append("")
        lines.append(
            "Before reporting completion, verify each item above was implemented.\n"
        )

        return prompt + "\n".join(lines)

    def _inject_memories(self, prompt: str, task: Task) -> str:
        """Append relevant memories from previous tasks to the prompt."""
        if not self.ctx.memory_retriever:
            return prompt

        repo_slug = task.context.get("github_repo")
        if not repo_slug:
            return prompt

        # Query context window manager for memory budget
        max_chars = None
        if self.ctx.context_window_manager:
            max_chars = self.ctx.context_window_manager.compute_memory_budget()
            if max_chars == 0:
                self.logger.debug("Skipping memory injection: context budget critical (>90% used)")
                return prompt

        # Build tag hints from task context
        task_tags = []
        if task.type:
            task_tags.append(get_type_str(task.type))
        jira_project = task.context.get("jira_project")
        if jira_project:
            task_tags.append(jira_project)

        memory_section = self.ctx.memory_retriever.format_for_prompt(
            repo_slug=repo_slug,
            agent_type=self.ctx.config.base_id,
            task_tags=task_tags,
            max_chars=max_chars,
        )

        if memory_section:
            self.logger.debug(f"Injected {len(memory_section)} chars of memory context (max={max_chars or 3000})")
            if self.ctx.session_logger:
                self.ctx.session_logger.log(
                    "memory_recall",
                    repo=repo_slug,
                    chars_injected=len(memory_section),
                    categories=task_tags,
                )
            return prompt + "\n" + memory_section

        return prompt

    def _inject_tool_tips(self, prompt: str, task: Task) -> str:
        """Append tool efficiency tips from previous session analysis."""
        if not self.ctx.tool_pattern_store:
            return prompt

        repo_slug = task.context.get("github_repo")
        if not repo_slug:
            return prompt

        max_count = self.ctx.optimization_config.get("tool_tips_max_count", 5)
        max_chars = self.ctx.optimization_config.get("tool_tips_max_chars", 1500)
        patterns = self.ctx.tool_pattern_store.get_top_patterns(
            repo_slug, limit=max_count, max_chars=max_chars,
        )
        if not patterns:
            return prompt

        tips_lines = [f"- {p.tip}" for p in patterns]
        tips_section = "## Tool Efficiency Tips\n\n" + "\n".join(tips_lines)

        self.logger.debug(f"Injected {len(patterns)} tool tips ({len(tips_section)} chars)")
        if self.ctx.session_logger:
            self.ctx.session_logger.log(
                "tool_tips_injected",
                repo=repo_slug,
                count=len(patterns),
                chars=len(tips_section),
            )
        return prompt + "\n\n" + tips_section

    def _inject_codebase_index(self, prompt: str, task: Task) -> str:
        """Append structural codebase overview and relevant symbols to the prompt."""
        if not self.ctx.code_index_query:
            return prompt

        cfg = self.ctx.code_indexing_config or {}
        inject_for = cfg.get("inject_for_agents", ["architect", "engineer", "qa"])
        if self.ctx.config.base_id not in inject_for:
            return prompt

        repo_slug = task.context.get("github_repo")
        if not repo_slug:
            return prompt

        # Context-window-aware budget — shrinks index when context is tight
        max_chars = cfg.get("max_prompt_chars", 4000)
        if self.ctx.context_window_manager:
            budget = self.ctx.context_window_manager.compute_memory_budget()
            if budget == 0:
                self.logger.debug("Skipping codebase index: context budget critical")
                return prompt
            max_chars = min(max_chars, budget)

        # Planning tasks get architectural overview; implementation tasks get scored symbols
        is_planning = task.type in (TaskType.PLANNING, TaskType.PREVIEW)
        if is_planning:
            index_section = self.ctx.code_index_query.format_overview_only(repo_slug)
        else:
            task_goal = self._build_index_query_goal(task)
            index_section = self.ctx.code_index_query.query_for_prompt(
                repo_slug, task_goal, max_chars=max_chars,
            )

        if not index_section:
            return prompt

        self.logger.debug(f"Injected {len(index_section)} chars of codebase index context")
        if self.ctx.session_logger:
            self.ctx.session_logger.log(
                "codebase_index_injected", repo=repo_slug, chars=len(index_section),
            )
        return prompt + "\n\n" + index_section

    def _build_index_query_goal(self, task: Task) -> str:
        """Build a rich query string from all available task context.

        Pulls keywords from multiple sources so symbol scoring
        matches against the actual problem space, not just the title.
        """
        parts = []
        if task.title:
            parts.append(task.title)
        if task.description:
            parts.append(task.description)
        # user_goal often has more detail than title
        user_goal = task.context.get("user_goal")
        if user_goal and user_goal != task.title:
            parts.append(user_goal)
        # Upstream findings mention specific files/classes
        upstream = task.context.get("upstream_summary", "")
        if upstream:
            parts.append(upstream[:1000])
        # Structured findings have precise file paths
        findings = task.context.get("structured_findings")
        if findings and isinstance(findings, dict):
            for file_path in findings.keys():
                parts.append(file_path)
        return " ".join(parts)

    # Budget for read cache section — keeps prompt size under control
    _READ_CACHE_MAX_CHARS = 6000

    @staticmethod
    def _repo_cache_slug(github_repo: str) -> str:
        """Convert 'owner/repo' to 'owner-repo' for cache file naming."""
        return github_repo.replace("/", "-")

    def _display_path(self, full_path: str) -> str:
        """Strip workspace prefix for shorter table display."""
        ws = str(self.ctx.workspace)
        if full_path.startswith(ws):
            return full_path[len(ws):].lstrip("/")
        return full_path

    @staticmethod
    def _migrate_legacy_cache_paths(entries: dict) -> dict:
        """Convert legacy absolute-path cache keys to repo-relative.

        Old cache entries used absolute worktree paths like:
          /path/to/worktrees/owner/repo/agent-key/src/file.py
        New entries use repo-relative paths: src/file.py

        Read-time migration only — doesn't rewrite the cache file.
        """
        import re
        migrated = {}
        # Matches worktree paths: .../worktrees/<owner>/<repo>/<agent-key>/rest
        worktree_pattern = re.compile(r".*/worktrees/[^/]+/[^/]+/[^/]+/(.*)")
        for path, entry in entries.items():
            if not path.startswith("/"):
                migrated[path] = entry
                continue
            m = worktree_pattern.match(path)
            if m:
                rel = m.group(1)
                # Don't overwrite an entry that already exists under the relative key
                if rel not in migrated:
                    migrated[rel] = entry
            else:
                migrated[path] = entry
        return migrated

    def _seed_from_repo_cache(self, cache_dir: Path, cache_file: Path, task: Task) -> bool:
        """Seed task-specific cache from repo-scoped cache.

        On a new attempt, there's no task-specific cache yet. If a repo-scoped
        cache exists from prior attempts, copy its entries so downstream steps
        and MCP tools work transparently.

        Returns True if seeded successfully.
        """
        try:
            github_repo = task.context.get("github_repo") if task.context else None
            if not github_repo:
                return False

            slug = self._repo_cache_slug(github_repo)
            repo_cache_file = cache_dir / f"_repo-{slug}.json"
            if not repo_cache_file.exists():
                return False

            repo_data = json.loads(repo_cache_file.read_text(encoding="utf-8"))
            if not isinstance(repo_data, dict):
                return False

            entries = repo_data.get("entries", {})
            if not entries:
                return False

            # Write as task-specific cache so MCP tools work transparently
            from ..utils.atomic_io import atomic_write_text

            cache_data = {
                "root_task_id": task.root_id,
                "entries": entries,
            }
            cache_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(cache_file, json.dumps(cache_data))
            self.logger.debug(
                f"Seeded read cache from repo cache ({len(entries)} entries) for {task.root_id}"
            )
            return True
        except Exception as e:
            self.logger.debug(f"Failed to seed from repo cache: {e}")
            return False

    def _inject_read_cache(self, prompt: str, task: Task) -> str:
        """Inject read cache manifest from previous chain steps.

        Tells the LLM which files were already analyzed so it can skip
        redundant reads and use get_cached_reads() for details.
        """
        cache_dir = self.ctx.workspace / ".agent-communication" / "read-cache"
        root_task_id = task.root_id
        cache_file = cache_dir / f"{root_task_id}.json"

        if not cache_file.exists():
            if not self._seed_from_repo_cache(cache_dir, cache_file, task):
                return prompt

        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return prompt
        except (json.JSONDecodeError, OSError) as e:
            self.logger.debug(f"Failed to load read cache: {e}")
            return prompt

        entries = data.get("entries", {})
        if not entries:
            return prompt

        # Migrate legacy absolute-path keys to repo-relative
        entries = self._migrate_legacy_cache_paths(entries)

        # Check if any entries have LLM-written summaries
        has_summaries = any(e.get("summary") for e in entries.values())

        if has_summaries:
            lines = ["## FILES ANALYZED BY PREVIOUS AGENTS — check these before reading\n"]
            lines.append("| File | Summary | Read By |")
            lines.append("|------|---------|---------|")
            for file_path, entry in entries.items():
                summary = entry.get("summary", "").replace("|", "/")
                read_by = entry.get("read_by", "unknown")
                step = entry.get("workflow_step", "")
                agent_label = f"{read_by} ({step})" if step else read_by
                # Truncate long summaries per-row
                if len(summary) > 120:
                    summary = summary[:117] + "..."
                lines.append(f"| {self._display_path(file_path)} | {summary} | {agent_label} |")
            lines.append("")
            footer = "Do NOT re-read these files unless you need to verify specific line-level details."
            if self.ctx.mcp_enabled:
                footer = "Use get_cached_reads() for full details. " + footer
            lines.append(footer + "\n")
        else:
            # Framework-populated paths only — no summaries
            paths = [self._display_path(p) for p in entries.keys()]
            paths_str = ", ".join(paths[:30])
            if len(paths) > 30:
                paths_str += f", ... ({len(paths) - 30} more)"
            header = "## FILES READ BY PREVIOUS AGENTS"
            if self.ctx.mcp_enabled:
                header += " (call get_cached_reads() for details)"
            lines = [
                header + "\n",
                paths_str,
                "",
            ]

        section = "\n".join(lines)

        # Respect budget
        if len(section) > self._READ_CACHE_MAX_CHARS:
            section = section[:self._READ_CACHE_MAX_CHARS] + "\n[truncated]\n"

        if self.ctx.session_logger:
            self.ctx.session_logger.log(
                "read_cache_injected",
                entry_count=len(entries),
                chars=len(section),
                has_summaries=has_summaries,
            )

        return prompt + "\n\n" + section

    def _inject_self_eval_context(self, prompt: str, task: Task) -> str:
        """Append self-evaluation critique from a previous attempt.

        Self-eval retries store critique in task.context but don't set
        _revised_plan or increment retry_count, so neither inject_replan_context
        nor _inject_retry_context picks it up. This method bridges that gap.

        Skips when _revised_plan exists — that combined case is already
        handled by inject_replan_context which includes the critique.
        """
        critique = task.context.get("_self_eval_critique")
        if not critique:
            return prompt

        # When replan also fired, inject_replan_context already includes critique
        if task.context.get("_revised_plan"):
            return prompt

        attempt = task.context.get("_self_eval_count", 1)

        section = (
            f"\n\n## SELF-EVALUATION FEEDBACK (attempt {attempt})\n\n"
            "Your previous output was reviewed against the acceptance criteria "
            "and found lacking. Address the gaps below:\n\n"
            f"{critique[:4000]}\n"
        )
        return prompt + section

    def _inject_replan_context(self, prompt: str, task: Task) -> str:
        """Append revised plan and attempt history to prompt if available.

        Delegates to ErrorRecoveryManager to avoid duplicating formatting logic.
        """
        if self.ctx.agent and hasattr(self.ctx.agent, '_error_recovery'):
            return self.ctx.agent._error_recovery.inject_replan_context(prompt, task)

        # Fallback for tests or standalone usage without agent reference
        revised_plan = task.context.get("_revised_plan")
        if not revised_plan:
            return prompt
        return prompt + f"\n\n## REVISED APPROACH (retry {task.retry_count})\n\n{revised_plan}"

    def _inject_retry_context(self, prompt: str, task: Task) -> str:
        """Append error + partial progress from the previous attempt on retries.

        Tells the LLM to continue rather than restart from scratch, and
        clarifies that upstream_summary is from a previous *agent*, not a
        previous attempt of the same agent.
        """
        if task.retry_count == 0:
            return prompt

        sections = []
        sections.append(f"## RETRY CONTEXT (attempt {task.retry_count + 1})")
        sections.append("")

        # Truncated error from previous attempt
        if task.last_error:
            from ..safeguards.escalation import EscalationHandler
            truncated = EscalationHandler().truncate_error(task.last_error)
            sections.append("### Previous Error")
            sections.append(truncated)
            sections.append("")

        # Partial progress extracted from the previous response
        prev_summary = task.context.get("_previous_attempt_summary")
        if prev_summary:
            sections.append("### Progress From Previous Attempt")
            sections.append(prev_summary)
            sections.append("")

        # Git diff from the previous attempt's worktree
        prev_git_diff = task.context.get("_previous_attempt_git_diff")
        if prev_git_diff:
            sections.append("### Code Changes From Previous Attempt")
            sections.append(prev_git_diff)
            sections.append("")

        # Committed work discovered on the branch from previous attempts
        branch_work = task.context.get("_previous_attempt_branch_work")
        if branch_work:
            sections.append("### Existing Code on Your Branch")
            sections.append(
                f"{branch_work['commit_count']} commit(s) with "
                f"{branch_work['insertions']} insertions(+) and "
                f"{branch_work['deletions']} deletions(-)."
            )
            sections.append("")
            sections.append(
                "This code is in your working directory RIGHT NOW. "
                "Do NOT rewrite files that already contain your implementation."
            )
            sections.append("")
            if branch_work.get("commit_log"):
                sections.append("Commit log:")
                sections.append(f"```\n{branch_work['commit_log']}\n```")
                sections.append("")
            if branch_work.get("diffstat"):
                sections.append("Diffstat:")
                sections.append(f"```\n{branch_work['diffstat']}\n```")
                sections.append("")
            if branch_work.get("file_list"):
                file_display = branch_work["file_list"][:30]
                sections.append("Files changed: " + ", ".join(file_display))
                if len(branch_work["file_list"]) > 30:
                    sections.append(f"  ... and {len(branch_work['file_list']) - 30} more")
                sections.append("")

        has_progress = bool(prev_summary) or bool(prev_git_diff) or bool(branch_work)

        if has_progress:
            if branch_work:
                sections.append(
                    "Do NOT restart from scratch. Run `git log --oneline` and "
                    "`git diff origin/main..HEAD` to review your existing code, "
                    "then continue from there."
                )
            elif task.last_error and task.last_error.startswith("Interrupted"):
                sections.append(
                    "Do NOT restart from scratch. The previous attempt was interrupted "
                    "before completion. Continue from the progress above."
                )
            else:
                sections.append(
                    "Do NOT restart from scratch. Continue from the progress above, "
                    "fixing the error that caused the previous attempt to fail."
                )
        else:
            sections.append(
                "Do NOT restart from scratch. The previous attempt wrote code but the "
                "progress could not be captured in this prompt. Run `git log --oneline -10` "
                "and `git diff HEAD~1` in your working directory to see what was already done, "
                "then continue from there."
            )

        # Disambiguate upstream context if present
        if task.context.get("upstream_summary"):
            sections.append("")
            sections.append(
                "NOTE: The 'UPSTREAM AGENT FINDINGS' section above is from a "
                "previous agent in the workflow chain, not from your previous attempt."
            )

        return prompt + "\n\n" + "\n".join(sections)

    def _inject_preview_mode(self, prompt: str, _task: Task) -> str:
        """Inject preview mode constraints when task is a preview."""
        preview_section = """
## PREVIEW MODE — READ-ONLY EXECUTION
You are in PREVIEW MODE. You must plan your implementation WITHOUT writing any files.

CONSTRAINTS:
- Do NOT use Write, Edit, or NotebookEdit tools
- Do NOT use Bash to create, modify, or delete files
- DO use Read, Glob, Grep, and Bash (read-only commands like git log, git diff, ls) to explore
- DO read every file you plan to modify to understand current state

REQUIRED OUTPUT — Produce a structured execution preview:

### Files to Modify
For each file, list:
- File path
- What changes will be made (specific, not vague)
- Estimated lines added/removed

### New Files to Create
For each new file:
- File path
- Purpose
- Key contents/structure
- Estimated line count

### Implementation Approach
- Step-by-step plan with ordering
- Which patterns from existing code will be followed
- Any dependencies between changes

### Risks and Edge Cases
- What could go wrong
- Edge cases to handle
- Backward compatibility concerns

### Estimated Total Change Size
- Total lines added/removed
- Number of files affected

This preview will be reviewed by the architect before implementation is authorized.
"""
        return preview_section + "\n\n" + prompt

    def _inject_human_guidance(self, prompt: str, task: Task) -> str:
        """Inject human guidance from escalation report if available."""
        # Check for human guidance in escalation report
        if task.escalation_report and task.escalation_report.human_guidance:
            guidance = task.escalation_report.human_guidance
            guidance_section = f"""

## CRITICAL: Human Guidance Provided

A human expert has reviewed this task and provided the following guidance to help you succeed:

{guidance}

Please carefully consider this guidance when approaching the task. This information may help you avoid the previous failures.

## Previous Failure Context

{task.escalation_report.root_cause_hypothesis}

Suggested interventions:
"""
            for i, intervention in enumerate(task.escalation_report.suggested_interventions, 1):
                guidance_section += f"{i}. {intervention}\n"

            return prompt + guidance_section

        # Fall back to context-based guidance (legacy support)
        context_guidance = task.context.get("human_guidance")
        if context_guidance:
            return prompt + f"""

## CRITICAL: Human Guidance Provided

A human expert has provided guidance for this task:

{context_guidance}

Please carefully consider this guidance when approaching the task.
"""

        return prompt

    def _append_test_failure_context(self, prompt: str, task: Task) -> str:
        """Append test failure report to prompt if present."""
        test_failure_report = task.context.get("_test_failure_report")
        if not test_failure_report:
            return prompt

        return prompt + f"""

## IMPORTANT: Previous Tests Failed

Your previous implementation had test failures. Please fix the issues below:

{test_failure_report}

Fix the failing tests and ensure all tests pass.
"""

    def _detect_engineer_specialization(self, task: Task) -> tuple[Optional["SpecializationProfile"], list[str]]:
        """Detect engineer specialization profile for this task.

        Returns (profile, files) so the caller can reuse the file list for
        model routing without a second detect_file_patterns() call.
        Returns (None, []) when specialization is disabled or no match found.
        """
        if self.ctx.config.base_id != "engineer":
            return None, []

        # Per-agent toggle from agents.yaml
        if self.ctx.agent_definition and not self.ctx.agent_definition.specialization_enabled:
            self.logger.debug("Specialization disabled for this agent via agents.yaml")
            return None, []

        # Global toggle from specializations.yaml
        from .engineer_specialization import (
            detect_specialization,
            get_specialization_enabled,
            get_auto_profile_config,
            detect_file_patterns,
            _load_profiles,
        )

        if not get_specialization_enabled():
            self.logger.debug("Specialization disabled globally via specializations.yaml")
            return None, []

        # Extract files once — reused for both static detection and auto-profile fallback
        files = detect_file_patterns(task)

        profile = detect_specialization(task, files=files)
        if profile:
            return profile, files

        # Auto-profile fallback: check registry, then generate
        auto_config = get_auto_profile_config()
        if auto_config is None or not auto_config.enabled:
            return None, []

        if not files:
            return None, []

        import time
        from .profile_registry import ProfileRegistry, GeneratedProfileEntry
        from .profile_generator import ProfileGenerator

        registry = ProfileRegistry(
            self.ctx.workspace,
            max_profiles=auto_config.max_cached_profiles,
            staleness_days=auto_config.staleness_days,
        )

        # Try cache first
        cached = registry.find_matching_profile(
            files,
            f"{task.title} {task.description}",
            min_score=auto_config.min_match_score,
        )
        if cached:
            self.logger.info("Matched cached generated profile '%s'", cached.id)
            return cached, files

        # Generate new profile
        generator = ProfileGenerator(self.ctx.workspace, model=auto_config.model)
        existing_ids = [p.id for p in _load_profiles()]
        generated = generator.generate_profile(task, files, existing_ids)
        if generated:
            now = time.time()
            registry.store_profile(GeneratedProfileEntry(
                profile=generated.profile,
                created_at=now,
                last_matched_at=now,
                match_count=1,
                source_task_id=task.id,
                tags=generated.tags,
                file_extensions=generated.file_extensions,
            ))
            self.logger.info("Generated new profile '%s'", generated.profile.id)
            return generated.profile, files

        return None, []
