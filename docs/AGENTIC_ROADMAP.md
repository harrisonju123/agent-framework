# Agentic Flow Features Roadmap

## Context

The agent framework has a solid foundation (queues, routing, safeguards, multi-repo, teams) but the agents themselves are relatively "dumb" — they retry the same failing approach, have no memory between tasks, follow a rigid linear workflow, and can't adapt to task complexity. This roadmap focuses on making agents **more autonomous, smarter, and better at their jobs**.

## Current State

- **Linear workflow only**: architect → engineer → qa → architect (PR) → done
- **No memory**: each task starts from scratch, even for the same repo
- **No self-evaluation**: quality gate is only QA, after full engineer cycle
- **Retry = same approach**: failures retry with same prompt + appended error
- **All engineers identical**: no specialization by language/domain
- **No dynamic decomposition**: one task = one engineer, regardless of size

---

## Sprint 1: Adaptive Intelligence (P0 — Highest Impact) ✅

These three features together likely improve task success rate by 30-50%.

### 1. Agent Memory System (M) ✅
**Why:** An engineer working on the same repo for the 10th time still rediscovers project structure, test commands, and conventions every time.

**What:** Persistent memory store (per repo, per agent type) that agents read at task start and write at task end. Stores patterns like "repo X uses factory pattern", "test suite requires Docker", "PR reviews always flag missing type annotations".

**Changes:**
- NEW: `src/agent_framework/memory/memory_store.py` — read/write JSON per repo/agent
- NEW: `src/agent_framework/memory/memory_retriever.py` — relevance scoring, recency decay
- `src/agent_framework/core/agent.py` — load memories in `_build_prompt()`, extract after success
- `mcp-servers/task-queue/src/index.ts` — add `remember`/`recall` MCP tools
- `config/agent-framework.yaml` — memory configuration section

### 2. Reflection & Self-Evaluation Loop (M) ✅
**Why:** The only quality gate is QA review after the full engineer cycle. Obvious issues propagate through the entire pipeline before being caught.

**What:** After LLM completes but before marking task done, agent reviews its own output against acceptance criteria using a cheap model (haiku). If gaps found, retries with critique as context — without consuming a queue-level retry.

**Changes:**
- `src/agent_framework/core/agent.py` — new `_self_evaluate()` method, called from `_handle_successful_response()` before workflow chain
- `config/agent-framework.yaml` — `max_self_eval_retries: 2`, model override for eval

### 3. Dynamic Replanning on Failure (M) ✅
**Why:** Currently `_handle_failure()` just bumps retry_count and resets to pending with the same prompt. 5 retries of the same approach = exponentially decreasing success.

**What:** On retry 2+, generate a revised plan (using haiku) that accounts for what failed. Store replan history so the agent sees what was already tried.

**Changes:**
- `src/agent_framework/core/agent.py` — new `_request_replan()`, modify `_handle_failure()`
- `src/agent_framework/core/task.py` — add `replan_history` field

---

## Sprint 2: Workflow Flexibility (P1)

### 4. Structured QA-Engineer Feedback (S)
**Why:** QA findings are parsed via regex and forwarded as truncated text. Engineers misinterpret or miss issues.

**What:** QA produces structured JSON findings (`{file, line, severity, description, suggested_fix}`). Engineer gets a numbered checklist to address.

**Changes:**
- `src/agent_framework/core/agent.py` — new `_parse_structured_findings()`, modify `_build_review_fix_task()`
- `config/agents.yaml` — update QA prompt for structured output

### 5. DAG-Based Workflow Engine (L)
**Why:** `_enforce_workflow_chain()` is purely linear. Real tasks need: parallel sub-tasks, conditional branches, skip-paths for simple changes.

**What:** Replace linear chain with DAG. Workflow steps have conditions on edges. Backward compatible — current linear chain becomes a simple DAG.

```yaml
workflows:
  default:
    steps:
      plan: {agent: architect, next: [implement]}
      implement: {agent: engineer, next: [qa_review]}
      qa_review:
        agent: qa
        next:
          - {condition: approved, target: create_pr}
          - {condition: needs_fix, target: implement}
      create_pr: {agent: architect}
```

**Changes:**
- NEW: `src/agent_framework/workflow/dag.py` — DAG definition and traversal
- NEW: `src/agent_framework/workflow/executor.py` — execution state tracking
- NEW: `src/agent_framework/workflow/conditions.py` — transition condition evaluators
- `src/agent_framework/core/agent.py` — replace `_enforce_workflow_chain()` with DAG executor
- `src/agent_framework/core/config.py` — new workflow model
- `config/agent-framework.yaml` — new workflow syntax

### 6. Conditional Workflow Branches (M)
**Why:** Not all tasks need full Architect-Engineer-QA. Docs-only changes can skip QA. Small fixes can skip architect review.

**What:** Condition evaluators on workflow edges: `files_match(pattern)`, `pr_size(threshold)`, `test_result(passed/failed)`.

**Changes:**
- `src/agent_framework/workflow/conditions.py` — evaluators
- `src/agent_framework/workflow/dag.py` — conditional edge resolution
- `config/agent-framework.yaml` — conditional workflow definitions

### 7. Intelligent Model Routing (M)
**Why:** Current `ModelSelector` is purely type-based. A trivial one-liner gets the same model as a complex refactor.

**What:** Route models based on task characteristics: code complexity, repo language, file count, historical success rate, remaining budget.

**Changes:**
- NEW: `src/agent_framework/llm/intelligent_router.py` — smart routing
- `src/agent_framework/llm/claude_cli_backend.py` — use new router

---

## Sprint 3: Agent Depth (P2)

### 8. Interactive Checkpoints (S)
**Why:** System is either fully autonomous or escalated. No middle ground for high-stakes changes.

**What:** Configurable pause points in workflows. `agent approve <task-id>` to resume.

**Changes:**
- `src/agent_framework/core/agent.py` — checkpoint checking in `_handle_successful_response()`
- `src/agent_framework/cli/main.py` — `approve` command
- `config/agent-framework.yaml` — checkpoint config per workflow

### 9. Guided Escalation (S)
**Why:** Escalations are opaque. Humans dig through logs to understand what happened.

**What:** Structured escalation reports with attempt history, root cause hypothesis, suggested interventions. `agent guide <id> --hint "..."` injects human guidance into retry.

**Changes:**
- `src/agent_framework/safeguards/escalation.py` — structured reports
- `src/agent_framework/cli/main.py` — `guide` command

### 10. Context Window Management (M)
**Why:** Long tasks can exceed context windows, causing quality decay.

**What:** Context budget tracking, progressive summarization of tool outputs, priority-based context inclusion.

**Changes:**
- NEW: `src/agent_framework/context/manager.py` — budget management
- NEW: `src/agent_framework/context/summarizer.py` — progressive summarization
- `src/agent_framework/core/agent.py` — integrate into prompt building

### 11. Engineer Specialization (M)
**Why:** All engineers get the same generic prompt regardless of language/domain.

**What:** Specialization profiles (backend, frontend, infra) selected based on task file patterns. Different prompts, teammates, tool guidance.

**Changes:**
- `config/agents.yaml` — engineer profiles section
- `src/agent_framework/core/agent.py` — specialize prompt based on profile
- `src/agent_framework/core/team_composer.py` — specialize teammates

### 12. Tool Use Optimization (S)
**Why:** Agents waste tokens on inefficient tool patterns (sequential reads vs grep, etc.).

**What:** Capture tool use patterns per task, store efficient patterns in memory, inject tool tips in prompts.

**Changes:**
- `src/agent_framework/core/agent.py` — capture patterns, inject guidance
- `src/agent_framework/memory/memory_store.py` — tool pattern storage

---

## Sprint 4: Scale (P3)

### 13. Dynamic Sub-Task Decomposition (L)
**Why:** A 500-line change gets one engineer one shot. Should be 3 smaller parallel tasks.

**What:** Architect decomposes plans into independent sub-tasks. Fan-out to multiple engineers, fan-in when all complete.

**Changes:**
- NEW: `src/agent_framework/workflow/decomposer.py` — task decomposition
- `src/agent_framework/core/task.py` — `parent_task_id`, `subtask_ids` fields
- `src/agent_framework/core/agent.py` — architect invokes decomposer

### 14. Execution Previews (M)
**Why:** Most expensive failures are when engineer implements a wrong approach (30+ min wasted).

**What:** New `TaskType.PREVIEW` where engineer plans changes without writing files. Architect reviews preview before authorizing full implementation.

**Changes:**
- `src/agent_framework/core/task.py` — new task type
- `src/agent_framework/core/agent.py` — preview mode in prompt
- `config/agents.yaml` — engineer preview prompt

### 15. Progressive Autonomy Levels (M)
**Why:** Different repos warrant different trust levels.

**What:** Per-repo autonomy: Level 1 (plan only), Level 2 (plan+implement, human reviews), Level 3 (full autonomy). Auto-adjusts based on success rate.

**Changes:**
- `src/agent_framework/core/config.py` — autonomy in `RepositoryConfig`
- `src/agent_framework/core/agent.py` — map level to checkpoints

---

## Sprint 5: Advanced Patterns (P4)

### 16. Cross-Repository Coordination (L)
Multi-repo workflows: architect creates cross-repo plan, engineers work each repo in parallel, system validates compatibility before PRs.

### 17. Agent Debate / Multi-Perspective Review (L)
For complex decisions, spawn agents with opposing perspectives. Advocate vs critic, arbiter synthesizes.

### 18. Autonomous Agent Spawning (L)
Architect dynamically creates specialized agent profiles at runtime for novel task types.

---

## Architecture Principles

1. **`task.context` is the state bus** — all feature state flows through context dict, preserving stateless architecture
2. **Off by default** — every feature enabled via config, gracefully degrades when disabled
3. **Backward compatible** — current linear workflow is a simple DAG, current prompts work without memory
4. **Use cheap models for meta-work** — reflection, replanning, memory retrieval use haiku to keep costs low
5. **Reference, don't embed** — large state (memories, previews) stored as files, referenced by path in context

## Key Integration Point

Almost all features modify one of these methods in `agent.py`:
- `_build_prompt()` — inject memories, tool tips, context summaries
- `_handle_successful_response()` — self-evaluation, memory extraction, checkpoint checks
- `_handle_failure()` — dynamic replanning
- `_enforce_workflow_chain()` → replaced by DAG executor

## Verification

For each sprint:
1. Run existing test suite: `pytest tests/ -v` (zero regressions)
2. Manual E2E: `agent work` with a real task, verify new behavior
3. A/B comparison: run same task with/without feature, compare success rate and token usage
