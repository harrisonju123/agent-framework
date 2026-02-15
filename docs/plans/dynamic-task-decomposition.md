# Dynamic Sub-Task Decomposition

**Plan ID:** planning-AF-1771135451
**Status:** Proposed
**Author:** architect
**Created:** 2026-02-15

## Problem Statement

Today, a 500-line change gets assigned to one engineer for a single attempt. If it fails, the whole thing retries. This is wasteful - large tasks should be decomposed into smaller, parallel subtasks (~150-250 lines each) that multiple engineers can work on concurrently.

### Current State
- Task model has `depends_on: list[str]` for sequential dependencies but **no parent-child hierarchy**
- Architect manually creates subtasks in prompt text, but there's **no structured decomposition logic**
- All subtasks are sequential (chained via depends_on) - **no parallel execution**
- No fan-in mechanism: when subtasks complete, nothing aggregates results back to the parent
- No `parent_task_id` or `subtask_ids` fields on the Task model

### Desired State
- Architect invokes a **task decomposer** that splits large plans into independent subtasks
- Subtasks have `parent_task_id` linking back; parent has `subtask_ids` listing children
- Independent subtasks execute in **parallel** (no depends_on between them)
- **Fan-in**: when all subtasks complete, parent task automatically advances to next workflow step
- Size guardrails: tasks estimated >500 lines trigger decomposition; subtasks target <300 lines

## Architecture

### New Fields on Task Model (`task.py`)

```python
# Parent-child hierarchy
parent_task_id: Optional[str] = None      # Points to decomposed parent
subtask_ids: list[str] = Field(default_factory=list)  # Children created by decomposition
decomposition_strategy: Optional[str] = None  # "by_feature", "by_layer", "by_refactor_feature"
```

### New Module: `src/agent_framework/core/task_decomposer.py`

Responsible for:
1. Analyzing a PlanDocument and estimating line counts per file/area
2. Identifying natural split boundaries (by feature, by layer, by refactor+feature)
3. Creating independent subtask Task objects with proper `parent_task_id`
4. Updating parent task's `subtask_ids`
5. Determining which subtasks can run in parallel vs. which need ordering

### Modified: Queue Fan-In Logic (`file_queue.py`)

New method `check_subtasks_complete(parent_task_id)`:
- Scans completed directory for all tasks with matching `parent_task_id`
- Returns True only when ALL subtask_ids are in completed with COMPLETED status
- On completion: creates a fan-in continuation task for the parent's next workflow step

### Modified: Agent Processing (`agent.py`)

- After architect creates a plan, check estimated size
- If >500 lines, invoke `TaskDecomposer` instead of queueing single implementation task
- Decomposer creates N subtasks queued to engineer (in parallel by default)
- Add fan-in check in the agent polling loop: when a subtask completes, check if all siblings are done

## Implementation Subtasks

This implementation is broken into **3 independent subtasks** that can be worked on in parallel, plus 1 sequential integration task.

---

### Subtask 1: Task Model + Decomposer Core (~200 lines)

**Files to modify:**
- `src/agent_framework/core/task.py` - Add parent_task_id, subtask_ids, decomposition_strategy fields
- `src/agent_framework/core/task_decomposer.py` (NEW) - Core decomposition logic

**Changes:**
1. Add 3 new fields to Task model:
   - `parent_task_id: Optional[str] = None`
   - `subtask_ids: list[str] = Field(default_factory=list)`
   - `decomposition_strategy: Optional[str] = None`

2. Create `TaskDecomposer` class with:
   - `should_decompose(plan: PlanDocument, estimated_lines: int) -> bool` - Returns True if >500 lines
   - `decompose(parent_task: Task, plan: PlanDocument) -> list[Task]` - Creates subtask list
   - `_estimate_subtask_size(files: list[str]) -> int` - Rough line estimate per file group
   - `_identify_split_boundaries(plan: PlanDocument) -> list[SubtaskBoundary]` - Find natural split points
   - `_create_subtask(parent: Task, boundary: SubtaskBoundary, index: int) -> Task` - Build child task

3. `SubtaskBoundary` dataclass:
   - `name: str` - Human-readable name (e.g. "Database schema changes")
   - `files: list[str]` - Files this subtask covers
   - `approach_steps: list[str]` - Subset of plan.approach relevant to this subtask
   - `depends_on_subtasks: list[int]` - Indices of subtasks this one depends on (for ordering)
   - `estimated_lines: int`

**Acceptance Criteria:**
- Task model serializes/deserializes with new fields (backward compatible - all Optional/default)
- TaskDecomposer.decompose() returns 2-5 subtasks from a sample plan
- Each subtask has correct parent_task_id
- Parent task has all subtask_ids populated
- Subtasks without inter-dependencies have empty depends_on

**Tests:** `tests/unit/test_task_decomposer.py` (~100 lines)

---

### Subtask 2: Queue Fan-In + Dependency Changes (~180 lines)

**Files to modify:**
- `src/agent_framework/queue/file_queue.py` - Fan-in detection, parent-child queries
- `src/agent_framework/core/task_builder.py` - Decomposition-aware task building helpers

**Changes:**
1. `FileQueue` new methods:
   - `check_subtasks_complete(parent_task_id: str) -> bool` - Check if all subtasks of a parent are completed
   - `get_subtasks(parent_task_id: str) -> list[Task]` - Get all subtasks across queues/completed
   - `get_parent_task(task: Task) -> Optional[Task]` - Load parent task from any location
   - `create_fan_in_task(parent_task: Task, subtasks: list[Task]) -> Task` - Create aggregation task

2. `create_fan_in_task` builds a continuation task that:
   - Has the parent's original context (github_repo, workflow, etc.)
   - Aggregates result_summary from all subtasks
   - Routes to the next workflow step (e.g., QA after all engineer subtasks done)
   - Type = parent's original type (carries forward)

3. `task_builder.py` additions:
   - `build_decomposed_subtask(parent_task, subtask_boundary, index) -> Task` helper
   - Ensures subtask IDs follow pattern: `{parent_id}-sub-{index}`

**Acceptance Criteria:**
- `check_subtasks_complete` returns False when any subtask is still pending/in_progress
- `check_subtasks_complete` returns True only when ALL subtask_ids are COMPLETED
- Fan-in task contains aggregated context from all subtasks
- Subtask IDs are deterministic and filesystem-safe

**Tests:** `tests/unit/test_fan_in.py` (~120 lines)

---

### Subtask 3: Agent Integration + Workflow Updates (~150 lines)

**Files to modify:**
- `src/agent_framework/core/agent.py` - Invoke decomposer from architect, fan-in check in poll loop
- `src/agent_framework/workflow/executor.py` - Handle fan-in routing
- `config/docs/change_metrics.md` - Document decomposition thresholds

**Changes:**
1. In `agent.py` architect task processing:
   - After plan creation, call `TaskDecomposer.should_decompose(plan, estimated_lines)`
   - If True: call `decomposer.decompose()`, push subtasks to engineer queue, update parent
   - If False: existing behavior (single engineer task)

2. In `agent.py` poll loop (post-task-completion):
   - After marking a subtask complete, call `queue.check_subtasks_complete(parent_task_id)`
   - If all done: call `queue.create_fan_in_task()` and route to next workflow step
   - If not: no-op, continue polling

3. In `executor.py`:
   - Recognize fan-in tasks (context has `fan_in: True`)
   - Route fan-in tasks through normal workflow DAG (they behave like completion of the parent)

4. Update `change_metrics.md`:
   - Document automatic decomposition threshold (>500 lines)
   - Document subtask size target (<300 lines)

**Acceptance Criteria:**
- Architect auto-decomposes tasks >500 estimated lines into parallel subtasks
- Subtask completion triggers fan-in check
- Fan-in task routes correctly through workflow DAG
- Tasks <500 lines proceed normally (no decomposition)

**Tests:** `tests/unit/test_task_decomposition_integration.py` (~80 lines)

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Backward compatibility - old tasks without new fields | All new fields are Optional with defaults; existing JSON deserializes fine |
| Race condition on fan-in check | Fan-in check is idempotent; `create_fan_in_task` checks if already created before writing |
| Subtask failure cascading | If any subtask fails, don't create fan-in. Escalation handler deals with retries per subtask |
| Over-decomposition (too many tiny subtasks) | Minimum subtask size of 50 lines; max 5 subtasks per decomposition |
| Circular parent-child references | Subtasks cannot themselves be decomposed (depth limit = 1) |

## Success Criteria

1. A task with 600 estimated lines automatically decomposes into 2-3 subtasks
2. Independent subtasks execute in parallel (no depends_on between them)
3. When all subtasks complete, fan-in creates continuation task for QA
4. Existing tasks without subtasks work identically to before (no regression)
5. All new code has unit test coverage

## Estimated Effort

- **Subtask 1 (Model + Decomposer):** ~200 lines code + ~100 lines test = 300 total
- **Subtask 2 (Queue Fan-In):** ~180 lines code + ~120 lines test = 300 total
- **Subtask 3 (Agent Integration):** ~150 lines code + ~80 lines test = 230 total
- **Total:** ~830 lines across 3 parallel subtasks

Each subtask is independently implementable and testable. Subtasks 1 and 2 have zero dependencies on each other. Subtask 3 depends on both 1 and 2 being complete.
