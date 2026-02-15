# Conditional Workflow Branches

## Overview

The agent framework supports **conditional workflow branches** using a DAG (Directed Acyclic Graph) based workflow engine. This allows workflows to dynamically route tasks based on runtime conditions like:

- **File patterns**: Skip QA for docs-only changes
- **PR size**: Fast-track small changes, full review for large PRs
- **Test results**: Route based on test pass/fail status
- **Agent signals**: Allow agents to explicitly route to specific next steps

This eliminates rigid linear workflows (architect → engineer → qa → architect) and enables intelligent routing that adapts to task characteristics.

---

## Quick Start

### 1. Enable Conditional Workflows in Config

Edit `config/agent-framework.yaml`:

```yaml
workflows:
  smart_workflow:
    description: "Skip QA for docs-only changes and small fixes"
    start_step: plan
    pr_creator: architect
    auto_review: true
    require_tests: true
    steps:
      plan:
        agent: architect
        next:
          - target: implement

      implement:
        agent: engineer
        next:
          # Priority 10: Skip QA if only markdown files changed
          - target: create_pr
            condition: files_match
            params:
              pattern: "*.md"
            priority: 10

          # Priority 5: Skip QA for small changes (< 3 files)
          - target: create_pr
            condition: pr_size_under
            params:
              max_files: 3
            priority: 5

          # Priority 1: Default path - go to QA
          - target: qa_review
            condition: always
            priority: 1

      qa_review:
        agent: qa
        next:
          # Priority 10: Approved → create PR
          - target: create_pr
            condition: approved
            priority: 10

          # Priority 5: Needs fix → back to engineer
          - target: implement
            condition: needs_fix
            priority: 5

      create_pr:
        agent: architect
        # Terminal step (no next)
```

### 2. Use the Workflow

Tasks automatically use the configured workflow. The framework evaluates conditions in priority order (highest first) and routes to the first matching edge.

---

## Condition Types

### File Pattern Matching

**Type:** `files_match`

**Parameters:**
- `pattern` (required): Glob pattern to match changed files

**Example:** Skip QA for documentation changes

```yaml
- target: create_pr
  condition: files_match
  params:
    pattern: "*.md"
  priority: 10
```

**Use cases:**
- Docs-only changes: `*.md`, `docs/**/*.rst`
- Config-only changes: `*.yaml`, `*.json`, `*.toml`
- Tests-only changes: `tests/**/*.py`

---

### PR Size Threshold

**Type:** `pr_size_under`

**Parameters:**
- `max_files` (required): Maximum number of changed files (exclusive)

**Example:** Fast-track small PRs

```yaml
- target: create_pr
  condition: pr_size_under
  params:
    max_files: 3
  priority: 5
```

**Use cases:**
- Skip QA for trivial fixes (1-2 files)
- Lightweight review for small changes (< 5 files)
- Full review for large refactors (≥ 10 files)

---

### Test Results

**Type:** `test_passed` / `test_failed`

**Parameters:** None

**Example:** Route based on test outcome

```yaml
- target: merge
  condition: test_passed
  priority: 10

- target: debug
  condition: test_failed
  priority: 5
```

**Note:** Test results must be set in task context by the testing agent:

```python
task.context["test_result"] = "passed"  # or "failed"
```

---

### QA Approval

**Type:** `approved` / `needs_fix`

**Parameters:** None

**Example:** QA feedback loop

```yaml
qa_review:
  agent: qa
  next:
    - target: create_pr
      condition: approved
      priority: 10

    - target: implement
      condition: needs_fix
      priority: 5
```

**Detection:** The framework automatically detects approval/rejection from QA response content:

- **Approval keywords:** `approved`, `LGTM`, `looks good`, `ready for merge`
- **Rejection keywords:** `needs fix`, `failed`, `issues found`, `problems detected`

---

### PR Creation Detection

**Type:** `pr_created` / `no_pr`

**Parameters:** None

**Example:** Route based on whether a PR was created

```yaml
- target: merge_queue
  condition: pr_created
  priority: 10

- target: qa_review
  condition: no_pr
  priority: 5
```

**Detection:** Checks for `pr_url` in task context or PR URL patterns in response.

---

### Routing Signals

**Type:** `signal_target`

**Parameters:**
- `target` (required): Agent name to match against routing signal

**Example:** Allow architect to override default routing

```yaml
- target: qa
  condition: signal_target
  params:
    target: qa
  priority: 10

- target: create_pr
  condition: always
  priority: 1
```

**Agent usage:** Agents can emit routing signals to control flow:

```python
# In agent response
routing_signal = {
    "target_agent": "qa",
    "reason": "Complex changes require thorough review"
}
```

---

### Unconditional (Always)

**Type:** `always`

**Parameters:** None

**Example:** Default fallback path

```yaml
- target: qa_review
  condition: always
  priority: 1
```

---

## Priority and Edge Evaluation

Edges are evaluated in **descending priority order** (highest first):

```yaml
next:
  - target: fast_path
    condition: files_match
    params: {pattern: "*.md"}
    priority: 10      # Evaluated first

  - target: medium_path
    condition: pr_size_under
    params: {max_files: 3}
    priority: 5       # Evaluated second

  - target: default_path
    condition: always
    priority: 1       # Evaluated last (fallback)
```

**Best practices:**
1. **Highest priority (10+)**: Most specific conditions (docs-only, config-only)
2. **Medium priority (5)**: Size-based or test-based routing
3. **Low priority (1)**: Default fallback (`always`)

---

## How Changed Files Are Detected

The framework automatically populates `changed_files` in the workflow context:

1. **Explicit context**: If task already has `changed_files` in context, use that
2. **Git diff**: Otherwise, run `git diff --name-only HEAD` to capture staged and unstaged changes
3. **Empty list**: If git fails or no changes, conditions using `changed_files` return `False`

**Note:** Changed files are relative paths from the repository root.

---

## Example Workflows

### 1. Skip QA for Documentation

```yaml
smart_docs_workflow:
  description: "Fast-track documentation changes"
  start_step: implement
  steps:
    implement:
      agent: engineer
      next:
        - target: create_pr
          condition: files_match
          params:
            pattern: "*.md"
          priority: 10

        - target: qa_review
          condition: always
          priority: 1

    qa_review:
      agent: qa
      next:
        - target: create_pr
          condition: approved

    create_pr:
      agent: architect
```

**Result:** Documentation changes skip QA and go directly to PR creation.

---

### 2. Size-Based Review Intensity

```yaml
tiered_review:
  description: "Review intensity scales with change size"
  start_step: implement
  steps:
    implement:
      agent: engineer
      next:
        # Trivial changes: skip QA
        - target: create_pr
          condition: pr_size_under
          params:
            max_files: 2
          priority: 10

        # Small changes: lightweight QA
        - target: quick_qa
          condition: pr_size_under
          params:
            max_files: 5
          priority: 5

        # Large changes: full QA
        - target: full_qa
          condition: always
          priority: 1

    quick_qa:
      agent: qa
      task_type: VERIFICATION  # Faster task type
      next:
        - target: create_pr
          condition: approved

    full_qa:
      agent: qa
      task_type: QA_VERIFICATION
      next:
        - target: create_pr
          condition: approved
        - target: implement
          condition: needs_fix

    create_pr:
      agent: architect
```

---

### 3. Combined Conditions (Multi-Path)

```yaml
intelligent_workflow:
  description: "Multiple skip conditions with fallback"
  start_step: plan
  steps:
    plan:
      agent: architect
      next:
        - target: implement

    implement:
      agent: engineer
      next:
        # Path 1: Docs only
        - target: create_pr
          condition: files_match
          params:
            pattern: "*.md"
          priority: 10

        # Path 2: Config only
        - target: create_pr
          condition: files_match
          params:
            pattern: "*.{yaml,json,toml}"
          priority: 9

        # Path 3: Small code changes
        - target: create_pr
          condition: pr_size_under
          params:
            max_files: 3
          priority: 5

        # Path 4: Default full review
        - target: qa_review
          condition: always
          priority: 1

    qa_review:
      agent: qa
      next:
        - target: create_pr
          condition: approved
          priority: 10
        - target: implement
          condition: needs_fix
          priority: 5

    create_pr:
      agent: architect
```

---

## QA Feedback Loop

The most common use case is the QA → Engineer feedback loop:

```yaml
qa_review:
  agent: qa
  next:
    # If QA approves, create PR
    - target: create_pr
      condition: approved
      priority: 10

    # If QA finds issues, send back to engineer
    - target: implement
      condition: needs_fix
      priority: 5
```

**How it works:**
1. QA agent reviews code and emits response with approval/rejection keywords
2. Framework evaluates `approved` condition based on response content
3. If approved: routes to `create_pr` (architect)
4. If rejected: routes back to `implement` (engineer) with structured findings

**Cycle limit:** The framework caps review cycles at 3 to prevent infinite loops.

---

## Advanced: Routing Signals

Agents can explicitly control workflow routing by emitting signals:

### Agent Code Example

```python
# In QA agent: force routing to architect for complex review
routing_signal = RoutingSignal(
    target_agent="architect",
    reason="Complex architectural changes require architect review",
    timestamp=datetime.utcnow().isoformat(),
    source_agent="qa"
)
```

### Workflow Configuration

```yaml
qa_review:
  agent: qa
  next:
    # Honor explicit routing signals
    - target: architect
      condition: signal_target
      params:
        target: architect
      priority: 10

    # Default: approved → PR
    - target: create_pr
      condition: approved
      priority: 5
```

**Use cases:**
- Architect review for complex changes
- Security review for sensitive code
- Skip steps based on agent judgment

---

## Backward Compatibility

The framework supports **legacy linear workflows** for backward compatibility:

### Legacy Format (Still Supported)

```yaml
workflows:
  default:
    description: "Linear workflow"
    agents: [architect, engineer, qa]
    pr_creator: architect
```

This is automatically converted to a DAG with unconditional edges:
- architect → engineer (always)
- engineer → qa (always)
- qa → architect (always)

### Migration Path

To migrate from linear to conditional workflows:

1. Start with linear format
2. Add `steps` and `start_step` to define DAG structure
3. Add conditional edges where needed
4. Remove `agents` list (mutually exclusive with `steps`)

---

## Testing Conditional Workflows

### Unit Tests

Test individual condition evaluators:

```python
from agent_framework.workflow.conditions import FilesMatchCondition, EdgeCondition, EdgeConditionType

def test_docs_only_detection():
    condition = EdgeCondition(
        EdgeConditionType.FILES_MATCH,
        params={"pattern": "*.md"}
    )

    evaluator = FilesMatchCondition()
    result = evaluator.evaluate(
        condition, task, response,
        context={"changed_files": ["README.md", "docs/GUIDE.md"]}
    )

    assert result is True
```

### Integration Tests

Test full workflow execution:

```python
from agent_framework.workflow.executor import WorkflowExecutor

def test_docs_skip_qa(workflow_executor):
    workflow = build_smart_workflow()
    task = make_task(changed_files=["README.md"])
    response = make_response()

    routed = workflow_executor.execute_step(
        workflow=workflow,
        task=task,
        response=response,
        current_agent_id="engineer",
        context={"changed_files": ["README.md"]}
    )

    # Verify task routed to architect, not QA
    assert routed is True
    assert_task_queued("architect")
    assert_task_not_queued("qa")
```

---

## Troubleshooting

### Condition Never Matches

**Symptom:** Tasks always take the default path, even when you expect a condition to match.

**Debugging:**
1. Check that `changed_files` is populated in task context:
   ```python
   print(task.context.get("changed_files"))
   ```
2. Verify pattern syntax (glob patterns, not regex):
   - ✅ `*.md` (glob)
   - ❌ `.*\.md$` (regex)
3. Check priority order - higher priority edges are evaluated first
4. Enable debug logging: `PYTHONPATH=. python -m agent_framework.run_agent --agent engineer --debug`

### Infinite Loops

**Symptom:** Tasks cycle between agents indefinitely.

**Prevention:**
- The framework automatically detects **unconditional cycles** and rejects them:
  ```yaml
  # This will raise ValueError: unconditional cycle
  step_a:
    next:
      - target: step_b
        condition: always
  step_b:
    next:
      - target: step_a
        condition: always
  ```
- **Conditional cycles** are allowed (e.g., QA → Engineer on failure)
- The framework caps review cycles at `MAX_REVIEW_CYCLES = 3`

### Git Diff Fails

**Symptom:** `changed_files` is empty even though files were modified.

**Causes:**
1. Not in a git repository
2. No commits on current branch
3. Git command timeout (> 10 seconds)

**Workaround:** Explicitly set `changed_files` in task context:
```python
task.context["changed_files"] = ["src/main.py", "tests/test_main.py"]
```

---

## Performance Considerations

### Git Diff Overhead

- **Cost:** ~10-50ms per task
- **Cached:** Results stored in `task.context["changed_files"]` after first call
- **Timeout:** 10 seconds (tasks with massive repos may timeout)

### Condition Evaluation

- **Cost:** Negligible (~1ms per condition)
- **Optimization:** Conditions evaluated in priority order, stops at first match

### Edge Sorting

- Edges are sorted by priority once during workflow initialization
- No runtime sorting overhead

---

## Limitations

1. **Changed files detection**: Requires git repository and committed changes
2. **Pattern matching**: Glob patterns only, not regex
3. **Cycle prevention**: Only detects unconditional cycles, conditional cycles allowed but capped at 3 iterations
4. **Context propagation**: Task context is mutable and shared across workflow steps

---

## Future Enhancements

### Planned Features (Roadmap)

1. **Advanced conditions**:
   - `code_complexity(threshold)`: Route based on cyclomatic complexity
   - `language_match(language)`: Route to specialized agents
   - `repo_match(pattern)`: Different workflows per repository

2. **Parallel execution**:
   - Fan-out to multiple agents (e.g., parallel QA and security review)
   - Fan-in when all parallel tasks complete

3. **Dynamic decomposition**:
   - Architect decomposes large tasks into subtasks
   - Each subtask follows its own workflow

4. **Workflow analytics**:
   - Track which edges are taken most frequently
   - Identify bottlenecks and optimize workflows

---

## Summary

Conditional workflows eliminate rigid linear chains and enable:

✅ **Smart routing**: Skip unnecessary steps based on change characteristics
✅ **Feedback loops**: QA can send tasks back to Engineer with structured findings
✅ **Flexibility**: Multiple conditions evaluated in priority order
✅ **Backward compatible**: Legacy linear workflows still supported
✅ **Easy configuration**: YAML-based workflow definitions

**Next steps:**
1. Review the example workflows above
2. Migrate your workflow to use conditional branches
3. Run integration tests to verify routing
4. Monitor workflow execution logs to optimize conditions

For questions or issues, see: [AGENTIC_ROADMAP.md](./AGENTIC_ROADMAP.md) Sprint 2.
