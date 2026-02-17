# Conditional Workflow Branches

The agent framework supports conditional workflow branches using a DAG-based workflow engine. Workflows dynamically route tasks based on runtime conditions like file patterns, PR size, test results, and agent signals.

## Quick Start

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
          - target: create_pr
            condition: files_match
            params:
              pattern: "*.md"
            priority: 10

          - target: create_pr
            condition: pr_size_under
            params:
              max_files: 3
            priority: 5

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

Tasks automatically use the configured workflow. The framework evaluates conditions in priority order (highest first) and routes to the first matching edge.

## Condition Types

### files_match

Routes based on glob patterns against changed files.

```yaml
- target: create_pr
  condition: files_match
  params:
    pattern: "*.md"   # glob, not regex
  priority: 10
```

### pr_size_under

Routes based on number of changed files.

```yaml
- target: create_pr
  condition: pr_size_under
  params:
    max_files: 3
  priority: 5
```

### test_passed / test_failed

Routes based on `task.context["test_result"]` (set by the testing agent).

### approved / needs_fix

Routes based on QA verdict. Detection uses structured `task.context["verdict"]` with regex fallback.

### signal_target

Allows agents to explicitly route by emitting a routing signal with `target_agent`.

### always

Unconditional fallback. Use at lowest priority.

## Priority and Edge Evaluation

Edges evaluate in descending priority order (highest first). Best practice:
- **10+**: Most specific conditions (docs-only, config-only)
- **5**: Size-based or test-based routing
- **1**: Default fallback (`always`)

## Changed Files Detection

1. If `task.context["changed_files"]` exists, uses that
2. Otherwise runs `git diff --name-only HEAD`
3. Falls back to empty list if git fails

## QA Feedback Loop

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

Review cycles are capped at `MAX_DAG_REVIEW_CYCLES` (default 2) to prevent infinite loops.

## Backward Compatibility

Legacy linear format is still supported and auto-converted to a DAG:

```yaml
workflows:
  default:
    agents: [architect, engineer, qa]
    pr_creator: architect
```

To migrate: add `steps` and `start_step`, add conditional edges, remove `agents` list.

## Troubleshooting

**Condition never matches**: Check that `changed_files` is populated, verify glob syntax (not regex), check priority order.

**Infinite loops**: Unconditional cycles are rejected at parse time. Conditional cycles are allowed but capped.

**Git diff fails**: Explicitly set `changed_files` in task context as a workaround.
