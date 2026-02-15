# Change Metrics and PR Size Guidelines

## Size Categories

| Category | Lines | Files | Action |
|----------|-------|-------|--------|
| Small | <100 | 1-3 | Ship it |
| Medium | 100-300 | 4-8 | Reviewable, proceed |
| Large | 300-500 | 9-15 | Consider splitting |
| Too Large | >500 | >15 | **Must split** — escalate to Architect |

## Escalation: Too-Large Changes

If >500 lines, do NOT create PR. Create escalation task for Architect:

```json
{
  "task_type": "breakdown",
  "assigned_to": "architect",
  "title": "Break down large implementation: {feature_name}",
  "context": {
    "original_task_id": "impl-xxx",
    "total_lines": 750,
    "total_files": 18,
    "suggested_split": [
      "Part 1: Database schema (100 lines, 2 files)",
      "Part 2: Core service (300 lines, 6 files)",
      "Part 3: API endpoints (200 lines, 5 files)"
    ]
  }
}
```

## Splitting Strategies

- **By feature**: Independent features → separate PRs
- **By layer**: Backend first → Frontend → Schema migrations
- **By refactor + feature**: Prep PR first → Feature PR builds on it

## Automatic Task Decomposition

The architect agent automatically decomposes tasks estimated >500 lines into parallel subtasks:

- **Threshold**: 500 lines estimated (files_to_modify count x 15 lines/file)
- **Target subtask size**: 250 lines (range: 50-300)
- **Max subtasks**: 5
- **Strategy**: Files grouped by directory prefix into natural clusters
- **Fan-in**: When all subtasks complete, a fan-in task advances the workflow to QA

## Exceptions

Large PRs are acceptable for: generated code, dependency updates, file renames/moves, and cohesive architectural changes. Document the reason in the PR description.
