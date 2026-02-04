# JIRA Integration Guidelines

## Overview

Agents integrate with JIRA using MCP (Model Context Protocol) tools. JIRA integration is **optional** and agents must handle both JIRA-connected and local-only modes gracefully.

## Graceful Handling Pattern

### When JIRA is Available (`context.jira_key` exists)

Perform all JIRA operations:
- Update ticket status using allowed transitions
- Add comments with progress updates
- Link PRs to JIRA tickets
- Include JIRA key in PR titles and descriptions

### When JIRA is NOT Available (local mode)

- **Skip all JIRA operations** without errors
- Continue with full GitHub workflow
- Use `task.id` instead of JIRA key in PR descriptions
- PR title format: `"Implements task {task_id} - {title}"`

## JIRA Status Transitions

### Product Owner

Allowed transitions:
- "Backlog" → "Ready for Dev"
- "Ready for Dev" → "In Progress"

Actions:
- Create epics and stories in JIRA
- Break down large initiatives
- Link related tickets

### Software Engineer

Allowed transitions:
- "In Progress" → "Code Review"
- "In Progress" → "Done"

Actions:
- Update status when starting work
- Link PR to JIRA ticket (if jira_key exists)
- Add implementation notes as comments

### QA Engineer

Allowed transitions:
- "Code Review" → "In Progress" (if verification fails)
- "Code Review" → "Done" (if verification passes)

Actions:
- Document test results in JIRA comments
- Link verification evidence
- Update status based on test outcomes

### Code Reviewer

Actions:
- Add review comments to JIRA
- Link code review findings
- Do NOT change ticket status

## MCP Tool Usage

### Check if JIRA is available

```python
if task.context.get("jira_key"):
    # JIRA operations
else:
    # Local-only mode
```

### Update JIRA Status

```json
{
  "tool": "jira_update_status",
  "params": {
    "issue_key": "PROJ-123",
    "status": "Code Review"
  }
}
```

### Add JIRA Comment

```json
{
  "tool": "jira_add_comment",
  "params": {
    "issue_key": "PROJ-123",
    "comment": "Implementation complete. PR #456 created."
  }
}
```

### Link PR to JIRA

```json
{
  "tool": "jira_link_pr",
  "params": {
    "issue_key": "PROJ-123",
    "pr_url": "https://github.com/org/repo/pull/456"
  }
}
```

## Error Handling

- **Never fail tasks** due to JIRA errors
- Log JIRA failures at WARNING level
- Continue with local workflow if JIRA is unavailable
- Include note in PR description if JIRA link failed

## PR Description Format

### With JIRA

```markdown
## Summary
Implements [PROJ-123](https://jira.company.com/browse/PROJ-123) - Add user authentication

...
```

### Without JIRA (local mode)

```markdown
## Summary
Implements task impl-auth-20260204 - Add user authentication

...
```

## Best Practices

1. **Always check for jira_key** before attempting JIRA operations
2. **Never assume JIRA is available** - graceful degradation is required
3. **Log JIRA operations** for debugging but don't fail on errors
4. **Use task.id as fallback** for all references when JIRA is unavailable
5. **Include JIRA context** in queued tasks so downstream agents know the status
