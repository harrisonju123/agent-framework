# Workflow Modes

**Version:** 2.0
**Last Updated:** 2026-02-07
**Status:** Active

## Overview

The agent framework uses 3 agents (Architect, Engineer, QA) with three workflow modes to balance speed, thoroughness, and team coordination. The workflow mode is set in `task.context.workflow`.

## Workflow Types

### Simple Workflow

**Use case:** Small, straightforward changes that don't need architecture review

**Flow:**
```
Architect → Engineer (creates PR + self-reviews)
```

**Characteristics:**
- Architect routes task directly to Engineer
- Engineer explores codebase independently
- Engineer creates PR directly after implementation
- Engineer self-reviews before submitting
- Fastest path to production

**When to use:**
- Bug fixes
- Small feature additions
- Documentation updates
- Configuration changes
- <100 lines of code

**Engineer responsibilities:**
- Explore codebase and understand structure
- Implement following existing patterns
- Write tests
- Use test-runner teammate to verify before PR
- Commit and create PR
- Update JIRA to "Code Review" (if jira_key exists)

### Standard Workflow

**Use case:** Medium-sized features that need testing and code review

**Flow:**
```
Architect → Engineer → QA (reviews + creates PR)
```

**Characteristics:**
- Architect routes task to Engineer
- Engineer explores codebase independently
- Engineer commits but does NOT create PR
- QA runs tests, linting, security scanning, and code review
- QA creates PR if everything passes
- Pre-PR verification ensures quality

**When to use:**
- Medium features (100-300 lines)
- Changes requiring integration testing
- Multi-file changes (4-8 files)
- Features with specific acceptance criteria

**Engineer responsibilities:**
- Implement and test
- Use test-runner teammate to catch issues early
- Commit changes locally
- DO NOT create PR
- Queue QA task with acceptance criteria

**QA responsibilities:**
- Run linting/static analysis
- Execute tests and verify acceptance criteria
- Code review (correctness, security, performance, readability)
- If checks pass: Create PR with structured description
- If checks fail: Queue fix task back to Engineer

### Full Workflow

**Use case:** Large features requiring architecture planning and review

**Flow:**
```
Architect → Engineer → QA → Architect (reviews + creates PR)
```

**Characteristics:**
- Architect creates detailed implementation plan
- Engineer follows architectural guidance
- QA verifies tests, linting, security, and reviews code
- Architect does final review and creates PR
- Most thorough review process

**When to use:**
- Large features (>300 lines)
- New system components
- API design changes
- Database schema changes
- Cross-cutting concerns
- Features affecting >8 files

**Architect responsibilities:**
- Review requirements
- Design system architecture
- Create detailed implementation plan
- Specify file changes and patterns
- Queue implementation task to Engineer
- Post-QA: Review implementation against plan, create PR

**Engineer responsibilities:**
- Review architect's plan
- Implement according to plan
- Write tests
- Use test-runner teammate to verify
- Commit changes locally
- DO NOT create PR
- Queue QA task

**QA responsibilities:**
- Run linting/static analysis
- Execute tests
- Verify acceptance criteria
- Code review (correctness, security, performance, readability)
- If checks pass: Queue review task to Architect
- If checks fail: Queue fix task to Engineer

### Failure Loop

When QA finds issues, the failure loop handles retries:

```
QA finds issues → queues fix task to Engineer → Engineer fixes → back to QA
After 5 retries → escalate to Architect for replanning
```

## Workflow Decision Matrix

| Criteria | Simple | Standard | Full |
|----------|--------|----------|------|
| Lines of code | <100 | 100-300 | >300 |
| Files changed | 1-3 | 4-8 | >8 |
| New API/schema | No | No | Yes |
| Architecture impact | None | Low | High |
| Time to PR | ~1 hour | ~2-3 hours | ~4-6 hours |
| Team review | Self-review | Pre-PR (QA) | Pre-PR (Architect + QA) |

## Checking Workflow Mode

In agent prompts, check the workflow mode:

```python
workflow = task.context.get("workflow", "full")  # Default to full if not set

if workflow == "simple":
    # Direct implementation → PR
    pass
elif workflow == "standard":
    # Implementation → QA → PR
    pass
elif workflow == "full":
    # Plan → Implementation → QA → Architect Review → PR
    pass
```

## Workflow Mode Assignment

**Architect sets workflow mode** when creating tasks:

```json
{
  "task_type": "implementation",
  "context": {
    "workflow": "simple",
    "jira_key": "PROJ-123",
    ...
  }
}
```

Decision criteria:
- Analyze task complexity
- Estimate lines of code
- Consider architectural impact
- Default to "full" if uncertain

## PR Creation Responsibility

| Workflow | Who Creates PR? | When? |
|----------|-----------------|-------|
| Simple | Engineer | Immediately after implementation |
| Standard | QA | After quality checks and code review pass |
| Full | Architect | After QA passes and architecture review |

## Benefits by Workflow

### Simple
- Fast iteration
- Minimal overhead
- Good for experienced changes
- Self-review catches obvious issues

### Standard
- Pre-PR quality gate
- Verified tests and linting
- Code review by QA
- Good balance of speed/quality

### Full
- Architectural consistency
- Design review before implementation
- Comprehensive verification
- Prevents rework on complex features

## Best Practices

1. **Start with full workflow** for new features until patterns are established
2. **Use simple workflow** for bug fixes and small tweaks
3. **Standard workflow** is the sweet spot for most features
4. **Escalate to full** if Engineer encounters architectural questions
5. **Track workflow effectiveness** - if many standard PRs need rework, use full more often
