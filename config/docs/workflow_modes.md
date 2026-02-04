# Workflow Modes

## Overview

The agent framework supports three workflow modes to balance speed, thoroughness, and team coordination. The workflow mode is set in `task.context.workflow`.

## Workflow Types

### Simple Workflow

**Use case:** Small, straightforward changes that don't need architecture review

**Flow:**
```
Product Owner → Engineer → Code Reviewer
```

**Characteristics:**
- Engineer explores codebase independently
- Engineer creates PR directly after implementation
- Code review happens post-PR (on GitHub)
- No QA verification step
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
- Commit and create PR immediately
- Queue code-reviewer task after PR creation
- Update JIRA to "Code Review" (if jira_key exists)

### Standard Workflow

**Use case:** Medium-sized features that need testing but not architecture planning

**Flow:**
```
Product Owner → Engineer → QA → (QA creates PR) → Code Reviewer
```

**Characteristics:**
- Engineer explores codebase independently
- Engineer commits but does NOT create PR
- QA verifies implementation and tests
- QA creates PR if verification passes
- Pre-PR verification ensures quality

**When to use:**
- Medium features (100-300 lines)
- Changes requiring integration testing
- Multi-file changes (4-8 files)
- Features with specific acceptance criteria

**Engineer responsibilities:**
- Implement and test
- Commit changes locally
- DO NOT create PR
- Queue QA task with acceptance criteria

**QA responsibilities:**
- Run linting/static analysis
- Execute tests and verify acceptance criteria
- If tests pass: Create PR with structured description
- If tests fail: Queue fix task back to Engineer
- Update JIRA status appropriately

### Full Workflow

**Use case:** Large features requiring architecture planning and review

**Flow:**
```
Product Owner → Architect → Engineer → QA → (Architect creates PR) → Code Reviewer
```

**Characteristics:**
- Architect creates detailed implementation plan
- Engineer follows architectural guidance
- QA verifies against acceptance criteria
- Architect reviews and creates PR
- Most thorough review process

**When to use:**
- Large features (>300 lines)
- New system components
- API design changes
- Database schema changes
- Cross-cutting concerns
- Features affecting >8 files

**Architect responsibilities:**
- Review product requirements
- Design system architecture
- Create detailed implementation plan
- Specify file changes and patterns
- Queue implementation task to Engineer

**Engineer responsibilities:**
- Review architect's plan
- Ask questions if plan is unclear (via task creation)
- Implement according to plan
- Write tests
- Commit changes locally
- DO NOT create PR
- Queue QA task

**QA responsibilities:**
- Run linting/static analysis
- Execute tests
- Verify acceptance criteria
- If tests pass: Queue review task to Architect
- If tests fail: Queue fix task to Engineer

**Architect (post-QA):**
- Review implementation against plan
- Verify architectural patterns followed
- Create PR if satisfied
- Otherwise queue fixes to Engineer

## Workflow Decision Matrix

| Criteria | Simple | Standard | Full |
|----------|--------|----------|------|
| Lines of code | <100 | 100-300 | >300 |
| Files changed | 1-3 | 4-8 | >8 |
| New API/schema | No | No | Yes |
| Architecture impact | None | Low | High |
| Time to PR | ~1 hour | ~2-3 hours | ~4-6 hours |
| Team review | Post-PR | Pre-PR (QA) | Pre-PR (Architect + QA) |

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

**Product Owner sets workflow mode** when creating tasks:

```json
{
  "task_type": "implementation",
  "context": {
    "workflow": "simple",  // or "standard" or "full"
    "jira_key": "PROJ-123",
    ...
  }
}
```

Decision criteria:
- Analyze task complexity
- Estimate lines of code
- Consider architectural impact
- Check if plan exists from Architect
- Default to "full" if uncertain

## PR Creation Responsibility

| Workflow | Who Creates PR? | When? |
|----------|-----------------|-------|
| Simple | Engineer | Immediately after implementation |
| Standard | QA | After verification passes |
| Full | Architect | After QA passes and architecture review |

## Benefits by Workflow

### Simple
- ✓ Fast iteration
- ✓ Minimal overhead
- ✓ Good for experienced changes
- ✗ Less thorough review

### Standard
- ✓ Pre-PR quality gate
- ✓ Verified tests
- ✓ Documented acceptance
- ✓ Good balance of speed/quality

### Full
- ✓ Architectural consistency
- ✓ Design review before implementation
- ✓ Comprehensive verification
- ✗ Slower (but prevents rework)

## Best Practices

1. **Start with full workflow** for new features until patterns are established
2. **Use simple workflow** for bug fixes and small tweaks
3. **Standard workflow** is the sweet spot for most features
4. **Escalate to full** if Engineer encounters architectural questions
5. **Track workflow effectiveness** - if many standard PRs need rework, use full more often
