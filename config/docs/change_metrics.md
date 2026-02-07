# Change Metrics and PR Size Guidelines

**Version:** 1.0
**Last Updated:** 2026-02-04
**Status:** Active

## Overview

Managing PR size is critical for effective code review and maintainability. This guide provides metrics, thresholds, and escalation procedures for keeping changes reviewable.

## Size Categories

### Small Changes (<100 lines, 1-3 files)

**Characteristics:**
- Quick to review (<15 minutes)
- Low risk
- Easy to understand
- Fast to merge

**Examples:**
- Bug fixes
- Configuration updates
- Documentation changes
- Single function additions
- Small refactorings

**Review expectations:**
- Same-day review
- Minimal back-and-forth
- Quick approval

### Medium Changes (100-300 lines, 4-8 files)

**Characteristics:**
- Reviewable but substantial (15-30 minutes)
- Moderate complexity
- May touch multiple concerns
- Requires focused review time

**Examples:**
- Feature additions
- Multi-file refactorings
- Test coverage improvements
- API additions

**Review expectations:**
- 1-2 day review turnaround
- May require discussion
- Could need revisions

### Large Changes (300-500 lines, 9-15 files)

**Characteristics:**
- Significant review effort (30-60 minutes)
- High complexity
- Cross-cutting changes
- Should consider splitting

**Examples:**
- Major features
- System refactorings
- Large bug fixes
- Integration work

**Review expectations:**
- 2-3 day review turnaround
- Likely needs revisions
- May require multiple reviewers
- Consider architectural review

**⚠️ Warning threshold:** Before creating PR, evaluate if this should be split.

### Too Large (>500 lines, >15 files)

**Characteristics:**
- Cannot be effectively reviewed (>60 minutes)
- Very high complexity
- High risk of issues
- **MUST be split**

**Action required:**
- DO NOT create PR
- Create escalation task for Architect
- Document why change is large
- Suggest how to split into smaller pieces
- Break into multiple PRs

## Measuring Change Size

### Git Diff Stats

Before creating a PR, run:

```bash
git diff --stat origin/main
```

Example output:
```
 src/api/auth.ts        | 145 +++++++++++++++
 src/middleware/jwt.ts  |  89 ++++++++
 src/api/auth.test.ts   | 234 +++++++++++++++++++++
 3 files changed, 468 insertions(+)
```

**Interpretation:**
- Total lines: 145 + 89 + 234 = 468 lines
- Files changed: 3 files
- Category: Large (but acceptable for cohesive feature)

### Analyzing Complexity

Not all lines are equal. Consider:

**High complexity (counts as 2x):**
- New algorithms
- State management
- Concurrency/async code
- Security-sensitive code
- Database schema changes

**Low complexity (counts as 0.5x):**
- Test code
- Documentation
- Configuration files
- Generated code
- Code moves (no logic change)

**Adjusted line count:**
```
200 lines implementation (complex) × 2 = 400
300 lines tests × 0.5 = 150
50 lines docs × 0.5 = 25
---
Adjusted total: 575 lines (consider splitting)
```

## PR Description Metrics

Include these metrics in every PR description:

```markdown
## Change Metrics
- 📊 {line_count} lines changed across {file_count} files
- ✓ Size: {category}
- 🧪 Test coverage: {percentage}%
- ⏱️ Estimated review time: {minutes} minutes
```

Examples:

```markdown
## Change Metrics
- 📊 89 lines changed across 2 files
- ✓ Size: Small (easy to review)
- 🧪 Test coverage: 95%
- ⏱️ Estimated review time: 10 minutes
```

```markdown
## Change Metrics
- 📊 468 lines changed across 7 files
- ⚠️ Size: Large (substantial but cohesive)
- 🧪 Test coverage: 87%
- ⏱️ Estimated review time: 45 minutes
- Note: Cohesive feature - not split further per architectural review
```

## Automated Checks

### Pre-PR Size Check

Engineers should run this before creating PRs:

```bash
#!/bin/bash
# check-pr-size.sh

LINES=$(git diff --stat origin/main | tail -1 | awk '{print $4}')
FILES=$(git diff --name-only origin/main | wc -l)

if [ $LINES -lt 100 ]; then
  echo "✓ Size: Small (<100 lines, $FILES files) - Easy to review"
elif [ $LINES -lt 300 ]; then
  echo "⚠️ Size: Medium ($LINES lines, $FILES files) - Reviewable but substantial"
elif [ $LINES -lt 500 ]; then
  echo "⚠️ Size: Large ($LINES lines, $FILES files) - Consider splitting"
else
  echo "❌ Size: Too Large ($LINES lines, $FILES files) - MUST split"
  echo "Create escalation task for Architect to break down"
  exit 1
fi
```

## Splitting Strategies

### When to Split

**Split by feature:**
- Extract independent features into separate PRs
- Each PR should be independently deployable
- Example: Split "Add auth + Add notifications" into 2 PRs

**Split by layer:**
- Backend changes first
- Frontend changes second
- Database migrations separate
- Example: PR1=API, PR2=UI, PR3=Schema

**Split by refactor + feature:**
- Refactoring PR first (prep work)
- Feature PR second (uses refactored code)
- Example: PR1=Extract service, PR2=Add feature using service

### How to Split

1. **Identify natural boundaries**
   - Look for independent modules
   - Find logical groupings
   - Check for reusable components

2. **Create dependency chain**
   - Order PRs logically
   - Each PR builds on previous
   - Document dependencies

3. **Communicate plan**
   - Create epic in JIRA (if available)
   - Link related PRs
   - Update task descriptions

## Escalation Process

### When Engineer Encounters Large Change

1. **Analyze the change**
   ```bash
   git diff --stat origin/main
   ```

2. **If >500 lines, create escalation task:**
   ```json
   {
     "task_type": "breakdown",
     "assigned_to": "architect",
     "title": "Break down large implementation: {feature_name}",
     "description": "Implementation of {feature} requires {line_count} lines across {file_count} files. This exceeds reviewable size (500 lines). Recommend splitting into: [list suggestions]",
     "context": {
       "original_task_id": "impl-xxx",
       "total_lines": 750,
       "total_files": 18,
       "suggested_split": [
         "Part 1: Database schema (100 lines, 2 files)",
         "Part 2: Core service (300 lines, 6 files)",
         "Part 3: API endpoints (200 lines, 5 files)",
         "Part 4: UI integration (150 lines, 5 files)"
       ]
     }
   }
   ```

3. **Wait for Architect breakdown**
   - Architect creates sub-tasks
   - Each sub-task is <500 lines
   - Tasks have clear dependencies

### When Architect Creates Large Task

Before assigning to Engineer, check complexity:

1. **Estimate lines of code**
   - Review similar features
   - Consider files that will change
   - Account for tests

2. **If estimate >500 lines, break down proactively:**
   - Create multiple implementation tasks
   - Define clear interfaces between parts
   - Set dependency order
   - Each task should be <300 lines (target)

## Monitoring Metrics

Track these metrics over time:

### PR Size Distribution
- % Small (<100 lines)
- % Medium (100-300 lines)
- % Large (300-500 lines)
- % Too Large (>500 lines) - should be near 0%

**Target distribution:**
- 40% Small
- 45% Medium
- 14% Large
- 1% Too Large (with justification)

### Review Time
- Average time to first review
- Average time to approval
- Correlation between size and review time

**Expected correlation:**
- Small: <1 day
- Medium: 1-2 days
- Large: 2-3 days

### Rework Rate
- % of PRs requiring changes
- Average number of revision rounds
- Correlation between size and rework

**Insight:** If Large PRs have >2x rework rate of Small PRs, split more aggressively.

## Best Practices

1. **Default to smaller PRs** - easier to review, faster to merge
2. **Use feature flags** to ship incomplete features safely
3. **Refactor separately** from feature work when possible
4. **Document why** if PR is large but not split
5. **Get early feedback** on architectural changes before full implementation
6. **Batch small related changes** - don't make PRs too granular
7. **Test coverage** should scale with PR size (larger PR = more tests)

## Exceptions

Some changes legitimately can't be split:

### Acceptable Large PRs

**Generated code:**
- Database migrations
- API client generation
- Schema updates
- Document in PR: "Generated code, not manually reviewable"

**Dependency updates:**
- Major version upgrades
- Framework migrations
- Document in PR: "Automated dependency update"

**Refactoring:**
- Renaming/moving files
- Consistent style updates
- Document in PR: "Automated refactoring, no logic change"

**Architectural changes:**
- System-wide pattern changes
- Must be cohesive for correctness
- Document in PR: "Cohesive architectural change, reviewed by architect"

For these cases:
- Include metrics showing lines added/deleted/moved
- Explain why splitting would break functionality
- Get architectural pre-approval
- Consider pair programming for review
