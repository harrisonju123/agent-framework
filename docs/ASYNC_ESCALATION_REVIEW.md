# Escalation Review Process - Manual Human-in-the-Loop

## Overview

When tasks fail after MAX_RETRIES (default: 5), they are automatically escalated to `agent-d` queue as **escalation tasks** with `needs_human_review: true`. This document explains how to review and resolve these escalations manually.

## What Causes Escalations?

Tasks escalate when they:
1. Fail 5 times (MAX_RETRIES exceeded)
2. Hit a blocker that agents cannot resolve
3. Require human decision-making
4. Need configuration/credentials agents don't have
5. Encounter bugs in the codebase that need fixing

## How to Review Escalations

### Quick Start

```bash
# Run the interactive review tool
ASYNC_AGENT_WORKSPACE=/Users/harrisonju/snapedge \
  bash /Users/harrisonju/openclaw/scripts/review-escalations.sh
```

### What You'll See

For each escalation, the tool shows:
- **Task ID**: Escalation identifier
- **Failed Task ID**: The original task that failed
- **Title**: "ESCALATION: Task failed after N retries"
- **Description**: Why it failed, original task details
- **Created By**: Which agent created the escalation
- **Status**: Current status (failed, retry count, etc.)

### Review Actions

For each escalation, you can choose:

| Action | What It Does | When to Use |
|--------|-------------|-------------|
| **1. Resolve as 'Fixed'** | Marks escalation complete, leaves failed task as-is | Issue was fixed externally (manual code change, config update) |
| **2. Resolve as 'Won't Fix'** | Marks escalation complete, acknowledges but won't fix | Task is obsolete, not worth fixing, or waiting on external dependency |
| **3. Retry Failed Task** | Resets retry count to 0, sets status to pending | Transient failure resolved, want to give it another chance |
| **4. Custom Resolution** | Provide custom notes, marks complete | Need to document decision or add context |
| **5. Skip** | Leave in queue for later review | Need more time to investigate |

## Step-by-Step Workflow

### Example 1: Transient Failure (Should Retry)

**Scenario**: Task failed due to rate limiting or temporary network issue.

1. Run review tool
2. Read escalation details
3. Check if blocker is resolved (e.g., rate limit reset)
4. Select **Action 3: Retry Failed Task**
5. Task is reset to pending and will be picked up by agents

**Result**: Failed task has `retry_count: 0` and `status: "pending"`, escalation moved to completed.

### Example 2: Missing Dependency (Won't Fix)

**Scenario**: Task requires ODDS_API_KEY which user hasn't provided yet.

1. Run review tool
2. Read escalation: "Blocked on ODDS_API_KEY"
3. Decision: Not ready to provide key yet
4. Select **Action 2: Resolve as 'Won't Fix'**
5. Add note: "Waiting for user to obtain API key"

**Result**: Escalation marked complete with resolution notes, failed task stays failed.

### Example 3: Bug Fixed Externally (Mark as Fixed)

**Scenario**: Task failed due to bug in code. You manually fixed the bug.

1. Manually fix the bug in your IDE
2. Commit the fix
3. Run review tool
4. Select **Action 1: Resolve as 'Fixed'**
5. Add note: "Fixed bug in commit abc123"

**Result**: Escalation marked complete, failed task stays in completed/ as historical record.

### Example 4: Need More Investigation (Skip)

**Scenario**: Unclear why task failed, need to read logs/debug.

1. Run review tool
2. Read escalation
3. Select **Action 5: Skip**
4. Task remains in escalation queue

**Result**: Can review later after investigation.

## Advanced Usage

### View Only (No Changes)

```bash
# Just view escalations without resolving
ASYNC_AGENT_WORKSPACE=/Users/harrisonju/snapedge \
  bash scripts/review-escalations.sh <<< "5
n"
# Selects Skip (5) for first escalation, then answers 'n' to "continue?"
```

### Batch Review

```bash
# Review all at once
cd /Users/harrisonju/snapedge
for esc in .agent-communication/queues/agent-d/escalation-*.json; do
  echo "Escalation: $(jq -r '.title' "$esc")"
  echo "Failed Task: $(jq -r '.failed_task_id' "$esc")"
  echo "---"
done
```

### Check Escalation Count

```bash
# Quick check
ASYNC_AGENT_WORKSPACE=/Users/harrisonju/snapedge \
  bash scripts/circuit-breaker.sh | grep "Escalation Count"
# Output: ✓ 15 escalations
```

## Common Escalation Patterns

### Pattern 1: Dependency Blocker

**Example**: "Blocked on ODDS_API_KEY"

**Resolution**:
- If you have the key: Provide it, then retry failed task
- If you don't: Mark as "Won't Fix" with note

**Command**:
```bash
# After providing API key
# Create completed marker
echo '{"id":"user-action-odds-api-key-setup","status":"completed"}' > \
  .agent-communication/completed/user-action-odds-api-key-setup.json

# Then retry failed tasks
bash scripts/review-escalations.sh
# Select Action 3 for each blocked task
```

### Pattern 2: Testing Failures

**Example**: "Tests failed - integration test timeout"

**Resolution**:
1. Investigate why tests failed
2. Fix the issue manually or via new task
3. Mark escalation as Fixed

### Pattern 3: Architecture Stuck

**Example**: "Architecture design failed after 5 retries"

**Resolution**:
- Often means agent-a needs more context or constraints
- Manually create architecture document
- Mark escalation as Fixed

### Pattern 4: Circular Dependencies

**Example**: "Task A depends on B, B depends on A"

**Resolution**:
1. Edit one task to remove circular dependency
2. Retry both tasks
3. Mark escalation as Fixed

## Escalation Lifecycle

```
Task Fails (1x) → Retry with backoff
Task Fails (2x) → Retry with longer backoff
Task Fails (3x) → Retry with Opus model (auto-escalate)
Task Fails (4x) → Retry with Opus
Task Fails (5x) → Mark as FAILED
                ↓
         Create ESCALATION
                ↓
        agent-d queue (needs_human_review: true)
                ↓
       Human Review (this tool!)
                ↓
   ┌──────────────┴─────────────────┐
   │                                │
Retry (reset to pending)    Resolve (mark complete)
   │                                │
   └→ Agent picks up again     └→ Done, historical record
```

## Files and Directories

| Path | Purpose |
|------|---------|
| `.agent-communication/queues/agent-d/escalation-*.json` | Active escalations waiting for review |
| `.agent-communication/completed/escalation-*.json` | Resolved escalations (historical) |
| `.agent-communication/logs/human-review.jsonl` | Log of all human review actions |
| `.agent-communication/queues/*/task-*.json` (failed) | Original failed tasks |

## Monitoring Escalations

### Daily Check

```bash
# Quick daily check
ASYNC_AGENT_WORKSPACE=/Users/harrisonju/snapedge \
  bash scripts/circuit-breaker.sh | grep -A 10 "Escalation"
```

### Weekly Review

```bash
# Full weekly review
ASYNC_AGENT_WORKSPACE=/Users/harrisonju/snapedge \
  bash scripts/review-escalations.sh
```

### Alerts (Optional Cron Job)

```bash
# Add to crontab: Alert if >20 escalations
0 9 * * * ASYNC_AGENT_WORKSPACE=/Users/harrisonju/snapedge bash -c 'COUNT=$(find .agent-communication/queues -name "escalation-*.json" | wc -l); if [ $COUNT -gt 20 ]; then echo "WARNING: $COUNT escalations need review"; fi'
```

## Tips & Best Practices

1. **Review Regularly**: Don't let escalations pile up. Review weekly minimum.

2. **Document Decisions**: Use "Custom Resolution" to add notes explaining why you chose an action.

3. **Look for Patterns**: If many escalations have same root cause, fix the root issue.

4. **Reset Wisely**: Only retry if you've actually fixed the blocker. Otherwise you'll just create more escalations.

5. **Use Circuit Breaker**: Run `circuit-breaker.sh` to see if escalations are growing exponentially (bad sign).

6. **Read Logs**: Before deciding, check logs:
   ```bash
   grep "task-id" .agent-communication/logs/*.jsonl | tail -20
   ```

7. **Check Dependencies**: Failed tasks often have unmet dependencies. Use investigation tool:
   ```bash
   ASYNC_AGENT_WORKSPACE=/path bash scripts/investigate-stalled-tasks.sh
   ```

## Troubleshooting

### Q: Escalation loop - escalations creating more escalations?

**A**: This should NEVER happen (prevented by code). If you see it, it's a critical bug.

**Check**:
```bash
# Should return 0
find .agent-communication/queues -name "escalation-*.json" -exec \
  jq -r 'select(.retry_count >= 1) | .id' {} \; | wc -l
```

If > 0, run: `bash scripts/circuit-breaker.sh --fix`

### Q: Too many escalations (>50)?

**A**: System is overwhelmed. Stop agents, review all escalations, then restart.

```bash
# Emergency procedure
bash scripts/stop-async-agents.sh
bash scripts/circuit-breaker.sh --fix
bash scripts/review-escalations.sh
# After reviewing
bash scripts/start-async-agents.sh --watchdog
```

### Q: Can't find failed task file?

**A**: Task may have been moved or deleted. Check completed/:

```bash
ls .agent-communication/completed/task-*.json
```

If still not found, select "Resolve as 'Won't Fix'" and add note.

### Q: Escalation review tool hangs?

**A**: Likely large task JSON or logs. Press Ctrl+C and review manually:

```bash
# View escalations
ls -la .agent-communication/queues/agent-d/escalation-*.json

# View specific escalation
jq . .agent-communication/queues/agent-d/escalation-123.json

# Manually resolve
mv escalation-123.json ../../completed/
```

## Integration with Intervention Claude

You can spawn Claude Code to help review escalations:

```bash
# Spawn intervention Claude for specific failed task
ASYNC_AGENT_WORKSPACE=/Users/harrisonju/snapedge \
  bash scripts/spawn-intervention-claude.sh task-1770003000-i11-testing

# Claude will show:
# - Task details
# - Dependencies
# - Logs
# - Suggested fixes
```

Then use review tool to apply Claude's recommendation.

## Summary

**Escalation Review Workflow:**
1. Run `review-escalations.sh` weekly (or when circuit breaker alerts)
2. For each escalation:
   - Understand why it failed (read description + logs)
   - Decide: Retry? Fix externally? Won't fix? Need more info?
   - Select appropriate action
3. Document your decision (custom notes)
4. Move to next escalation

**Key Principle**: Escalations are the "human override" mechanism. Agents did their best (5 tries!) and need your expertise to proceed.
