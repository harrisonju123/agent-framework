# Async Agent System - Infinite Loop Prevention & Safeguards

## Overview

The async agent system has multiple layers of protection against infinite loops, resource exhaustion, and runaway task creation.

## Safeguards by Category

### 1. Retry Limits

**Problem**: Tasks could fail and retry indefinitely.

**Safeguards**:
- `MAX_RETRIES=5` - Tasks fail permanently after 5 retry attempts
- Exponential backoff: 30s → 60s → 120s → 240s (max) between retries
- Failed tasks are marked with `status: "failed"` and moved to completed/
- After MAX_RETRIES, an escalation task is created (see #2)

**Configuration**:
```bash
MAX_RETRIES=5 bash scripts/start-async-agents.sh
```

**Location**: `scripts/async-agent-runner.sh:22, 371-391`

---

### 2. Escalation Loop Prevention

**Problem**: Escalation tasks could fail and create more escalations, causing infinite loops.

**Safeguards**:
- **CRITICAL**: Escalations with type `"escalation"` CANNOT create more escalations
- If an escalation fails 5 times, it's marked failed but NO new escalation is created
- Instead, it logs: "Escalation task failed - requires immediate human intervention"
- Escalations automatically use Opus model (strongest reasoning) to maximize success
- Circuit breaker monitors escalation count (limit: 50)

**Code Location**: `scripts/async-agent-runner.sh:373-383`

```bash
if [[ "$task_type" == "escalation" ]]; then
    log "ERROR" "Escalation task $task_id failed after $retry_count retries - NOT creating another escalation (would cause infinite loop)"
    log "ERROR" "This escalation requires immediate human intervention"
else
    create_escalation "$task_file"
fi
```

---

### 3. Circular Dependency Detection

**Problem**: Task A depends on Task B, Task B depends on Task A → deadlock.

**Safeguards**:
- Circuit breaker checks for circular dependencies (max depth: 5)
- Runs recursive dependency check: `check_circular_deps()`
- Reports: `CIRCULAR: task-a -> task-b -> task-a`

**Usage**:
```bash
# Run circuit breaker to detect circular deps
ASYNC_AGENT_WORKSPACE=/path bash scripts/circuit-breaker.sh
```

**Location**: `scripts/circuit-breaker.sh:70-100`

---

### 4. Queue Size Limits

**Problem**: Unbounded queue growth from task creation cascades.

**Safeguards**:
- Max queue size per agent: 100 tasks (configurable via `MAX_QUEUE_SIZE`)
- Circuit breaker alerts when limit exceeded
- `--fix` mode archives old tasks (7+ days) to prevent growth

**Configuration**:
```bash
MAX_QUEUE_SIZE=100 ASYNC_AGENT_WORKSPACE=/path bash scripts/circuit-breaker.sh
```

**Auto-fix**:
```bash
# Archive old tasks and reduce queue size
ASYNC_AGENT_WORKSPACE=/path bash scripts/circuit-breaker.sh --fix
```

**Location**: `scripts/circuit-breaker.sh:31-51`

---

### 5. Task Creation Rate Monitoring

**Problem**: Exponential task spawning (task creates 10 subtasks, each creates 10 more...).

**Safeguards**:
- Circuit breaker monitors task creation rate (last 5 minutes)
- Alerts if >50 tasks created in 5 minutes
- Logs: "Possible cascade - monitor closely for exponential growth"

**Usage**:
```bash
# Monitor task creation rate
ASYNC_AGENT_WORKSPACE=/path bash scripts/circuit-breaker.sh
```

**Location**: `scripts/circuit-breaker.sh:122-132`

---

### 6. Task Timeout

**Problem**: Tasks could hang forever, blocking agent.

**Safeguards**:
- Default task timeout: 1800 seconds (30 minutes)
- Uses `timeout` command (GNU coreutils) when available
- On timeout, task is marked failed and retried (subject to MAX_RETRIES)

**Configuration**:
```bash
TASK_TIMEOUT=3600 bash scripts/start-async-agents.sh  # 1 hour timeout
```

**Location**: `scripts/async-agent-runner.sh:21, 340-348`

---

### 7. Stale Lock Recovery

**Problem**: Crashed agents leave locks, blocking tasks forever.

**Safeguards**:
- Watchdog detects dead agents (heartbeat >90s old)
- Removes stale locks automatically
- Resets in-progress tasks to pending
- Restarts crashed agents

**Location**: `scripts/watchdog.sh:52-71, 87-118`

---

### 8. Duplicate Task Prevention

**Problem**: Same task ID in multiple queues causes conflicts.

**Safeguards**:
- Circuit breaker checks for duplicate task IDs across all queues
- Atomic task writes (write to .tmp, then `mv` to .json)
- File locks prevent concurrent task processing

**Location**: `scripts/circuit-breaker.sh:145-157`

---

### 9. Stale Task Archival

**Problem**: Old unprocessed tasks accumulate forever.

**Safeguards**:
- Circuit breaker detects tasks older than 7 days (configurable)
- `--fix` mode moves to archived/ directory
- Prevents queue bloat from abandoned tasks

**Configuration**:
```bash
MAX_TASK_AGE_DAYS=14 bash scripts/circuit-breaker.sh --fix
```

**Location**: `scripts/circuit-breaker.sh:160-176`

---

### 10. Watchdog Auto-Recovery

**Problem**: Agents crash mid-task and never restart.

**Safeguards**:
- Watchdog checks heartbeats every 60 seconds
- Restarts agents with heartbeat >90s old
- Resets their in-progress tasks
- Cleans up stale locks

**Usage**:
```bash
# Start system with watchdog
ASYNC_AGENT_WORKSPACE=/path bash scripts/start-async-agents.sh --watchdog
```

**Location**: `scripts/watchdog.sh:87-160`

---

## Circuit Breaker Checks Summary

Run `bash scripts/circuit-breaker.sh` to perform all safety checks:

| Check | What It Detects | Threshold |
|-------|----------------|-----------|
| Queue Size | Unbounded growth | 100 tasks/agent |
| Escalation Count | Too many failed tasks | 50 escalations |
| Escalation Retries | Escalations creating escalations | Any retry on escalation |
| Circular Dependencies | Task A → B → A deadlocks | Depth > 5 |
| Task Creation Rate | Exponential spawning | >50 tasks in 5 min |
| Stuck Tasks | Tasks failing repeatedly | ≥3 retries |
| Duplicate IDs | Same task in multiple queues | Any duplicates |
| Stale Tasks | Old unprocessed tasks | >7 days old |

**Exit codes**:
- `0` = All checks passed
- `1` = Issues detected (see output for details)

---

## Monitoring Commands

```bash
# Run circuit breaker
ASYNC_AGENT_WORKSPACE=/path bash scripts/circuit-breaker.sh

# Auto-fix detected issues
ASYNC_AGENT_WORKSPACE=/path bash scripts/circuit-breaker.sh --fix

# Check agent status
ASYNC_AGENT_WORKSPACE=/path bash scripts/status-async-agents.sh

# Investigate stalled tasks
ASYNC_AGENT_WORKSPACE=/path bash scripts/investigate-stalled-tasks.sh

# Watch for infinite loop indicators
watch -n 10 'ls -la /path/.agent-communication/queues/*/ | grep "total"'
```

---

## Emergency Procedures

### If Infinite Loop Detected

1. **Stop all agents immediately**:
   ```bash
   ASYNC_AGENT_WORKSPACE=/path bash scripts/stop-async-agents.sh
   ```

2. **Run circuit breaker analysis**:
   ```bash
   ASYNC_AGENT_WORKSPACE=/path bash scripts/circuit-breaker.sh
   ```

3. **Fix detected issues**:
   ```bash
   ASYNC_AGENT_WORKSPACE=/path bash scripts/circuit-breaker.sh --fix
   ```

4. **Manual cleanup if needed**:
   ```bash
   # Move problematic tasks to archived
   mkdir -p /path/.agent-communication/archived
   mv /path/.agent-communication/queues/agent-d/escalation-*.json \
      /path/.agent-communication/archived/

   # Reset high-retry tasks
   find /path/.agent-communication/queues -name "*.json" -exec \
     jq '.retry_count = 0 | del(.last_failed_at)' {} -c > {}.tmp && mv {}.tmp {}
   ```

5. **Restart with monitoring**:
   ```bash
   ASYNC_AGENT_WORKSPACE=/path bash scripts/start-async-agents.sh --watchdog
   tail -f /path/.agent-communication/logs/*.jsonl | grep -E "ERROR|WARN"
   ```

---

## Configuration Summary

All thresholds are configurable via environment variables:

```bash
# Retry limits
MAX_RETRIES=5              # Max task retry attempts (default: 5)
TASK_TIMEOUT=1800          # Task timeout in seconds (default: 1800 = 30min)
POLL_INTERVAL=30           # Queue polling interval (default: 30s)

# Circuit breaker thresholds
MAX_QUEUE_SIZE=100         # Max tasks per agent queue (default: 100)
MAX_ESCALATIONS=50         # Max total escalations (default: 50)
MAX_TASK_AGE_DAYS=7        # Archive tasks older than N days (default: 7)
MAX_CIRCULAR_DEPS=5        # Max depth for circular dependency check (default: 5)
```

---

## Testing Safeguards

### Test 1: Verify Escalation Loop Prevention
```bash
# Create a task that will fail repeatedly
cat > /path/.agent-communication/queues/agent-d/test-escalation-loop.json << 'EOF'
{
  "id": "test-escalation-loop",
  "type": "escalation",
  "status": "pending",
  "priority": 0,
  "created_by": "test",
  "assigned_to": "agent-d",
  "created_at": "2026-01-01T00:00:00Z",
  "depends_on": [],
  "title": "Test Escalation Loop Prevention",
  "description": "This escalation will fail. It should NOT create another escalation.",
  "failed_task_id": "test-original-task",
  "needs_human_review": true,
  "retry_count": 4
}
EOF

# Watch logs for "NOT creating another escalation" message
tail -f /path/.agent-communication/logs/agent-d.jsonl | grep escalation
```

### Test 2: Verify Circuit Breaker
```bash
# Create 110 tasks in one queue (exceeds MAX_QUEUE_SIZE=100)
for i in {1..110}; do
  cat > /path/.agent-communication/queues/agent-d/test-$i.json << EOF
{"id":"test-$i","type":"testing","status":"pending"}
EOF
done

# Run circuit breaker - should fail check #1
ASYNC_AGENT_WORKSPACE=/path bash scripts/circuit-breaker.sh
# Expected: "✗ agent-d: 110 tasks (exceeds limit: 100)"
```

---

## Architecture Decisions

**Why MAX_RETRIES=5?**
- Balance between giving tasks multiple chances and preventing infinite loops
- With exponential backoff, 5 retries = 30s + 60s + 120s + 240s + 240s = 690s (~11.5 min)
- Enough time to handle transient failures, not so long to block progress

**Why Escalations Use Opus?**
- Escalations represent already-failed tasks
- Need strongest reasoning to understand and resolve failures
- Worth the cost to prevent human intervention

**Why No Escalations from Escalations?**
- Single most critical safeguard against infinite loops
- If an escalation fails 5 times, it genuinely needs human intervention
- Creating more escalations just delays the inevitable

**Why Circuit Breaker Is Separate Script?**
- Can run independently without stopping agents
- Non-invasive monitoring
- Can schedule via cron for continuous monitoring
