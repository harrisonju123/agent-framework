# Quick Start Example

This example shows how to set up the async agent system for a new project.

## Step 1: Set Up Workspace

```bash
# Create workspace directory
export WORKSPACE=/path/to/your/workspace
mkdir -p $WORKSPACE

# Initialize directory structure (will be created automatically by start script)
# .agent-communication/
#   ├── queues/agent-{a,b,c,d}/
#   ├── completed/
#   ├── locks/
#   ├── heartbeats/
#   └── logs/
```

## Step 2: Create Bootstrap Task

The bootstrap task tells Agent D (CPO) what to work on. Create this file:

`$WORKSPACE/.agent-communication/queues/agent-d/task-bootstrap.json`:

```json
{
  "id": "task-bootstrap",
  "type": "initialize",
  "status": "pending",
  "priority": 0,
  "created_by": "system",
  "assigned_to": "agent-d",
  "created_at": "2026-02-02T00:00:00Z",
  "depends_on": [],
  "title": "Initialize project roadmap",
  "description": "Read the project requirements and create initial tasks.\n\nProject: Build a todo list app\n\nRequirements:\n- User authentication (login/signup)\n- CRUD operations for todos\n- Mark todos as complete/incomplete\n- Filter by status (all/active/completed)\n- Responsive UI (mobile-friendly)\n\nTech stack:\n- Backend: Node.js + Express + PostgreSQL\n- Frontend: React + TypeScript + Tailwind\n- Testing: Jest + React Testing Library\n\nYour job (Agent D - CPO):\n1. Break down requirements into features\n2. Prioritize features (MVP first)\n3. Create architecture tasks for Agent A (CTO)\n4. Define acceptance criteria for each feature",
  "acceptance_criteria": [
    "Created at least one product requirement task in agent-a queue",
    "Features prioritized (MVP vs. future enhancements)",
    "Clear acceptance criteria defined",
    "Dependencies identified"
  ]
}
```

## Step 3: Start the System

```bash
# Set workspace
export ASYNC_AGENT_WORKSPACE=$WORKSPACE

# Start agents with watchdog
bash scripts/start-async-agents.sh --watchdog

# Output:
# 🚀 Starting Async Agent System
# ✓ Agent D (CPO) started - PID: 12345
# ✓ Agent A (CTO) started - PID: 12346
# ✓ Agent B (Engineer) started - PID: 12347
# ✓ Agent C (QA) started - PID: 12348
# 🐕 Watchdog started - PID: 12349
```

## Step 4: Monitor Progress

```bash
# Check status (every 30 seconds)
watch -n 30 'ASYNC_AGENT_WORKSPACE=$WORKSPACE bash scripts/status-async-agents.sh'

# View logs in real-time
tail -f $WORKSPACE/.agent-communication/logs/*.jsonl

# Check for errors
tail -f $WORKSPACE/.agent-communication/logs/*.jsonl | grep ERROR

# Investigate stalled tasks
ASYNC_AGENT_WORKSPACE=$WORKSPACE bash scripts/investigate-stalled-tasks.sh
```

## Step 5: What Happens Next

1. **Agent D (CPO)** reads the bootstrap task
2. Creates product requirement tasks for **Agent A (CTO)**
3. **Agent A** designs architecture
4. Creates implementation tasks for **Agent B (Engineer)**
5. **Agent B** implements features
6. Creates verification tasks for **Agent C (QA)**
7. **Agent C** tests and either:
   - Approves → creates commit task for **Agent A**
   - Finds bugs → creates fix tasks for **Agent B**
8. **Agent A** commits approved code

## Step 6: Review Escalations

When tasks fail 5 times, they escalate:

```bash
# Review escalated tasks
ASYNC_AGENT_WORKSPACE=$WORKSPACE bash scripts/review-escalations.sh

# For each escalation, you can:
# - Resolve as fixed (you fixed it manually)
# - Retry (reset and try again)
# - Won't fix (acknowledge but not fixing)
# - Skip (review later)
```

## Step 7: Stop When Done

```bash
ASYNC_AGENT_WORKSPACE=$WORKSPACE bash scripts/stop-async-agents.sh
```

## Example Task Flow

```
Bootstrap Task (Agent D)
    ↓
Product Requirement: "User Authentication" (Agent D → Agent A)
    ↓
Architecture Design: "Auth System" (Agent A → Agent B)
    ↓
Implementation: "Login Endpoint" (Agent B → Agent C)
    ↓
Verification: "Test Login" (Agent C)
    ↓
    ├─ Pass → Commit Request (Agent C → Agent A) → Done
    └─ Fail → Fix Request (Agent C → Agent B) → Re-verify
```

## Tips

1. **Start Small**: Begin with a simple bootstrap task to understand the flow
2. **Monitor Closely**: Watch logs during first run to see how agents communicate
3. **Review Escalations**: Check daily for tasks that need human intervention
4. **Use Circuit Breaker**: Run `circuit-breaker.sh` to catch issues early
5. **Cost Optimization**: Let auto model selection save costs (default behavior)

## Common Issues

**Agents not processing tasks?**
- Check if agents are running: `ps aux | grep async-agent-runner`
- Check logs: `tail -f $WORKSPACE/.agent-communication/logs/*.jsonl`

**Tasks stuck in retry loop?**
- Run: `bash scripts/investigate-stalled-tasks.sh`
- Check for missing dependencies or circular deps

**Too many escalations?**
- Review with: `bash scripts/review-escalations.sh`
- Check root cause: Similar failures indicate systemic issue

## Next Steps

- Read [../docs/ASYNC_COST_OPTIMIZATION.md](../docs/ASYNC_COST_OPTIMIZATION.md) for cost savings
- Read [../docs/ASYNC_SAFEGUARDS.md](../docs/ASYNC_SAFEGUARDS.md) for safety features
- Read [../docs/ASYNC_ESCALATION_REVIEW.md](../docs/ASYNC_ESCALATION_REVIEW.md) for review process
