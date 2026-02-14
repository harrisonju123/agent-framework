# Code Review Workflow

This document describes how pull requests are queued for and picked up by the code review agent.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Complete Workflow](#complete-workflow)
- [Task Queue System](#task-queue-system)
- [Agent Polling Loop](#agent-polling-loop)
- [Scalability](#scalability)
- [Key Files](#key-files)
- [Verification](#verification)
- [Future Enhancements](#future-enhancements)

## Overview

The code review agent uses a **pull-based queue model** rather than webhooks or push notifications. PRs don't automatically trigger reviews - instead, tasks are explicitly created and queued for the qa agent.

### Key Characteristics

- **Manual Task Creation**: Tasks must be explicitly queued (no automatic webhook listeners)
- **Polling-Based**: Agents poll their queues every 30 seconds
- **File-Based Locking**: Prevents duplicate processing across multiple reviewers
- **Dependency-Aware**: Tasks only execute when dependencies are met
- **MCP-Powered**: Uses MCP tools for GitHub/JIRA interactions during execution

## Architecture

### Pull-Based Queue Model

```
┌─────────────────────────────────────────────────────────┐
│                Engineer/QA Agent                         │
│                                                          │
│  1. Creates PR via github_create_pr                     │
│  2. Transitions JIRA to "Code Review"                   │
│  3. Queues task: queue_task_for_agent(                  │
│       agent_id: "qa",                        │
│       task_type: "review",                              │
│       context: { pr_number, jira_key, ... }             │
│     )                                                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Writes to file system
                     ↓
         ┌───────────────────────────┐
         │ .agent-communication/     │
         │   queues/qa/   │
         │     task-abc123.json      │  ← PENDING
         └───────────┬───────────────┘
                     │
                     │ Polled every 30s
                     ↓
         ┌───────────────────────────┐
         │   Code Reviewer Agent     │
         │                           │
         │  while True:              │
         │    task = queue.pop()     │
         │    if task:               │
         │      process_task(task)   │
         │    sleep(30s)             │
         └───────────┬───────────────┘
                     │
                     │ Uses MCP tools
                     ↓
        ┌────────────┴─────────────┐
        │                          │
        ↓                          ↓
  ┌──────────┐              ┌──────────┐
  │GitHub API│              │ JIRA API │
  │          │              │          │
  │• Get PR  │              │• Comment │
  │• Comment │              │• Trans-  │
  │• Review  │              │  ition   │
  └──────────┘              └──────────┘
```

### Task Creation Flow

Tasks are created programmatically using the `queue_task_for_agent` MCP tool:

```typescript
// Example: Engineer queues review task after creating PR
queue_task_for_agent({
  agent_id: "qa",
  task_type: "review",
  title: "Review PR #456 - Add authentication",
  description: "PR ready for review. Check security and correctness.",
  context: {
    jira_key: "PROJ-123",
    github_repo: "owner/repo",
    pr_number: 456
  }
})
```

**Location**: `mcp-servers/task-queue/src/index.ts:24-100`

## Complete Workflow

### Step 1: Engineer/QA Creates PR

**Agent**: Engineer or QA
**Actions**:
- Commits code and pushes branch
- Creates PR via `github_create_pr` MCP tool
- PR is linked to JIRA ticket

### Step 2: JIRA Status Updated

**Location**: `src/agent_framework/core/agent.py:1223`

```python
# Transition ticket to "Code Review" status
self.jira_client.transition_ticket(jira_key, "code_review")
self.jira_client.add_comment(
    jira_key,
    f"Pull request created: {pr.html_url}"
)
```

This signals that the PR is ready for review.

### Step 3: Task Queued for Code Reviewer

**Agent**: Engineer or upstream workflow
**Action**: Calls `queue_task_for_agent` MCP tool

```typescript
{
  agent_id: "qa",
  task_type: "review",
  title: "Review PR #456",
  description: "Review diff against criteria",
  context: {
    jira_key: "PROJ-123",
    pr_number: 456,
    github_repo: "owner/repo"
  }
}
```

**Result**: Creates JSON file at:
```
.agent-communication/queues/qa/{task-id}.json
```

**Implementation**: `mcp-servers/task-queue/src/queue-tools.ts:59-100`

### Step 4: Code Reviewer Picks Up Task

**Polling Loop**: `src/agent_framework/core/agent.py:179-212`

```python
while self._running:
    # Poll for next task
    task = self.queue.pop(self.config.queue)

    if task:
        await self._handle_task(task)
    else:
        self.logger.debug(
            f"No tasks available for {self.config.id}, "
            f"sleeping for {self.config.poll_interval}s"
        )

    await asyncio.sleep(self.config.poll_interval)  # Default: 30s
```

**Task Selection Logic**: `src/agent_framework/queue/file_queue.py:74-120`

The queue returns the first task that meets all criteria:
1. **Status**: PENDING only
2. **Dependencies**: All `depends_on` tasks must be COMPLETED
3. **Retry Backoff**: Respects exponential backoff for failed tasks
4. **Locking**: Acquires file lock to prevent duplicate processing

### Step 5: Code Review Execution

**Agent Prompt**: `config/agents.yaml:259-306`

The code reviewer analyzes the PR against these criteria:

1. **Correctness**: Logic errors, edge cases, return values
2. **Security**: Vulnerabilities, input validation, secrets
3. **Performance**: Inefficient algorithms, N+1 queries, memory leaks
4. **Readability**: Clear naming, maintainability, duplication
5. **Best Practices**: Conventions, error handling, testing

**Workflow Steps**:

```yaml
# From config/agents.yaml:300-306
1. Fetch PR details using github_get_pr or github_get_pr_by_branch
2. Review the diff against criteria above
3. Post review comments on PR using github_add_pr_comment
4. Update JIRA with review summary using jira_add_comment
5. If approved: transition JIRA to appropriate status
6. If changes needed: create fix task for engineer queue
```

**Available MCP Tools**:
- `github_get_pr` - Fetch PR details and diff
- `github_get_pr_by_branch` - Find PR by branch name
- `github_add_pr_comment` - Post review comments
- `jira_add_comment` - Update JIRA with review notes
- `jira_transition_issue` - Change ticket status
- `queue_task_for_agent` - Create fix task if changes needed

### Step 6: Post-Review Actions

**If Approved**:
- Transition JIRA: `code_review` → `approved`
- Add approval comment to PR
- Mark task as COMPLETED

**If Changes Requested**:
- Post detailed review comments on GitHub
- Create fix task for engineer queue with structured findings:
  ```typescript
  queue_task_for_agent({
    agent_id: "engineer",
    task_type: "fix",
    title: "Address code review feedback for PROJ-123",
    description: `Fix issues identified in review:

## Review Findings

1. [ ] **CRITICAL** (security): src/api/auth.ts:45
    Issue: SQL injection vulnerability in login handler
    Fix: Use parameterized queries with prepared statements

2. [ ] **HIGH** (performance): src/db/query.ts:89
    Issue: N+1 query detected in user data fetch
    Fix: Add eager loading or batch query
`,
    context: {
      jira_key: "PROJ-123",
      pr_number: 456,
      github_repo: "owner/repo"
    },
    depends_on: []  // Can be picked up immediately
  })
  ```
- Transition JIRA: `code_review` → `changes_requested`
- Mark review task as COMPLETED

## Task Queue System

### Directory Structure

```
.agent-communication/
├── queues/
│   ├── qa/          # Code review tasks
│   │   ├── task-abc123.json    # PENDING
│   │   └── task-def456.json    # IN_PROGRESS
│   ├── engineer/               # Implementation tasks
│   ├── qa/                     # QA verification tasks
│   └── architect/              # Architecture planning tasks
├── completed/                  # Archived completed tasks
│   └── qa/
│       └── task-abc123.json
└── locks/                      # File locks for in-progress tasks
    └── task-def456.lock
```

### Task JSON Format

```json
{
  "id": "review-qa-1738588800000-a1b2c3",
  "type": "review",
  "status": "pending",
  "priority": 50,
  "created_by": "engineer-1",
  "assigned_to": "qa",
  "created_at": "2026-02-03T10:00:00.000Z",
  "title": "Review PR #456 - Add authentication",
  "description": "Review PR for security and correctness",
  "depends_on": [],
  "blocks": [],
  "acceptance_criteria": [
    "All security checks pass",
    "Code follows style guidelines"
  ],
  "deliverables": [],
  "notes": [],
  "context": {
    "jira_key": "PROJ-123",
    "jira_project": "PROJ",
    "github_repo": "owner/repo",
    "pr_number": 456,
    "branch_name": "feature/authentication",
    "epic_key": "PROJ-100"
  },
  "retry_count": 0,
  "plan": null
}
```

### Task Statuses

| Status | Meaning | File Location |
|--------|---------|---------------|
| **PENDING** | Queued, not started | `queues/{agent}/` |
| **IN_PROGRESS** | Being processed | `queues/{agent}/` (with lock) |
| **COMPLETED** | Successfully finished | `completed/{agent}/` |
| **FAILED** | Failed after retries | `queues/{agent}/` (archived) |

### Atomic Operations

**Location**: `mcp-servers/task-queue/src/queue-tools.ts:91-96`

Tasks are written atomically to prevent corruption:

```typescript
// Write to tmp then rename (atomic on POSIX)
writeFileSync(tmpFile, JSON.stringify(task, null, 2));
await fs.rename(tmpFile, taskFile);
```

### File Locking

**Location**: `src/agent_framework/queue/file_queue.py:110`

When an agent picks up a task:
1. Acquires exclusive lock: `.agent-communication/locks/{task-id}.lock`
2. Updates task status to `IN_PROGRESS`
3. Processes task
4. Updates status to `COMPLETED`
5. Releases lock

Multiple agents can poll the same queue safely - only one will acquire each task's lock.

## Agent Polling Loop

### Configuration

**Location**: `config/agents.yaml:255-314`

```yaml
- id: qa
  name: Code Reviewer
  queue: qa            # Queue directory name
  enabled: true
  poll_interval: 30               # Seconds between polls (default)
  max_retries: 5                  # Max attempts for failed tasks
  timeout: 1800                   # Task timeout (30 minutes)
```

### Polling Behavior

**Location**: `src/agent_framework/core/agent.py:179-212`

**Interval**: 30 seconds (configurable via `config.poll_interval`)
**Selection**: First PENDING task with met dependencies
**Locking**: File-based locks prevent duplicate processing
**Pause Support**: Responds to pause signals (`.agent-communication/pause/{agent-id}`)

```python
while self._running:
    # Write heartbeat to show agent is alive
    self._write_heartbeat()

    # Check for pause signal
    if self._check_pause_signal():
        if not self._paused:
            self.logger.info(f"Agent {self.config.id} paused")
            self._paused = True
        await asyncio.sleep(self.config.poll_interval)
        continue

    # Poll for next task
    task = self.queue.pop(self.config.queue)

    if task:
        await self._handle_task(task)
    else:
        self.logger.debug(f"No tasks available")

    await asyncio.sleep(self.config.poll_interval)
```

### Task Selection Logic

**Location**: `src/agent_framework/queue/file_queue.py:74-120`

The `pop()` method returns the first eligible task:

```python
def pop(self, queue_id: str) -> Optional[Task]:
    """Get the next available task from a queue."""
    queue_path = self.queue_dir / queue_id
    task_files = sorted(queue_path.glob("*.json"))  # Chronological order

    for task_file in task_files:
        task = self._load_task(task_file)

        # Only process pending tasks
        if task.status != TaskStatus.PENDING:
            continue

        # Check exponential backoff for retries
        if not self._can_retry(task):
            continue

        # Check dependencies are met
        if not self._dependencies_met(task):
            continue

        return task  # First matching task

    return None
```

**Filtering**:
1. ✅ Status is PENDING
2. ✅ Not in retry backoff period
3. ✅ All `depends_on` tasks are COMPLETED

## Scalability

### Multiple Reviewer Instances

**Location**: `src/agent_framework/core/orchestrator.py`

You can run multiple qa agents in parallel:

```bash
# Start 3 code reviewer replicas
agent start --replicas 3
```

This spawns:
- `qa-1`
- `qa-2`
- `qa-3`

All poll the same `qa` queue concurrently.

### How Parallel Processing Works

```
.agent-communication/queues/qa/
├── task-1.json   ← qa-1 acquires lock
├── task-2.json   ← qa-2 acquires lock
└── task-3.json   ← qa-3 acquires lock

All 3 tasks processed simultaneously!
```

**Concurrency Safety**:
1. Each agent polls the same queue directory
2. File locking prevents duplicate processing
3. First agent to acquire lock gets the task
4. Other agents skip and check next task

**Benefits**:
- Review multiple PRs in parallel
- Reduce review latency during high load
- Scale up/down based on workload

**Cost Efficiency**:
- Idle agents cost $0 (no token usage)
- Only active agents consume tokens
- Poll interval (30s) is free

### Configuration

**Location**: `config/agent-framework.yaml`

```yaml
task:
  poll_interval: 30          # Seconds between polls
  max_retries: 5            # Retry failed tasks
  timeout: 1800             # 30 min task timeout

safeguards:
  max_queue_size: 100       # Max tasks per agent
  heartbeat_timeout: 600    # 10 min before considering agent dead
  watchdog_interval: 60     # Check for stale locks every 60s
```

## Key Files

### Core Implementation

| File | Lines | Purpose |
|------|-------|---------|
| `src/agent_framework/core/agent.py` | 179-212 | Agent polling loop |
| `src/agent_framework/queue/file_queue.py` | 74-120 | Queue pop logic with filtering |
| `src/agent_framework/core/orchestrator.py` | - | Spawns multiple agent replicas |
| `config/agents.yaml` | 255-314 | Code reviewer agent definition |

### MCP Tools

| File | Lines | Purpose |
|------|-------|---------|
| `mcp-servers/task-queue/src/index.ts` | 24-100 | Task queue MCP tool definitions |
| `mcp-servers/task-queue/src/queue-tools.ts` | 59-100 | `queue_task_for_agent` implementation |
| `mcp-servers/github/src/index.ts` | - | GitHub MCP tools (PR ops) |
| `mcp-servers/jira/src/index.ts` | - | JIRA MCP tools (transitions) |

### Configuration

| File | Purpose |
|------|---------|
| `config/agents.yaml` | Agent prompts, permissions, queues |
| `config/agent-framework.yaml` | Polling, timeouts, safeguards |
| `config/mcp-config.json` | MCP server configuration |

## Verification

### Check Queue Status

```bash
# View pending review tasks
ls -la .agent-communication/queues/qa/

# Example output:
# task-review-qa-1738588800000-a1b2c3.json  # PENDING
# task-review-qa-1738588900000-d4e5f6.json  # PENDING
```

### Monitor Agent Logs

```bash
# Tail code reviewer logs
tail -f logs/qa.log

# Example log output:
# 16:17:23 INFO [qa] Polling queue: qa
# 16:17:23 INFO [qa] Found task: review-qa-...
# 16:17:23 INFO [qa] [PROJ-123] Starting review of PR #456
# 16:18:45 INFO [qa] [PROJ-123] Review completed (82.3s)
```

### Check Agent Heartbeats

```bash
# View agent heartbeats (updated every poll)
ls -la .agent-communication/heartbeats/

# Example:
# qa.json    # Updated 5s ago
# engineer-1.json       # Updated 10s ago
```

### Test End-to-End

Create a test task manually:

```bash
# Create test review task
cat > .agent-communication/queues/qa/test-task.json << 'EOF'
{
  "id": "test-review-task",
  "type": "review",
  "status": "pending",
  "priority": 50,
  "created_by": "manual-test",
  "assigned_to": "qa",
  "created_at": "2026-02-03T10:00:00.000Z",
  "title": "Test Review",
  "description": "Test task for verification",
  "depends_on": [],
  "blocks": [],
  "context": {
    "jira_key": "TEST-123",
    "github_repo": "owner/repo",
    "pr_number": 999
  },
  "retry_count": 0
}
EOF

# Watch logs for pickup
tail -f logs/qa.log

# Expect: Task picked up within 30 seconds
```

### Verify Task Transitions

```bash
# Check task status changes
watch -n 5 'ls -la .agent-communication/queues/qa/'

# Lifecycle:
# 1. task-abc123.json exists (PENDING)
# 2. .agent-communication/locks/task-abc123.lock created
# 3. task-abc123.json status changes to IN_PROGRESS
# 4. Task completes, moved to completed/qa/
# 5. Lock released
```

## Future Enhancements

The current system requires **manual task queueing**, providing control but requiring explicit workflow orchestration.

### Potential Automation Options

If automatic PR detection is desired, consider these approaches:

#### 1. GitHub Webhook Listener

**Concept**: Service receives GitHub webhook events for PR opens/updates

```
GitHub PR Event → Webhook Listener → queue_task_for_agent → Code Reviewer
```

**Implementation**:
- HTTP server listening for GitHub webhooks
- Validates webhook signatures
- Calls `queue_task_for_agent` MCP tool
- Requires public endpoint or ngrok for local dev

#### 2. Polling Service

**Concept**: Background job periodically scans for new PRs

```
Cron Job → Scan GitHub API → Find PRs in "Code Review" → Queue Tasks
```

**Implementation**:
```bash
# Every 5 minutes, check for PRs
*/5 * * * * python scripts/queue_prs_for_review.py
```

**Filters**:
- PRs with label `ready-for-review`
- PRs where JIRA status is "Code Review"
- PRs not already queued

#### 3. GitHub Actions Integration

**Concept**: Trigger review on PR open/update via GitHub Actions

```yaml
# .github/workflows/queue-review.yml
name: Queue Code Review
on:
  pull_request:
    types: [opened, synchronize]

jobs:
  queue-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Queue review task
        run: |
          # Call API endpoint or directly write task file
          curl -X POST $QUEUE_ENDPOINT/queue \
            -d '{"agent_id":"qa","pr":"${{ github.event.number }}"}'
```

#### 4. JIRA Automation

**Concept**: JIRA automation rule triggers on status change

```
JIRA Ticket → Transitions to "Code Review" → Automation Rule → Webhook → Queue Task
```

**Setup**:
1. Create JIRA automation rule
2. Trigger: Status changes to "Code Review"
3. Action: Send webhook to queue service
4. Service queues task for qa

### Tradeoffs

| Approach | Pros | Cons |
|----------|------|------|
| **Manual (Current)** | Full control, explicit workflow | Requires orchestration |
| **Webhook Listener** | Real-time, event-driven | Requires public endpoint |
| **Polling Service** | Simple, no infrastructure | Delayed (polling interval) |
| **GitHub Actions** | Native integration, secure | Requires repo config per project |
| **JIRA Automation** | Consistent with workflow | Requires JIRA admin access |

### Why Current Design?

The pull-based model provides:
- **Explicit Control**: Tasks only created when explicitly requested
- **Dependency Awareness**: Tasks can depend on other tasks completing
- **Workflow Flexibility**: Easy to customize when/how reviews are triggered
- **Debugging**: Clear audit trail of who queued what

For most use cases, having the engineer/QA agent queue the review task (as part of their workflow) is sufficient and maintains clear workflow boundaries.

## Structured QA Findings

The QA → Engineer feedback mechanism uses structured JSON findings for clear, actionable feedback.

### QA Output Format

QA agents output findings in JSON format wrapped in code blocks:

```json
[
  {
    "file": "src/api/auth.ts",
    "line_number": 45,
    "severity": "CRITICAL",
    "description": "SQL injection vulnerability in login handler",
    "suggested_fix": "Use parameterized queries with prepared statements",
    "category": "security"
  }
]
```

**Severity Levels**: CRITICAL, HIGH, MAJOR, MEDIUM, LOW, MINOR, SUGGESTION
**Categories**: security, performance, correctness, readability, testing, best_practices

### Engineer Checklist Format

Engineers receive a numbered checklist in fix tasks:

```markdown
## Review Findings

1. [ ] **CRITICAL** (security): src/api/auth.ts:45
    Issue: SQL injection vulnerability in login handler
    Fix: Use parameterized queries with prepared statements

2. [ ] **HIGH** (performance): src/db/query.ts:89
    Issue: N+1 query detected in user data fetch
    Fix: Add eager loading or batch query
```

### Backward Compatibility

The system maintains backward compatibility with legacy text-based findings:
- If JSON parsing fails, falls back to regex extraction
- Existing review tasks continue to work
- Engineers receive text summary if no structured findings available

### Implementation

**Location**: `src/agent_framework/core/agent.py`

Key components:
- `QAFinding` dataclass: Structured finding with file, line, severity, description, fix, category
- `ReviewOutcome.structured_findings`: List of parsed QAFinding objects
- `_extract_review_findings()`: Parses JSON blocks and creates QAFinding objects
- `_build_review_fix_task()`: Generates numbered checklist from structured findings

## Summary

The code review workflow is a pull-based system where:

1. **Tasks are explicitly queued** by other agents using `queue_task_for_agent`
2. **Code reviewers poll** their queue every 30 seconds
3. **File-based locking** enables safe parallel processing
4. **MCP tools** provide real-time GitHub/JIRA access during review
5. **Multiple replicas** can be deployed for scalability
6. **Structured findings** provide clear, actionable feedback with JSON + numbered checklists

This design prioritizes explicit control and workflow transparency over automatic triggering.
