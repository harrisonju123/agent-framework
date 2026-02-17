# Code Review Workflow

## Overview

The code review agent uses a pull-based queue model. PRs don't automatically trigger reviews — tasks are explicitly created and queued for the QA agent.

- **Manual task creation**: Tasks must be explicitly queued
- **Polling-based**: Agents poll their queues every 30 seconds
- **File-based locking**: Prevents duplicate processing across replicas
- **MCP-powered**: Uses MCP tools for GitHub/JIRA interactions during execution

## Architecture

```
Engineer/QA Agent
  1. Creates PR via github_create_pr
  2. Transitions JIRA to "Code Review"
  3. Queues task via queue_task_for_agent()
         │
         ▼
  .agent-communication/queues/qa/
    task-abc123.json  ← PENDING
         │
         │ Polled every 30s
         ▼
  Code Reviewer Agent
    → Uses MCP tools (GitHub API, JIRA API)
    → Posts review comments
    → If approved: transitions JIRA, marks complete
    → If changes needed: creates fix task for engineer
```

## Task Queue

### Directory Structure

```
.agent-communication/
├── queues/{agent}/     # Pending/in-progress tasks
├── completed/{agent}/  # Archived completed tasks
└── locks/              # File locks for in-progress tasks
```

### Task Selection

The queue returns the first task matching all criteria:
1. Status is PENDING
2. All `depends_on` tasks are COMPLETED
3. Not in retry backoff period

Multiple agents can poll the same queue safely — file locking prevents duplicate processing.

### Scalability

Run multiple replicas with `agent start --replicas 3`. All poll the same queue concurrently. Idle agents cost nothing (no token usage).

## Structured QA Findings

QA agents output findings in JSON format:

```json
[
  {
    "file": "src/api/auth.ts",
    "line_number": 45,
    "severity": "CRITICAL",
    "description": "SQL injection vulnerability in login handler",
    "suggested_fix": "Use parameterized queries",
    "category": "security"
  }
]
```

**Severity levels**: CRITICAL, HIGH, MAJOR, MEDIUM, LOW, MINOR, SUGGESTION

Engineers receive a numbered checklist in fix tasks. The system falls back to text-based findings if JSON parsing fails.

## Key Files

| File | Purpose |
|------|---------|
| `src/agent_framework/core/agent.py` | Agent polling loop, review parsing |
| `src/agent_framework/queue/file_queue.py` | Queue pop logic with filtering |
| `config/agents.yaml` | Agent prompts and queue config |
| `mcp-servers/task-queue/src/queue-tools.ts` | `queue_task_for_agent` implementation |
