# Autonomous Pipeline Context for Agent Teams

This document is injected into interactive Agent Team sessions so teammates
understand how to interact with the autonomous pipeline.

## Available MCP Tools

The task-queue MCP server provides these tools for pipeline integration:

- **queue_task_for_agent** — Push a task to any agent queue (engineer, qa, architect, etc.)
- **listPendingTasks** — See what's queued across all agents
- **getQueueStatus** — Check queue depths and health
- **getTaskDetails** — Inspect a specific task's full context

## Queue Structure

Tasks flow through file-based queues at `.agent-communication/queues/{agent_id}/`.
Each task is a JSON file with: id, type, status, title, description, context, acceptance_criteria.

## Handoff Conventions

When handing work to the autonomous pipeline:
1. Use `queue_task_for_agent` with a clear title and description
2. Include `github_repo` and `jira_key` in the task context when available
3. Set task type to match the work: `implementation`, `testing`, `review`, `bug-fix`
4. The autonomous pipeline will pick up tasks, create branches, implement, and open PRs

## Receiving Work from the Pipeline

Failed autonomous tasks may be escalated to an interactive team for resolution.
The escalation context includes: original task details, error history, retry count,
and JIRA context. Use this to diagnose and fix the issue, then hand the fix back.
