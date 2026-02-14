# Guided Escalation

## Overview

The guided escalation feature provides structured escalation reports with detailed diagnostics when tasks fail after multiple retry attempts. It allows humans to inject targeted guidance into failed tasks and retry them with additional context.

## Problem Statement

Previously, when tasks failed and escalated:
- Escalations were opaque - just basic error messages
- Humans had to dig through logs to understand what happened
- No structured way to provide guidance for retry
- Difficult to identify failure patterns and root causes

## Solution

The guided escalation feature provides:

1. **Structured Escalation Reports** - Detailed failure analysis with:
   - Complete attempt history
   - Root cause hypothesis
   - Suggested interventions
   - Failure pattern detection

2. **Human Guidance Injection** - CLI command to inject expert guidance:
   ```bash
   agent guide <task-id> --hint "Your guidance here"
   ```

3. **Intelligent Retry** - Failed tasks retry with:
   - Human guidance injected into LLM prompt
   - Previous failure context
   - Categorized error history

## Architecture

### Models

#### `RetryAttempt`
Records each retry attempt with:
- `attempt_number`: Sequential attempt counter
- `timestamp`: When the attempt occurred
- `error_message`: Full error message
- `agent_id`: Which agent made the attempt
- `error_type`: Categorized error (network, authentication, validation, etc.)
- `context_snapshot`: Relevant context at time of failure

#### `EscalationReport`
Structured diagnostic report with:
- `task_id`: Original failed task ID
- `original_title`: Task title
- `total_attempts`: Number of retry attempts
- `attempt_history`: List of all `RetryAttempt` records
- `root_cause_hypothesis`: AI-generated hypothesis about failure cause
- `suggested_interventions`: List of actionable suggestions
- `failure_pattern`: Pattern classification (consistent, intermittent, varied)
- `human_guidance`: Optional human-provided guidance for retry

### Error Categorization

Errors are automatically categorized into:
- **network**: Connection issues, timeouts, DNS failures
- **authentication**: Auth failures, permission denied, 401/403
- **validation**: Schema errors, type errors, invalid input
- **resource**: Memory issues, disk full, resource exhaustion
- **logic**: Null references, index errors, assertions
- **unknown**: Uncategorized errors

### Failure Pattern Detection

The system detects patterns across retry attempts:
- **consistent**: Same error type across all attempts
- **intermittent_network**: Primarily network failures
- **varied**: Different error types across attempts
- **single_failure**: Only one attempt before escalation

## Usage

### CLI Commands

#### View Failed Tasks
```bash
agent retry --all  # List all failed tasks
```

#### Inject Human Guidance
```bash
agent guide task-12345 --hint "Use backup API endpoint: https://backup.api.com"
agent guide escalation-67890 --hint "Authentication header changed to X-API-Key"
```

The `guide` command:
1. Finds the failed task
2. Shows current escalation report (if available)
3. Injects your guidance into the task
4. Resets retry count
5. Re-queues the task with guidance

### Escalation Report Structure

When a task escalates, the report includes:

```markdown
## Root Cause Analysis
**Failure Pattern**: consistent
**Hypothesis**: Consistent network errors across all attempts suggest...

## Attempt History (3 attempts)
- **Attempt 1** (2026-02-14 14:30:00)
  Agent: engineer, Type: network
  Error: Connection refused: API endpoint not reachable

- **Attempt 2** (2026-02-14 14:31:00)
  Agent: engineer, Type: network
  Error: Connection timeout after 30s

- **Attempt 3** (2026-02-14 14:32:00)
  Agent: engineer, Type: network
  Error: Network unreachable

## Suggested Interventions
1. Check network connectivity and firewall rules
2. Verify API endpoints are accessible
3. Review rate limiting and retry backoff settings
4. Consider increasing timeout values

## Next Steps
Use `agent guide task-12345 --hint "<your guidance>"` to inject guidance and retry.
```

## Integration with Agent Processing

### Prompt Injection

When a task has human guidance, it's automatically injected into the LLM prompt:

```
## CRITICAL: Human Guidance Provided

A human expert has reviewed this task and provided the following guidance:

<your guidance here>

Please carefully consider this guidance when approaching the task.

## Previous Failure Context

<root cause hypothesis>

Suggested interventions:
1. <intervention 1>
2. <intervention 2>
...
```

### Retry Attempt Tracking

Every time a task fails, the system records:
- Error message and type
- Agent that attempted it
- Timestamp
- Context snapshot

This history is preserved through escalation and used for pattern analysis.

## Example Workflow

1. **Task fails after 5 retries** with network errors
2. **Escalation created** with structured report
3. **Architect reviews** escalation report
4. **Architect runs**: `agent guide task-123 --hint "Use VPN endpoint: https://vpn.internal.api.com"`
5. **Task re-queued** with guidance injected
6. **Engineer agent receives** prompt with human guidance
7. **Task succeeds** using the VPN endpoint

## Benefits

1. **Transparency**: Clear visibility into what failed and why
2. **Efficiency**: Targeted guidance instead of generic retries
3. **Learning**: Pattern detection helps identify systemic issues
4. **Collaboration**: Structured way for humans to assist agents
5. **Debugging**: Complete history for post-mortem analysis

## Configuration

No additional configuration required. The feature is automatically enabled when:
- Task retry count is tracked
- Escalations are created
- Agent processes tasks

## Testing

Comprehensive test coverage includes:
- Escalation report generation
- Error categorization
- Failure pattern detection
- Human guidance injection
- Attempt history preservation
- Description formatting

Run tests:
```bash
pytest tests/unit/test_guided_escalation.py
pytest tests/unit/test_escalation.py  # Verify backward compatibility
```

## Future Enhancements

Potential improvements:
- Machine learning for root cause prediction
- Automatic similarity detection across failed tasks
- Guidance suggestion from past successful retries
- Integration with monitoring/alerting systems
- Batch guidance application to similar failures
