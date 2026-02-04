# Logging Standards

## Overview

Consistent logging is critical for debugging, monitoring, and understanding system behavior. This document defines logging standards for the agent framework.

## Log Levels

### ERROR (40)
**When to use:** Errors that prevent a feature from working
- Failed API calls that can't be retried
- Database connection failures
- Critical configuration errors
- Unhandled exceptions that cause task failures

**Examples:**
```python
logger.error(f"Failed to load configuration from {config_path}: {e}")
logger.error(f"Task {task.id} failed after {max_retries} attempts: {error}")
```

### WARNING (30)
**When to use:** Issues that don't prevent operation but should be addressed
- Deprecated feature usage
- Configuration inconsistencies
- Recoverable errors (will retry)
- Resource constraints (disk space, memory)

**Examples:**
```python
logger.warning(f"Could not acquire lock, will retry later")
logger.warning(f"Token usage exceeded budget: {tokens} > {budget}")
logger.warning(f"JIRA operation failed, continuing with local workflow: {e}")
```

### INFO (20)
**When to use:** Important state changes and milestones
- Task lifecycle events (started, completed, failed)
- Agent state changes (idle, working)
- PR creation, merges
- Configuration loading

**Examples:**
```python
logger.info(f"Task {task.id} moved to completed")
logger.info(f"Using repository: {repo} at {path}")
logger.info(f"Queued code review for PR #{pr_number}")
```

### DEBUG (10)
**When to use:** Detailed information for troubleshooting
- Internal state changes
- Detailed timing information
- Intermediate results
- Verbose operation details

**Examples:**
```python
logger.debug(f"Built prompt preview: {prompt[:500]}")
logger.debug(f"Running post-LLM workflow for {task.id}")
logger.debug(f"No PR found in task {task.id}")
```

## Emoji Usage

**Policy: Use sparingly and consistently**

### Approved Emojis

Only use these emojis with their specific meanings:

| Emoji | Meaning | Usage | Example |
|-------|---------|-------|---------|
| ✅ | Success/Completion | Task completed | `✅ Task {id} moved to completed` |
| ❌ | Failure/Error | Critical failure | `❌ Task {id} failed permanently` |
| ⚠️ | Warning | Non-critical issue | `⚠️ Size: Large (consider splitting)` |
| 🔍 | Review/Inspection | Code review queued | `🔍 Queued code review for PR #{n}` |
| 🤖 | LLM Operation | Calling LLM | `🤖 Calling LLM (model: {type})` |
| ⏸️ | Paused/Waiting | Lock not acquired | `⏸️ Could not acquire lock, will retry` |

### Emoji Guidelines

1. **Don't overuse** - Limit to 1 emoji per log statement
2. **Be consistent** - Always use same emoji for same type of event
3. **INFO level and above** - Don't use emojis in DEBUG logs
4. **Optional** - Plain text is always acceptable

**Good:**
```python
logger.info(f"✅ Task {task.id} completed successfully")
logger.warning(f"⚠️ Token budget exceeded: {tokens} > {budget}")
```

**Avoid:**
```python
logger.info(f"✅🎉 Task completed! 🚀")  # Too many emojis
logger.debug(f"🔍 Checking lock status")  # Don't use in DEBUG
```

## Message Format Standards

### Task-related Logs

**Pattern:** `[Status] Task {task_id} - {description}`

```python
# Good
logger.info(f"Task {task.id} started - {task.title}")
logger.error(f"Task {task.id} failed after {retries} retries: {error}")

# Avoid
logger.info(f"Starting task...")  # Missing task ID
logger.info(f"Task failed")  # Missing context
```

### File Operations

**Pattern:** Include operation type and file path

```python
# Good
logger.info(f"Writing task to {queue_path / task_file}")
logger.error(f"Failed to read configuration from {config_path}: {e}")

# Avoid
logger.info(f"Writing file")  # What file?
logger.error(f"Read failed")  # Which file? Why?
```

### API/Network Operations

**Pattern:** Include endpoint/resource and outcome

```python
# Good
logger.info(f"Created PR #{pr_number} in {repo}")
logger.warning(f"JIRA API rate limited, retrying in {delay}s")

# Avoid
logger.info(f"API call succeeded")  # Which API?
logger.warning(f"Retrying...")  # Why?
```

### Time-sensitive Operations

**Include duration for operations > 1 second**

```python
# Good
duration = time.time() - start
logger.info(f"LLM call completed in {duration:.2f}s")
logger.info(f"Task completed in {duration_ms}ms")

# Avoid
logger.info(f"LLM call completed")  # How long did it take?
```

## Sensitive Data Handling

### Never Log These

- API keys, tokens, passwords
- Full JIRA keys (use masked versions)
- Email addresses
- Full file contents
- Full git diffs (log stats only)

### Safe Logging Patterns

```python
# Bad
logger.debug(f"Using token: {github_token}")
logger.info(f"Processing ticket PROJ-123")

# Good
logger.debug(f"Using token: {github_token[:8]}...")
logger.info(f"Processing ticket {jira_key or 'local-task'}")

# For debugging, sanitize:
task_id_short = task.id[:8] + "..." if len(task.id) > 8 else task.id
logger.debug(f"Task {task_id_short}: prompt length = {len(prompt)}")
```

## Error Logging Best Practices

### Include Context

```python
# Bad
logger.error(f"Error: {e}")

# Good
logger.error(f"Failed to create PR in {repo}: {e}")
logger.error(f"Task {task.id} validation failed: {'; '.join(errors)}")
```

### Use Exception Logging

```python
# Use logger.exception() in except blocks to include stack trace
try:
    result = risky_operation()
except Exception as e:
    logger.exception(f"Error during risky operation: {e}")
    # Stack trace automatically included
```

### Log Before Raising

```python
# Good pattern
try:
    validate_config(config)
except ValueError as e:
    logger.error(f"Invalid configuration: {e}")
    raise  # Re-raise after logging
```

## Structured Logging

### Key-Value Pairs (for machine parsing)

When logs will be parsed by tools, use structured format:

```python
logger.info(
    f"task_completed",
    extra={
        "task_id": task.id,
        "duration_ms": duration_ms,
        "tokens": total_tokens,
        "cost": cost,
    }
)
```

### Multi-line Logs

For complex information, use consistent formatting:

```python
logger.info(
    f"Agent {agent_id} summary:\n"
    f"  Tasks completed: {completed}\n"
    f"  Tasks failed: {failed}\n"
    f"  Average duration: {avg_duration}s"
)
```

## Testing and Debugging

### Debug Logging in Tests

Use `caplog` fixture to test logging:

```python
def test_error_logging(caplog):
    with caplog.at_level(logging.ERROR):
        agent.process_task(invalid_task)

    assert "Task validation failed" in caplog.text
```

### Log Levels in Development

**Development:**
```python
logging.basicConfig(level=logging.DEBUG)
```

**Production:**
```python
logging.basicConfig(level=logging.INFO)
```

**CI/Testing:**
```python
logging.basicConfig(level=logging.WARNING)
```

## Performance Considerations

### Avoid String Formatting in Hot Paths

```python
# Bad (formats even if DEBUG disabled)
logger.debug(f"Processing {expensive_format(data)}")

# Good (only formats if DEBUG enabled)
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Processing {expensive_format(data)}")
```

### Rate Limit Repeated Logs

```python
# For logs in loops, rate limit:
if iteration % 100 == 0:
    logger.debug(f"Processed {iteration} items")
```

## Migration Guide

### Standardizing Existing Logs

**Step 1:** Audit current logging
```bash
# Find logs without context
grep -r "logger\.(error|warning)\(.*\)" --include="*.py" | grep -v "f\""

# Find emoji usage
grep -r "logger\.*[🔧🤖✅]" --include="*.py"
```

**Step 2:** Add context to error logs
```python
# Before
logger.error(f"Failed: {e}")

# After
logger.error(f"Failed to process task {task.id}: {e}")
```

**Step 3:** Standardize emoji usage
```python
# Before
logger.info(f"🔧 Task processing...")
logger.info(f"Task done! ✓")

# After
logger.info(f"Processing task {task.id}")
logger.info(f"✅ Task {task.id} completed")
```

## Examples by Component

### Agent Task Processing

```python
logger.info(f"Task {task.id} started - {task.title}")
logger.debug(f"Built prompt: {len(prompt)} chars")
logger.info(f"🤖 Calling LLM (model: {model}, attempt: {attempt})")
logger.info(f"✅ Task {task.id} completed in {duration}s")
```

### Worktree Management

```python
logger.info(f"Creating worktree for {repo} at {path}")
logger.debug(f"Branch name: {branch}")
logger.warning(f"Worktree limit reached: {count}/{max_count}")
logger.info(f"Cleaned up worktree: {path}")
```

### Queue Operations

```python
logger.debug(f"Checking queue: {queue_id}")
logger.info(f"Task {task.id} added to queue: {queue_id}")
logger.warning(f"⏸️ Could not acquire lock for {task.id}, will retry")
logger.error(f"Failed to write task to queue {queue_id}: {e}")
```

## Tools and Utilities

### Log Analysis

```bash
# Find all ERROR logs
grep "ERROR" agent.log

# Find all task completions
grep "✅ Task" agent.log

# Find slow operations (>10s)
grep -E "completed in [0-9]{2,}\." agent.log
```

### Log Aggregation

Consider using structured logging libraries:
- `structlog` - Structured logging
- `python-json-logger` - JSON formatting
- Integrate with log aggregation services (Datadog, Splunk, ELK)

## References

- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)
- [Logging Best Practices](https://docs.python-guide.org/writing/logging/)
- Agent Framework utilities: `src/agent_framework/utils/error_handling.py`
