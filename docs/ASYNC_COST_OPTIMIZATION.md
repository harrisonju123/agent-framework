# Async Agent System - Cost Optimization Guide

## Overview

The async agent system uses Claude AI for task processing. To optimize API costs while maintaining performance, the system automatically selects the appropriate model based on task type and complexity.

## Model Selection Strategy

| Model | Cost | Use Cases | Task Types |
|-------|------|-----------|------------|
| **Haiku** | Cheapest (~$0.25/1M input tokens) | Simple, fast tasks | `testing`, `verification`, `fix`, `bugfix`, `bug-fix`, `coordination`, `status_report`, `documentation` |
| **Sonnet** | Balanced (~$3/1M input tokens) | Core development work | `implementation`, `architecture`, `planning`, `review`, `enhancement` |
| **Opus** | Expensive (~$15/1M input tokens) | Complex reasoning | Tasks with 3+ retries (auto-escalated when difficult) |

## Automatic Model Selection

The system automatically selects the model based on:

1. **Task Type**: Simple tasks (testing, fixes) use Haiku; complex tasks (architecture, implementation) use Sonnet
2. **Retry Count**: Tasks that fail 3+ times are automatically escalated to Opus (assumed to be difficult)
3. **Force Override**: You can force a specific model for all tasks via `FORCE_MODEL` environment variable

## Cost Savings Example

**Typical task distribution in a development sprint:**
- 30% testing/verification tasks → Haiku instead of Sonnet = **92% cost reduction**
- 10% fixes/coordination → Haiku instead of Sonnet = **92% cost reduction**
- 50% implementation/architecture → Sonnet (no change)
- 10% difficult tasks (3+ retries) → Opus (necessary complexity)

**Overall estimated savings: 40-50% token cost reduction**

## Usage

### Default Behavior (Recommended)
```bash
# Models automatically selected based on task type
ASYNC_AGENT_WORKSPACE=/path/to/workspace bash scripts/start-async-agents.sh --watchdog
```

### Force Specific Model for All Tasks
```bash
# Use Haiku for everything (cheapest, but may fail on complex tasks)
FORCE_MODEL=haiku ASYNC_AGENT_WORKSPACE=/path/to/workspace bash scripts/start-async-agents.sh

# Use Sonnet for everything (balanced)
FORCE_MODEL=sonnet ASYNC_AGENT_WORKSPACE=/path/to/workspace bash scripts/start-async-agents.sh

# Use Opus for everything (most capable, most expensive)
FORCE_MODEL=opus ASYNC_AGENT_WORKSPACE=/path/to/workspace bash scripts/start-async-agents.sh
```

### Per-Agent Model Override
```bash
# Agent-d uses Haiku, others use automatic selection
FORCE_MODEL=haiku bash scripts/async-agent-runner.sh agent-d &
bash scripts/async-agent-runner.sh agent-a &
bash scripts/async-agent-runner.sh agent-b &
bash scripts/async-agent-runner.sh agent-c &
```

## Monitoring Model Usage

Check which models are being used:
```bash
# View model selection in logs
grep "Processing task" /path/to/workspace/.agent-communication/logs/*.jsonl | grep -o "model.*)" | sort | uniq -c

# Example output:
#   45 model, type: testing)         → Haiku
#   120 model, type: implementation) → Sonnet
#   5 model, type: architecture)     → Sonnet
#   2 model, type: fix)              → Haiku
```

## Task Type Classification

When creating tasks, ensure the `type` field matches one of these categories for optimal model selection:

**Haiku-optimized types:**
- `testing` - Running tests, checking outputs
- `verification` - Verifying implementation correctness
- `fix` / `bugfix` / `bug-fix` - Small targeted fixes
- `coordination` - Task routing, status updates
- `status_report` - Progress reports
- `documentation` - Writing docs

**Sonnet-optimized types:**
- `implementation` - Writing new features
- `architecture` - Design decisions
- `planning` - Breaking down initiatives
- `review` - Code review
- `enhancement` - Feature improvements

**Opus escalation:**
- Any task with `retry_count >= 3` automatically uses Opus (regardless of type)

## Best Practices

1. **Classify tasks accurately**: Ensure task `type` field reflects the actual work to get optimal model selection
2. **Monitor retry patterns**: If many tasks hit 3+ retries, consider improving task descriptions or dependencies
3. **Use FORCE_MODEL sparingly**: Automatic selection usually gives best cost/quality balance
4. **Budget for Opus escalation**: Plan for ~10% of tasks to escalate to Opus due to complexity
5. **Review escalations**: Check `grep "opus" logs/*.jsonl` to see which tasks needed Opus - these may indicate areas needing better decomposition

## Cost Comparison

Example for 1,000 tasks (average 50K input tokens + 10K output tokens each):

| Scenario | Total Cost |
|----------|-----------|
| All Sonnet (old behavior) | $150 input + $45 output = **$195** |
| Auto-selected (new behavior) | $60 input + $18 output = **$78** |
| **Savings** | **$117 (60% reduction)** |

*Actual savings depend on your task distribution. The more testing/verification/fix tasks you have, the higher the savings.*

## Troubleshooting

**Tasks failing with Haiku:**
- Check if task is too complex for Haiku
- Consider changing task `type` to force Sonnet selection
- Or use `FORCE_MODEL=sonnet` temporarily

**Too many Opus escalations:**
- Review tasks with 3+ retries to identify root causes
- Improve task descriptions or dependencies
- Consider breaking complex tasks into smaller subtasks

**Still hitting usage limits:**
- Reduce `max-turns` per task (currently 999, very high)
- Increase `POLL_INTERVAL` to reduce task frequency
- Use `FORCE_MODEL=haiku` for entire system (risky but cheapest)
