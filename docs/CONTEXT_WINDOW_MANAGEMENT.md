# Context Window Management

Tracks token budgets, manages message history progressively, and prioritizes context inclusion to keep tasks within model limits.

## Features

### Token Budget Tracking

Per-task budgets based on task type (implementation: 50K, planning: 30K, review: 25K, testing: 20K, docs: 15K). Warnings at 80%, checkpoint trigger at 90%.

### Progressive Message History

Keeps recent messages verbatim (default: 3), summarizes older messages once threshold exceeded (default: 10 messages).

### Priority-Based Context Inclusion

- **CRITICAL**: Task definition, acceptance criteria — always included
- **HIGH**: Recent messages, error context — included if budget allows
- **MEDIUM**: Tool outputs, summaries — included if space available
- **LOW**: Historical context, metadata — dropped first

### Tool Output Summarization

Large outputs (>1000 tokens) are automatically truncated: keeps first/last 20 lines for reads, error lines + last 10 for bash output.

## Configuration

```yaml
optimization:
  enable_token_budget_warnings: true
  budget_warning_threshold: 1.3
  context_window:
    output_reserve: 4096
    summary_threshold: 10
    min_message_retention: 3
  token_budgets:
    implementation: 50000
    planning: 30000
    review: 25000
    testing: 20000
```

## Usage

Context window management is automatic for all tasks. For programmatic usage:

```python
from agent_framework.core import ContextWindowManager, ContextPriority

manager = ContextWindowManager(
    total_budget=50000,
    output_reserve=4096,
    summary_threshold=10,
    min_message_retention=3,
)

manager.add_context_item(content="...", priority=ContextPriority.CRITICAL, category="task_definition")
manager.update_token_usage(input_tokens=1000, output_tokens=500)

context, metadata = manager.build_context()
if manager.should_trigger_checkpoint():
    # Consider splitting task
    ...
```

Token estimation uses ~1 token per 3 characters (conservative heuristic).
