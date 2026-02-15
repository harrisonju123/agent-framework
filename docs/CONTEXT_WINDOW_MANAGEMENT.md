# Context Window Management

Comprehensive context window management system to prevent quality decay in long-running tasks.

## Overview

The Context Window Manager tracks token budgets, manages message history progressively, and prioritizes context inclusion to keep tasks within model limits while maintaining quality.

## Features

### 1. Token Budget Tracking

- Per-task token budgets based on task type
- Real-time tracking of input/output tokens
- Configurable budgets via `optimization.token_budgets` in config
- Default budgets:
  - Implementation: 50,000 tokens
  - Planning: 30,000 tokens
  - Review: 25,000 tokens
  - Testing: 20,000 tokens
  - Documentation: 15,000 tokens

### 2. Progressive Message History

- Keeps recent messages verbatim (configurable, default: 3 messages)
- Summarizes older messages once threshold exceeded (default: 10 messages)
- Prevents context window from growing unbounded

### 3. Priority-Based Context Inclusion

Four priority levels for context items:

- **CRITICAL** (1): Task definition, acceptance criteria - always included
- **HIGH** (2): Recent messages, error context - included if budget allows
- **MEDIUM** (3): Tool outputs, summaries - included if space available
- **LOW** (4): Historical context, metadata - dropped first when budget tight

### 4. Automatic Tool Output Summarization

Large tool outputs (>1000 tokens) are automatically summarized:

- **Read/Grep**: Keeps first 20 and last 20 lines with "omitted" marker
- **Bash**: Preserves error lines and last 10 lines of output
- **Generic**: Keeps head and tail portions

### 5. Real-Time Warnings

- **80% budget**: Warning logged, near-limit flag set
- **90% budget**: Checkpoint trigger, suggests task splitting

## Configuration

Add to your `config/config.yaml`:

```yaml
optimization:
  # Token budget tracking
  enable_token_budget_warnings: true
  budget_warning_threshold: 1.3  # Warn at 130% of budget

  # Context window management
  context_window:
    output_reserve: 4096           # Tokens reserved for model output
    summary_threshold: 10          # Messages before summarization
    min_message_retention: 3       # Recent messages to keep verbatim

  # Custom token budgets by task type
  token_budgets:
    implementation: 50000
    planning: 30000
    review: 25000
    testing: 20000
```

## Usage

### Automatic Integration

Context window management is automatically enabled for all tasks. The system:

1. Initializes a `ContextWindowManager` when task starts
2. Tracks token usage after each LLM call
3. Logs warnings when approaching limits
4. Triggers checkpoints when critically low

### Programmatic Usage

```python
from agent_framework.core import ContextWindowManager, ContextPriority

# Create manager with budget
manager = ContextWindowManager(
    total_budget=50000,
    output_reserve=4096,
    summary_threshold=10,
    min_message_retention=3,
)

# Add context items with priority
manager.add_context_item(
    content="Task description...",
    priority=ContextPriority.CRITICAL,
    category="task_definition",
)

manager.add_message("User: Can you implement X?", role="user")
manager.add_tool_output("Read", file_content, summarize=True)

# Update after LLM call
manager.update_token_usage(input_tokens=1000, output_tokens=500)

# Build context respecting budget
context, metadata = manager.build_context()

# Check budget status
status = manager.get_budget_status()
if manager.should_trigger_checkpoint():
    print("Consider splitting task")
```

## Token Estimation

The system uses a heuristic for token estimation:

- **1 token ≈ 3 characters** for English text
- Conservative estimate to avoid underestimation
- Actual tokenization varies by model and content

## Monitoring

### Logs

```
[INFO] Context window manager initialized: budget=50000, available_for_input=45904
[DEBUG] Context budget: 12.5% used (6250/50000 tokens)
[WARNING] Context budget near limit: 82.0% used (41000/50000 tokens)
[WARNING] Context budget critically low (>90% used). Consider splitting task.
```

### Activity Events

The system emits activity events for monitoring:

- `token_budget_exceeded`: When budget exceeded by threshold
- `context_budget_critical`: When >90% budget used

## Benefits

1. **Prevents quality decay**: Keeps context focused and relevant
2. **Avoids token limit errors**: Proactive budget management
3. **Reduces costs**: Summarizes unnecessary verbose outputs
4. **Maintains context**: Keeps critical information prioritized
5. **Enables long tasks**: Progressive summarization allows extended operations

## Implementation Details

### Files

- `src/agent_framework/core/context_window_manager.py`: Core implementation
- `src/agent_framework/core/agent.py`: Integration with Agent class
- `src/agent_framework/core/config.py`: Configuration schema
- `tests/unit/test_context_window_manager.py`: Comprehensive tests

### Classes

- `ContextWindowManager`: Main manager class
- `ContextBudget`: Budget tracking dataclass
- `ContextItem`: Individual context item with priority
- `ContextPriority`: Enum for priority levels

## Testing

Run tests:

```bash
pytest tests/unit/test_context_window_manager.py -v
```

20 comprehensive tests covering:
- Budget initialization and tracking
- Message history management
- Priority-based inclusion
- Tool output summarization
- Progressive summarization
- Token estimation
- Checkpoint triggers

## Future Enhancements

Potential improvements:

1. **Adaptive budgets**: Adjust based on task complexity
2. **Smart summarization**: Use LLM for better summaries (optional)
3. **Context caching**: Reuse common context across tasks
4. **Multi-turn optimization**: Optimize across conversation turns
5. **Budget prediction**: Estimate remaining task budget needs

## Related

- [Optimization Strategies](../tests/test_optimization.py)
- [Token Budget Configuration](config.py)
- [Agent Implementation](agent.py)
