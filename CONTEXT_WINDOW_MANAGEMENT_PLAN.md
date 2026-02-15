# Context Window Management - Implementation Plan

## Executive Summary

This feature implements comprehensive context budget tracking, progressive summarization of tool outputs, and priority-based context inclusion to prevent quality decay during long-running tasks. The implementation builds on existing optimization infrastructure while adding real-time budget tracking and intelligent content prioritization.

## Problem Statement

Long-running tasks can exceed Claude's context window limits, causing:
- Quality decay as important context is lost
- Token budget overruns without warning
- Inability to complete complex multi-step tasks
- Poor prioritization of what context to include

## Solution Architecture

### Core Components

1. **ContextBudgetTracker** - Real-time token budget monitoring
2. **ContentSummarizer** - Progressive summarization of tool outputs
3. **ContextPrioritizer** - Priority-based context inclusion
4. **ContextManager** - Orchestration layer integrating all components

### Integration Points

- **LLM Backends** (`claude_cli_backend.py`, `litellm_backend.py`): Pass context budget to tracker
- **Agent Core** (`core/agent.py`): Integrate ContextManager into prompt building
- **Session Logger** (`core/session_logger.py`): Log context management decisions
- **Configuration** (`config/agent-framework.yaml`): Add context management settings

## Detailed Design

### 1. Context Budget Tracker

**File**: `src/agent_framework/context/budget_tracker.py`

```python
class ContextBudgetTracker:
    """Tracks token usage against budgets in real-time."""

    def __init__(self, budget: int, warning_threshold: float = 0.8):
        self.budget = budget
        self.warning_threshold = warning_threshold
        self.used_tokens = 0
        self.history = []

    def add_usage(self, tokens: int, source: str) -> BudgetStatus
    def get_remaining(self) -> int
    def should_warn(self) -> bool
    def should_truncate(self) -> bool
    def get_report(self) -> dict
```

**Features**:
- Real-time token tracking with source attribution
- Configurable warning thresholds (default 80%, 90%, 95%)
- Budget status reporting (OK, WARNING, CRITICAL, EXCEEDED)
- Historical tracking for post-mortem analysis

### 2. Content Summarizer

**File**: `src/agent_framework/context/content_summarizer.py`

```python
class ContentSummarizer:
    """Progressive summarization of tool outputs and context."""

    def __init__(self, llm_backend: LLMBackend, model: str = "haiku"):
        self.llm = llm_backend
        self.model = model
        self.cache = {}

    async def summarize_tool_output(self, tool_name: str, output: str,
                                   max_length: int = 500) -> str
    async def summarize_file_content(self, content: str,
                                    file_path: str) -> str
    async def summarize_conversation_history(self, messages: list) -> str
```

**Features**:
- Intelligent extraction of key information
- Tool-specific summarization strategies:
  - `Read`: Extract file structure, key functions, important patterns
  - `Bash`: Capture command, exit code, critical errors
  - `Grep`: Show match count, sample matches, file list
  - `Edit/Write`: Track files modified, change type
- LRU cache to avoid re-summarizing same content
- Progressive compression as budget tightens

### 3. Context Prioritizer

**File**: `src/agent_framework/context/context_prioritizer.py`

```python
class ContextPrioritizer:
    """Priority-based context inclusion with intelligent ranking."""

    def __init__(self, task: Task):
        self.task = task
        self.priorities = self._calculate_priorities()

    def prioritize_context(self, items: List[ContextItem]) -> List[ContextItem]
    def _calculate_priorities(self) -> dict
    def _score_item(self, item: ContextItem) -> float
```

**Priority Levels** (highest to lowest):
1. **CRITICAL**: Task definition, acceptance criteria, current goal
2. **HIGH**: Recent tool outputs (last 3), current file context, error messages
3. **MEDIUM**: Task history, memory entries, plan context
4. **LOW**: Historical tool outputs (>3 back), auxiliary context
5. **MINIMAL**: Nice-to-have context, background information

**Scoring Factors**:
- Recency weight (exponential decay)
- Task relevance (keyword matching)
- Information density
- Source authority (direct vs derived)

### 4. Context Manager

**File**: `src/agent_framework/context/context_manager.py`

```python
class ContextManager:
    """Orchestrates context budget tracking and optimization."""

    def __init__(self,
                 budget: int,
                 llm_backend: LLMBackend,
                 task: Task,
                 config: dict):
        self.tracker = ContextBudgetTracker(budget)
        self.summarizer = ContentSummarizer(llm_backend)
        self.prioritizer = ContextPrioritizer(task)

    async def build_optimized_prompt(self,
                                    base_prompt: str,
                                    context_items: List[ContextItem]) -> str
    def estimate_tokens(self, text: str) -> int
    async def get_budget_report(self) -> dict
```

**Workflow**:
1. Estimate tokens for base prompt
2. Calculate remaining budget
3. Prioritize context items
4. Include items until budget exhausted
5. Summarize lower-priority items if needed
6. Log decisions for transparency

## Implementation Strategy

### Phase 1: Core Infrastructure (Est. ~250 lines)

**Files to Create**:
- `src/agent_framework/context/__init__.py`
- `src/agent_framework/context/budget_tracker.py` (~100 lines)
- `src/agent_framework/context/content_summarizer.py` (~150 lines)

**Integration**:
- Wire ContextBudgetTracker into Agent class
- Add token estimation utility functions
- Update session logger to track budget events

### Phase 2: Prioritization & Management (Est. ~200 lines)

**Files to Create**:
- `src/agent_framework/context/context_prioritizer.py` (~120 lines)
- `src/agent_framework/context/context_manager.py` (~180 lines)

**Integration**:
- Integrate ContextManager into `Agent._build_prompt()`
- Add priority scoring for different context types
- Update optimization config with context settings

### Phase 3: Configuration & Testing (Est. ~150 lines)

**Files to Modify**:
- `config/agent-framework.yaml.example` - Add context management config
- `src/agent_framework/core/agent.py` - Integrate ContextManager
- `src/agent_framework/llm/base.py` - Add context budget to LLMRequest

**Files to Create**:
- `tests/unit/test_context_budget.py` (~200 lines)
- `tests/unit/test_content_summarizer.py` (~150 lines)
- `tests/unit/test_context_manager.py` (~180 lines)

## Configuration Schema

```yaml
# config/agent-framework.yaml
context_management:
  enabled: true

  # Token budgets per task type
  budgets:
    planning: 30000
    implementation: 60000
    review: 40000
    testing: 35000
    default: 45000

  # Warning thresholds (% of budget)
  warning_thresholds:
    - 0.80  # 80% - log warning
    - 0.90  # 90% - enable aggressive summarization
    - 0.95  # 95% - critical mode, minimal context only

  # Summarization settings
  summarization:
    enabled: true
    model: haiku  # Use cheap model for summarization
    max_summary_length: 500
    cache_summaries: true

    # Tool-specific limits (chars before summarization)
    tool_output_limits:
      Read: 2000
      Bash: 1000
      Grep: 1500
      Glob: 500
      default: 1000

  # Prioritization settings
  prioritization:
    enabled: true
    recency_decay_factor: 0.5  # Exponential decay per step back
    min_priority_score: 0.3    # Drop items below this score
```

## Success Metrics

1. **Budget Adherence**: 95% of tasks stay within token budget
2. **Quality Maintenance**: No degradation in task completion rate
3. **Warning Accuracy**: Warnings fire before budget exceeded
4. **Performance**: <100ms overhead for context management
5. **Transparency**: All context decisions logged for audit

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Summarization loses critical info | HIGH | Tool-specific strategies, preserve errors/key data |
| Overhead degrades performance | MEDIUM | Cache summaries, async processing, lazy evaluation |
| Token estimation inaccurate | MEDIUM | Use conservative estimates, 10% buffer |
| Priority scoring misses important context | HIGH | Configurable scoring weights, manual override support |
| Breaking changes to existing code | LOW | Feature flag, gradual rollout, backward compatibility |

## Testing Strategy

1. **Unit Tests**:
   - ContextBudgetTracker: budget tracking, threshold detection
   - ContentSummarizer: tool-specific summarization strategies
   - ContextPrioritizer: priority scoring, item ranking
   - ContextManager: end-to-end prompt building

2. **Integration Tests**:
   - Test with real tasks across all task types
   - Verify budget tracking accuracy with actual LLM calls
   - Test context prioritization under budget pressure
   - Validate summarization preserves critical information

3. **Regression Tests**:
   - Ensure existing tasks still complete successfully
   - Verify no performance degradation
   - Check backward compatibility with disabled feature

## Rollout Plan

1. **Phase 1**: Feature flag disabled, shadow mode metrics collection
2. **Phase 2**: Enable for 10% of tasks (canary)
3. **Phase 3**: Enable for 50% of tasks
4. **Phase 4**: Full rollout to 100%

## Documentation Requirements

1. **User Documentation**:
   - Configuration guide for context management
   - Budget tuning recommendations per task type
   - Troubleshooting guide for budget exceeded errors

2. **Developer Documentation**:
   - Architecture overview and integration points
   - Adding custom prioritization strategies
   - Extending summarization for new tools

## Dependencies

**Internal**:
- `llm.base.LLMBackend` - For summarization calls
- `core.task.Task` - Task context and metadata
- `core.session_logger.SessionLogger` - Logging decisions

**External**:
- None (uses existing dependencies)

## Estimated Effort

- **Implementation**: ~600 lines of production code
- **Testing**: ~530 lines of test code
- **Documentation**: ~200 lines
- **Total**: ~1330 lines

**Time Estimate**: 2-3 engineering days for full implementation and testing

## File Manifest

### New Files
1. `src/agent_framework/context/__init__.py`
2. `src/agent_framework/context/budget_tracker.py`
3. `src/agent_framework/context/content_summarizer.py`
4. `src/agent_framework/context/context_prioritizer.py`
5. `src/agent_framework/context/context_manager.py`
6. `tests/unit/test_context_budget.py`
7. `tests/unit/test_content_summarizer.py`
8. `tests/unit/test_context_manager.py`

### Modified Files
1. `src/agent_framework/core/agent.py` - Integrate ContextManager (~50 line changes)
2. `src/agent_framework/llm/base.py` - Add context_budget field (~10 lines)
3. `config/agent-framework.yaml.example` - Add config section (~40 lines)
4. `src/agent_framework/core/session_logger.py` - Add context events (~20 lines)

## Success Criteria

✅ Budget tracker accurately tracks token usage within 5% error
✅ Summarization preserves critical information (test failures, errors, key outputs)
✅ Context prioritization reduces prompt size by 30-50% under budget pressure
✅ No increase in task failure rate
✅ <100ms overhead for context management operations
✅ All context decisions logged to session logs
✅ Configuration supports per-task-type budgets
✅ Feature can be disabled via config flag

## Next Steps

1. Create JIRA epic for Context Window Management
2. Break down into subtasks (one per file/component)
3. Queue implementation task to Engineer
4. Set up metrics collection for shadow mode testing
