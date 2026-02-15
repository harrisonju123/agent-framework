# Architect Report: Context Window Management

**Date**: 2026-02-15
**Task ID**: chain-chain-chain--architect-pr
**Status**: Planning Complete, Ready for Implementation

---

## Executive Summary

Successfully analyzed the codebase and created a comprehensive architectural plan for implementing **Context Window Management**. The feature addresses the critical problem of long-running tasks exceeding context windows and causing quality decay.

The solution is broken into 4 core components with a phased rollout strategy, estimated at ~600 lines of production code and ~530 lines of tests (total ~1,330 lines, 2-3 engineering days).

---

## Architecture Overview

### Problem Analysis

Current system has:
- ✅ Basic token tracking in optimization config
- ✅ Error truncation for escalation tasks
- ✅ Result summarization patterns
- ✅ Session logging infrastructure

Missing capabilities:
- ❌ Real-time budget tracking during task execution
- ❌ Progressive summarization as budget tightens
- ❌ Priority-based context inclusion
- ❌ Per-task-type budget configuration

### Solution Design

Created 4 modular components:

1. **ContextBudgetTracker** (`context/budget_tracker.py`)
   - Real-time token monitoring with source attribution
   - Configurable thresholds (80%, 90%, 95%)
   - Budget status: OK → WARNING → CRITICAL → EXCEEDED
   - Historical tracking for post-mortem analysis

2. **ContentSummarizer** (`context/content_summarizer.py`)
   - Tool-specific summarization strategies
   - LRU cache to avoid redundant work
   - Progressive compression as budget tightens
   - Preserves errors, test failures, key outputs

3. **ContextPrioritizer** (`context/context_prioritizer.py`)
   - 5-tier priority system (CRITICAL → MINIMAL)
   - Scoring based on recency, relevance, density
   - Exponential decay for older context
   - Task-aware prioritization

4. **ContextManager** (`context/context_manager.py`)
   - Orchestration layer integrating all components
   - Builds optimized prompts under budget constraints
   - Logs all decisions for transparency
   - Feature flag controlled

### Integration Points

| Component | Integration | Impact |
|-----------|-------------|--------|
| `core/agent.py` | Wire ContextManager into `_build_prompt()` | ~50 lines |
| `llm/base.py` | Add `context_budget` field to LLMRequest | ~10 lines |
| `core/session_logger.py` | Log context management events | ~20 lines |
| `config/agent-framework.yaml` | Add context_management section | ~40 lines |

**Total Integration Impact**: ~120 lines of modifications to existing code

---

## Implementation Plan

### Phase 1: Core Infrastructure (~250 lines)

**Deliverables**:
- `src/agent_framework/context/__init__.py`
- `src/agent_framework/context/budget_tracker.py` (~100 lines)
- `src/agent_framework/context/content_summarizer.py` (~150 lines)
- `tests/unit/test_context_budget.py` (~200 lines)
- `tests/unit/test_content_summarizer.py` (~150 lines)

**Acceptance Criteria**:
- ✅ Budget tracker tracks within 5% accuracy
- ✅ Tool-specific summarization strategies working
- ✅ LRU cache prevents redundant summarization
- ✅ Unit tests with >80% coverage

**Status**: ✅ Queued to engineer (`context-mgmt-impl-20260215.json`)

### Phase 2: Prioritization & Management (~200 lines)

**Deliverables**:
- `src/agent_framework/context/context_prioritizer.py` (~120 lines)
- `src/agent_framework/context/context_manager.py` (~180 lines)
- Integration with Agent class

**Will be queued after Phase 1 completion**

### Phase 3: Configuration & Testing (~150 lines + integration)

**Deliverables**:
- Configuration updates
- Agent integration
- End-to-end testing
- Documentation

**Will be queued after Phase 2 completion**

---

## Configuration Schema

```yaml
context_management:
  enabled: true

  budgets:
    planning: 30000
    implementation: 60000
    review: 40000
    testing: 35000
    default: 45000

  warning_thresholds: [0.80, 0.90, 0.95]

  summarization:
    enabled: true
    model: haiku
    max_summary_length: 500
    cache_summaries: true
    tool_output_limits:
      Read: 2000
      Bash: 1000
      Grep: 1500
      default: 1000

  prioritization:
    enabled: true
    recency_decay_factor: 0.5
    min_priority_score: 0.3
```

---

## Technical Decisions

### 1. Token Estimation Strategy

**Decision**: Use 4 chars/token approximation (Claude's average)
**Rationale**: Simple, fast, conservative. Add 10% buffer for safety.
**Alternative Considered**: Use tiktoken library (rejected: external dependency, overkill)

### 2. Summarization Model

**Decision**: Use Haiku for all summarization
**Rationale**: Cheap ($0.25/1M input), fast, good enough for extraction
**Cost Impact**: ~$0.001 per summarization call, negligible

### 3. Priority Levels

**Decision**: 5-tier system (CRITICAL/HIGH/MEDIUM/LOW/MINIMAL)
**Rationale**: Provides fine-grained control without overwhelming complexity
**Alternative Considered**: 3-tier (rejected: insufficient granularity)

### 4. Integration Approach

**Decision**: Feature flag with gradual rollout (0% → 10% → 50% → 100%)
**Rationale**: Minimize risk, collect metrics, validate before full deployment
**Rollback Plan**: Disable via config flag, no code changes needed

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Summarization loses critical info | Medium | High | Tool-specific strategies, preserve errors |
| Token estimation inaccurate | Medium | Medium | Conservative estimates, 10% buffer |
| Performance overhead | Low | Medium | Cache summaries, lazy evaluation |
| Priority scoring misses context | Low | High | Configurable weights, manual override |
| Breaking changes | Very Low | High | Feature flag, backward compatibility |

---

## Success Metrics

Target metrics after full rollout:

1. **Budget Adherence**: ≥95% of tasks stay within budget
2. **Quality Maintenance**: No increase in task failure rate
3. **Warning Accuracy**: Warnings fire before budget exceeded
4. **Performance**: <100ms overhead for context management
5. **Transparency**: 100% of context decisions logged

---

## Codebase Analysis Summary

**Total Python Files**: 94
**Key Modules Analyzed**:
- ✅ `llm/claude_cli_backend.py` - Token tracking via stream JSON
- ✅ `llm/base.py` - LLMRequest/Response interfaces
- ✅ `core/agent.py` - Main agent loop, prompt building
- ✅ `core/session_logger.py` - Event logging
- ✅ `safeguards/escalation.py` - Error truncation patterns
- ✅ `tests/test_optimization.py` - Existing optimization tests

**Existing Infrastructure Leveraged**:
- Session logging for context events
- Optimization config framework
- Error truncation utilities
- Token tracking patterns

---

## Next Steps

### Immediate Actions (This Task)

1. ✅ Create detailed architectural plan
2. ✅ Queue Phase 1 implementation to engineer
3. ⏳ Create JIRA epic and subtasks
4. ⏳ Create pull request for architectural plan

### Engineer Workflow

1. Implement Phase 1 components
2. Write comprehensive unit tests
3. Integrate with session logger
4. Create PR for review

### QA Workflow (After Engineer)

1. Verify unit test coverage >80%
2. Test token estimation accuracy
3. Validate summarization quality
4. Check session logging output
5. Run regression tests

### Architect Review (After QA)

1. Review implementation against plan
2. Validate architectural decisions
3. Check for technical debt
4. Approve PR for merge

---

## Files Created

1. **CONTEXT_WINDOW_MANAGEMENT_PLAN.md** - Full architectural specification (341 lines)
2. **context-mgmt-impl-20260215.json** - Engineer task queue (Phase 1)
3. **ARCHITECT_REPORT.md** - This report

---

## Estimates

| Phase | Production Code | Test Code | Total | Effort |
|-------|----------------|-----------|-------|--------|
| Phase 1 | ~250 lines | ~350 lines | ~600 lines | 1 day |
| Phase 2 | ~200 lines | ~100 lines | ~300 lines | 0.5-1 day |
| Phase 3 | ~150 lines | ~80 lines | ~230 lines | 0.5-1 day |
| **Total** | **~600 lines** | **~530 lines** | **~1,130 lines** | **2-3 days** |

Integration changes: ~120 lines across 4 existing files

**Grand Total: ~1,250 lines** (production + tests + integration)

---

## Validation Strategy

### Unit Testing
- ContextBudgetTracker: threshold detection, status reporting
- ContentSummarizer: tool-specific strategies, cache behavior
- ContextPrioritizer: scoring, ranking, filtering
- ContextManager: end-to-end prompt building

### Integration Testing
- Test with real tasks across all task types
- Verify budget tracking with actual LLM calls
- Validate summarization preserves critical data
- Check context decisions in session logs

### Shadow Mode Testing
- Run both old and new prompt building in parallel
- Collect metrics on token savings
- Validate quality parity
- Identify edge cases

---

## Conclusion

This architectural plan provides a comprehensive, modular, and low-risk approach to implementing Context Window Management. The design leverages existing infrastructure, minimizes breaking changes, and includes robust testing and rollout strategies.

**Recommendation**: Proceed with Phase 1 implementation as queued to engineer.

**Estimated Timeline**:
- Phase 1: 1 day
- Phase 2: 0.5-1 day
- Phase 3: 0.5-1 day
- **Total: 2-3 engineering days to full deployment**

---

**Architect**: Claude Sonnet 4.5
**Branch**: `agent/architect/task-chain-ch`
**Commit**: 73c5fde
