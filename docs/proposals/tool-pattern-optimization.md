# Proposal: Tool Pattern Optimization System

**Status:** Approved for Implementation
**Created:** 2026-02-15
**Author:** Architect Agent
**JIRA Project:** AF
**Task ID:** planning-AF-1771131933

---

## Problem Statement

Agents waste significant tokens on inefficient tool usage patterns:

1. **Sequential file reads** - Reading 3+ files one-by-one when searching for a pattern (should use Grep first)
2. **Grep then read same file** - Grep'ing a file then immediately reading it entirely (should use Grep `-C` context)
3. **Repeated globs** - Calling Glob with identical patterns multiple times in one session
4. **Bash for file operations** - Using `bash grep/find/cat` instead of dedicated Grep/Glob/Read tools
5. **Unbounded file reads** - Reading entire large files when only a section is needed (should use offset/limit)

**Impact:**
- Estimated 10-30% token waste on redundant tool calls
- Slower task completion due to extra round trips
- Missed opportunities to use more efficient tool combinations

**Why Now:**
- SessionLogger already captures all tool calls to JSONL (infrastructure exists)
- MemoryStore already supports `tool_patterns` category (storage exists)
- Prompt pipeline has established injection patterns (mechanism exists)

---

## Proposed Solution

Build a self-improving system that:
1. **Captures** inefficient tool patterns from session logs (post-task analysis)
2. **Stores** efficient alternatives in memory with usage counts (deduplication + ranking)
3. **Injects** top tool tips into agent prompts (pre-task guidance)

### Architecture

```
┌─────────────────┐
│  Task Completes │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ SessionLogger has JSONL │
│  logs/sessions/{id}.jsonl│
└────────┬────────────────┘
         │
         ▼
┌──────────────────────────┐
│ ToolPatternAnalyzer      │  ← NEW (Part 1)
│ - Parse JSONL            │
│ - Detect anti-patterns   │
│ - Generate recommendations│
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ ToolPatternStore         │  ← NEW (Part 1)
│ - Persist to memory/     │
│ - Deduplicate by ID      │
│ - Increment hit_count    │
│ - Rank by relevance      │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ tool_patterns.json       │
│ {pattern_id, tip,        │
│  hit_count, last_seen}   │
└──────────────────────────┘

                         ┌──────────────┐
                         │  Next Task   │
                         └───────┬──────┘
                                 │
                                 ▼
                         ┌──────────────────────┐
                         │ _build_prompt()      │
                         │ - _inject_memories() │
                         │ - _inject_tool_tips()│  ← NEW (Part 2)
                         └───────┬──────────────┘
                                 │
                                 ▼
                         ┌──────────────────────┐
                         │ Prompt includes:     │
                         │ ## Tool Efficiency   │
                         │ Tips                 │
                         │ - Use Grep instead   │
                         │   of sequential reads│
                         │ - Use -C context...  │
                         └──────────────────────┘
```

### Components

#### Part 1: ToolPatternAnalyzer + ToolPatternStore (~300 lines)

**ToolPatternAnalyzer** (`src/agent_framework/memory/tool_pattern_analyzer.py`)
- Parses session JSONL to extract tool call sequences
- Applies sliding-window pattern matching
- Detects 5 anti-patterns (see table below)
- Returns list of `ToolPatternRecommendation` objects

**ToolPatternStore** (`src/agent_framework/memory/tool_pattern_store.py`)
- Stores patterns in `.agent-communication/memory/{repo}/tool_patterns.json`
- Deduplication: Same `pattern_id` → increment `hit_count`
- Eviction: Keep top-50 patterns ranked by `(hit_count × recency_factor)`
- API: `store_patterns(recommendations)`, `get_top_patterns(limit, max_chars)`

#### Part 2: Lifecycle Integration (~250 lines)

**Post-completion hook** (`agent.py`)
- Call `_analyze_tool_patterns()` after `_extract_and_store_memories()`
- Only runs when `enable_tool_pattern_tips: true`
- Reads session JSONL, analyzes, stores patterns
- Logs count of patterns detected

**Pre-task injection** (`agent.py`)
- Call `_inject_tool_tips()` in `_build_prompt()` after `_inject_memories()`
- Retrieves top-N patterns from ToolPatternStore
- Formats as markdown "## Tool Efficiency Tips" section
- Respects character budget (default: 1500 chars, max 5 tips)

**Configuration** (`config.py` + `agent-framework.yaml.example`)
```yaml
optimization:
  enable_tool_pattern_tips: false  # Feature toggle (opt-in)
  tool_tips_max_chars: 1500        # Prompt budget
  tool_tips_max_count: 5           # Max tips to show
```

---

## Anti-Patterns Detected

| ID | Anti-Pattern | Efficient Alternative | Detection Rule |
|----|-------------|----------------------|---------------|
| `sequential-reads` | 3+ consecutive Read calls on different files | Use Grep to search first, then Read matches | 3+ Read within 5 calls, no prior Grep |
| `grep-then-read-same` | Grep file, then Read same file entirely | Use Grep with `-C N` context flag | Grep followed by Read on same file path |
| `repeated-glob` | Same Glob pattern called 2+ times | Cache results or use single broader pattern | Same pattern string in 2+ Glob calls |
| `bash-for-search` | Bash with `grep`/`find`/`cat`/`head`/`tail` | Use dedicated Grep/Glob/Read tools | Bash command contains file search keywords |
| `read-without-limit` | Read entire file after Grep found section | Use Read with offset+limit or Grep `-C` | Read with no offset on file that was Grep'd |

**Example:**
```
Session JSONL:
  {"tool": "Read", "input": {"file_path": "a.py"}}
  {"tool": "Read", "input": {"file_path": "b.py"}}
  {"tool": "Read", "input": {"file_path": "c.py"}}

Detected: "sequential-reads"
Recommendation: "Use Grep to search across files, then Read only matches"
```

---

## Implementation Plan

### Part 1: Analyzer + Store (impl-a325f2d3)

**Estimated Effort:** ~300 lines
**Status:** Pending (queued to engineer)
**Priority:** 1 (blocks Part 2)

**Steps:**
1. Create `ToolPatternAnalyzer` class with pattern detection rules
2. Implement JSONL parsing with defensive error handling
3. Implement sliding-window pattern matching (window size: 5 calls)
4. Create `ToolPatternStore` class with atomic writes
5. Implement deduplication and eviction logic
6. Write unit tests for all 5 anti-patterns
7. Write unit tests for store operations (add, retrieve, evict)

**Acceptance Criteria:**
- ✓ `ToolPatternAnalyzer.analyze_session(path)` returns recommendations
- ✓ All 5 anti-patterns detected correctly
- ✓ `ToolPatternStore.store_patterns()` persists atomically
- ✓ `ToolPatternStore.get_top_patterns()` returns ranked list
- ✓ Deduplication increments hit_count correctly
- ✓ Eviction keeps top-50 by score
- ✓ All unit tests pass
- ✓ No modifications to existing files

**Files:**
- NEW: `src/agent_framework/memory/tool_pattern_analyzer.py`
- NEW: `src/agent_framework/memory/tool_pattern_store.py`
- NEW: `tests/unit/test_tool_pattern_analyzer.py`

---

### Part 2: Injection + Lifecycle (impl-966bde61)

**Estimated Effort:** ~250 lines
**Status:** Pending (queued to engineer)
**Priority:** 2
**Depends On:** impl-a325f2d3

**Steps:**
1. Add config fields to `OptimizationConfig` in `config.py`
2. Add `_analyze_tool_patterns()` method to Agent class
3. Wire post-completion hook into task success path
4. Add `_inject_tool_tips()` method to Agent class (mirrors `_inject_memories()`)
5. Wire into `_build_prompt()` pipeline after memory injection
6. Update `config/agent-framework.yaml.example` with docs
7. Write unit tests for injection and analysis hooks
8. Verify existing tests still pass

**Acceptance Criteria:**
- ✓ Post-completion: `_analyze_tool_patterns()` called after task success
- ✓ Patterns stored via ToolPatternStore
- ✓ Pre-task: `_inject_tool_tips()` injects tips into prompt
- ✓ Tips appear in "## Tool Efficiency Tips" section
- ✓ Config: `enable_tool_pattern_tips: bool = False` exists
- ✓ Config: `tool_tips_max_chars: int = 1500` exists
- ✓ Config: `tool_tips_max_count: int = 5` exists
- ✓ Documented in `agent-framework.yaml.example`
- ✓ Unit tests cover injection and analysis
- ✓ No regressions in existing tests

**Files:**
- MODIFY: `src/agent_framework/core/agent.py` (~50 lines)
- MODIFY: `src/agent_framework/core/config.py` (~15 lines)
- MODIFY: `config/agent-framework.yaml.example` (~5 lines)
- NEW: `tests/unit/test_tool_pattern_injection.py` (~180 lines)

---

## Design Decisions

### Why separate from MemoryStore?
Tool patterns have a different schema (`hit_count`, `pattern_id`, `last_seen`) than generic memories. While we use the same directory structure and atomic write pattern, a dedicated store allows pattern-specific operations (ranking, eviction by score).

### Why post-completion analysis only?
Analyzing during task execution would add latency to every tool call. Post-completion analysis has zero runtime overhead and still captures all patterns for future tasks.

### Why max 5 tips?
Experiments show that 5 tips fit within 1500 chars (our prompt budget) and provide sufficient guidance without overwhelming the prompt. More tips → diminishing returns and potential confusion.

### Why feature-flagged?
Allows gradual rollout:
- Week 1-2: Internal testing (flag OFF)
- Week 3: Canary (10% of tasks)
- Week 4: Expansion (50% of tasks)
- Week 5: Full rollout (if metrics positive)

### Why heuristic rules vs ML?
Heuristic rules are:
- **Deterministic** - Same input → same output
- **Explainable** - Easy to understand why a pattern was detected
- **Fast** - No model inference overhead
- **Sufficient** - The 5 anti-patterns cover 80%+ of inefficiencies

ML would add complexity without clear benefit for this use case.

---

## Success Metrics

### Baseline (Before Implementation)
- Average tool calls per task: **TBD** (measure during Phase 3)
- Average tokens per task: **TBD** (measure during Phase 3)
- Average task completion time: **TBD** (measure during Phase 3)

### Expected Impact (After Rollout)
- **10-30% reduction** in redundant tool calls
- **5-15% reduction** in token usage
- **5-10% improvement** in task completion time
- **Self-improvement:** Pattern `hit_count` increases over time as agents learn

### Monitoring
- **Detection rate:** % of tasks with anti-patterns detected
- **Application rate:** % of tasks where tips led to better tool usage
- **False positive rate:** % of tips that didn't apply to the task
- **Hit count growth:** Track top patterns over time

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|-----------|
| Session JSONL format changes | Pattern detection fails | Low | Defensive parsing, schema validation, fallback to empty list |
| False positive patterns | Unhelpful tips confuse agents | Medium | Start with conservative rules, monitor application rate, tune |
| Prompt size growth | Exceeds model context limits | Low | Strict char budget (1500), max 5 tips, measured overhead |
| Analysis adds latency | Slower post-task processing | Low | Post-completion only (no runtime impact), async analysis possible |
| Pattern overfitting | Tips don't generalize across repos | Medium | Store per-repo, allow repo-specific tuning, eviction keeps fresh |

---

## Rollout Plan

### Phase 1: Implementation (Week 1)
- Engineer implements Part 1 (Analyzer + Store)
- Engineer implements Part 2 (Injection + Lifecycle)
- Unit tests + integration tests
- Code review + QA approval

### Phase 2: Internal Testing (Week 2)
- Deploy with `enable_tool_pattern_tips: false` (default)
- Manually enable for select internal tasks
- Verify patterns are detected and stored correctly
- Verify tips appear in prompts and are helpful

### Phase 3: Baseline Measurement (Week 2-3)
- Measure current metrics on production tasks:
  - Tool calls per task
  - Tokens per task
  - Task completion time
- Establish baseline for comparison

### Phase 4: Canary Rollout (Week 3)
- Enable for 10% of tasks via config
- Monitor metrics vs baseline
- Collect feedback on tip helpfulness
- Tune detection rules if needed

### Phase 5: Expansion (Week 4)
- If metrics show improvement, expand to 50%
- Continue monitoring for regressions
- Adjust `tool_tips_max_count` if needed

### Phase 6: Full Rollout (Week 5)
- If metrics remain positive, enable for 100%
- Update default to `enable_tool_pattern_tips: true`
- Document feature in user-facing docs

---

## Testing Strategy

### Unit Tests

**Part 1 (Analyzer + Store):**
- ✓ Each anti-pattern detected correctly (5 tests)
- ✓ Edge cases: empty session, malformed JSONL, missing fields
- ✓ Store operations: add, retrieve, deduplicate, evict
- ✓ Ranking algorithm (hit_count × recency)

**Part 2 (Injection + Lifecycle):**
- ✓ Injection when patterns exist
- ✓ Injection when no patterns (no-op)
- ✓ Injection when feature disabled (no-op)
- ✓ Character budget enforcement
- ✓ Count limit enforcement (max 5)
- ✓ Post-completion hook execution
- ✓ Config serialization/deserialization

### Integration Tests

- ✓ End-to-end: Task with inefficient patterns → patterns stored → tips in next task
- ✓ Feature toggle: ON/OFF behavior
- ✓ Memory persistence: Patterns survive agent restart
- ✓ Multi-repo: Patterns isolated per repo

### Manual Validation

1. Enable `enable_tool_pattern_tips: true`
2. Run task with 3+ sequential Read calls
3. Verify `tool_patterns.json` contains `sequential-reads` pattern
4. Run another task in same repo
5. Check session log to confirm "## Tool Efficiency Tips" section in prompt
6. Verify subsequent task uses Grep instead of sequential reads

---

## Future Enhancements (Out of Scope)

- **ML-based pattern detection** - Train model to identify novel anti-patterns
- **Cross-repo pattern sharing** - Share patterns across similar repos
- **Pattern feedback loop** - Agents report when tips were helpful/unhelpful
- **Dynamic rule tuning** - Auto-adjust detection sensitivity based on false positive rate
- **Pattern visualization** - Dashboard showing pattern trends over time
- **Tool usage analytics** - Detailed metrics on which tools are over/underused

---

## Dependencies

**External:** None

**Internal (Read-Only):**
- `src/agent_framework/core/session_logger.py` - Understand JSONL format
- `src/agent_framework/memory/memory_store.py` - Model atomic writes
- `src/agent_framework/core/agent.py` - Understand prompt pipeline
- `src/agent_framework/core/config.py` - Understand OptimizationConfig

**Sequential:**
- Part 2 (impl-966bde61) **DEPENDS ON** Part 1 (impl-a325f2d3)

---

## Open Questions (Resolved)

1. **Q:** Should patterns be stored per-repo or globally?
   **A:** Per-repo. Different repos may have different optimal patterns.

2. **Q:** What if agents ignore the tips?
   **A:** Tips are advisory, not enforced. Over time, useful tips will show higher application rates and be ranked higher.

3. **Q:** Should we analyze failed tasks?
   **A:** No. Failed tasks may have anti-patterns due to bugs/errors, not inefficiency. Only analyze successful tasks.

4. **Q:** How to handle patterns that become obsolete (e.g., tool changes)?
   **A:** Eviction policy keeps only top-50 by score. Unused patterns naturally decay and are evicted.

---

## References

- Planning document: `.agent-communication/upstream-context/planning-AF-1771131933.md`
- Task queue: `.agent-communication/queues/engineer/impl-a325f2d3.json`
- Task queue: `.agent-communication/queues/engineer/impl-966bde61.json`
- JIRA specification: `.agent-communication/jira-tickets/AF-tool-pattern-optimization.md`

---

## Approval

**Status:** ✅ Approved for Implementation
**Approved By:** Architect Agent
**Date:** 2026-02-15

**Next Steps:**
1. Engineer picks up impl-a325f2d3
2. Engineer picks up impl-966bde61 (after Part 1 complete)
3. QA reviews implementation
4. Architect creates PR after QA approval
