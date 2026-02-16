# Implementation Plan: Specialization-Based Model Routing

**Status:** Approved for Implementation
**Estimated Effort:** Medium (~150 lines, 3-4 files)
**Created:** 2026-02-15
**Architect:** architect

---

## Objectives

Enable ModelSelector to route tasks to appropriate models based on specialization profile (backend/frontend/infrastructure), combining domain knowledge with existing retry escalation logic.

### Problem Statement

Currently, model selection only considers task type and retry count. A frontend task with 3 CSS files gets the same model as a complex backend distributed systems change. The specialization system already detects task domains — we should use this signal for smarter model routing.

### Success Criteria

1. ModelSelector accepts optional `specialization_profile` parameter
2. Backend/infrastructure tasks with high file count → premium model
3. Frontend/docs tasks → default model
4. Existing retry escalation logic preserved
5. All existing tests pass
6. New tests cover specialization-based routing

---

## Approach

### 1. Extend ModelSelector.select() Signature

**File:** `src/agent_framework/llm/model_selector.py`

Add optional `specialization_profile` parameter:

```python
def select(
    self,
    task_type: TaskType,
    retry_count: int = 0,
    specialization_profile: Optional['SpecializationProfile'] = None,
    file_count: Optional[int] = None,
) -> str:
```

**Logic:**
- If `retry_count >= 3` → premium model (existing behavior)
- If `task_type` in premium_types → premium model (existing behavior)
- **NEW:** If `specialization_profile` provided:
  - Backend/infrastructure + `file_count > 10` → premium model
  - Frontend/docs → default model (no escalation)
- If `task_type` in cheap_types → cheap model (existing behavior)
- Default → default model

**Rationale:** Layer specialization logic between retry escalation and task type checks. This allows:
- Retry escalation always wins (complex tasks get premium)
- Specialization provides signal for borderline cases
- Task type fallback for unspecialized agents

### 2. Pass Profile from Agent

**File:** `src/agent_framework/core/agent.py`

Update LLM request construction to include specialization context:

**Location:** `_build_llm_request()` or wherever `LLMRequest` is created

```python
# After detecting specialization in _build_prompt()
# self._current_specialization is already set

# Pass to LLM backends via request context
request = LLMRequest(
    prompt=prompt,
    task_type=task.task_type,
    retry_count=task.retry_count,
    # NEW: add specialization context
    context={
        "specialization_profile": self._current_specialization,
        "file_count": len(task.plan.files_to_modify) if task.plan else None,
    }
)
```

### 3. Update Backend Model Selection

**Files:**
- `src/agent_framework/llm/claude_cli_backend.py`
- `src/agent_framework/llm/litellm_backend.py`

Update `select_model()` to extract and pass specialization:

```python
def select_model(self, task_type: TaskType, retry_count: int, context: Optional[dict] = None) -> str:
    """Select appropriate model based on task type, retry count, and specialization."""
    profile = context.get("specialization_profile") if context else None
    file_count = context.get("file_count") if context else None
    return self.model_selector.select(task_type, retry_count, profile, file_count)
```

Update call sites in `complete()`:

```python
if request.task_type:
    model = self.select_model(
        request.task_type,
        request.retry_count,
        context=request.context  # NEW
    )
```

### 4. Update LLMRequest Dataclass

**File:** `src/agent_framework/llm/base.py`

Add `context` field:

```python
@dataclass
class LLMRequest:
    prompt: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    task_type: Optional[TaskType] = None
    retry_count: int = 0
    max_tokens: int = 8192
    temperature: float = 1.0
    working_dir: Optional[Path] = None
    agents: Optional[dict] = None
    context: Optional[dict] = None  # NEW: for specialization and metadata
```

---

## Files to Modify

1. **src/agent_framework/llm/model_selector.py** (~40 lines)
   - Add `specialization_profile` and `file_count` parameters
   - Add specialization-based routing logic
   - Update docstring

2. **src/agent_framework/llm/base.py** (~5 lines)
   - Add `context: Optional[dict]` field to `LLMRequest`

3. **src/agent_framework/core/agent.py** (~15 lines)
   - Extract file count from task.plan
   - Pass specialization profile and file count in LLMRequest context

4. **src/agent_framework/llm/claude_cli_backend.py** (~10 lines)
   - Update `select_model()` signature to accept context
   - Extract and pass profile/file_count to model_selector

5. **src/agent_framework/llm/litellm_backend.py** (~10 lines)
   - Update `select_model()` signature to accept context
   - Extract and pass profile/file_count to model_selector

6. **tests/unit/test_model_selector.py** (~60 lines)
   - Add test cases for specialization-based routing
   - Test backend + high file count → premium
   - Test frontend → default (no escalation)
   - Test infrastructure + high file count → premium

**Total Estimated Lines:** ~140 lines across 6 files

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking change to ModelSelector.select() | High | Use optional parameters with defaults (backward compatible) |
| Specialization profile import cycle | Medium | Use TYPE_CHECKING import or accept profile ID as string |
| Over-aggressive premium model usage | Medium | Start conservative (file_count > 10 threshold), monitor metrics |
| Missing specialization profile | Low | Gracefully handle None, fall back to existing logic |

---

## Testing Strategy

### Unit Tests

1. `test_model_selector.py`:
   - Test specialization routing with mock profiles
   - Test file count thresholds
   - Test backward compatibility (no profile provided)
   - Test retry escalation overrides specialization

2. Integration test:
   - End-to-end test with real task → specialization detection → model selection

### Test Cases

```python
def test_backend_high_file_count_uses_premium():
    selector = ModelSelector()
    backend_profile = SpecializationProfile(id="backend", ...)
    model = selector.select(
        TaskType.IMPLEMENTATION,
        retry_count=0,
        specialization_profile=backend_profile,
        file_count=15
    )
    assert model == "opus"

def test_frontend_uses_default():
    selector = ModelSelector()
    frontend_profile = SpecializationProfile(id="frontend", ...)
    model = selector.select(
        TaskType.IMPLEMENTATION,
        retry_count=0,
        specialization_profile=frontend_profile,
        file_count=3
    )
    assert model == "claude-sonnet-4-5-20250929"

def test_no_profile_uses_existing_logic():
    # Backward compatibility test
    selector = ModelSelector()
    model = selector.select(TaskType.IMPLEMENTATION, retry_count=0)
    assert model == "claude-sonnet-4-5-20250929"
```

---

## Implementation Order

1. **Phase 1:** Update `model_selector.py` with new parameters (backward compatible)
2. **Phase 2:** Update `base.py` to add context field to LLMRequest
3. **Phase 3:** Update backends to accept and pass context
4. **Phase 4:** Update agent.py to populate context with specialization
5. **Phase 5:** Add unit tests

This order ensures each change is independently testable and deployable.

---

## Deployment Notes

- Backward compatible - old code paths work unchanged
- No configuration changes required
- No database migrations
- Monitoring: Track model usage by specialization in metrics

---

## Alternative Approaches Considered

1. **Pass profile ID as string** - Simpler but loses type safety
2. **Add profile to Task object** - More invasive, affects many callsites
3. **Create separate SpecializationModelSelector** - More complex, harder to maintain

**Selected approach:** Optional parameters preserve backward compatibility while enabling new behavior.

---

## References

- User requirement: `.agent-context/summaries/planning-s6-spec-routing-1771204106-architect.md` (not found, relying on task description)
- Existing code: `src/agent_framework/llm/model_selector.py:29-69`
- Specialization system: `src/agent_framework/core/engineer_specialization.py`
