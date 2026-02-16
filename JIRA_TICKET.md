# JIRA Ticket: AF-XXX

**Status:** To Be Created
**Project:** AF
**Type:** Story
**Priority:** Medium

---

## Summary

Extend ModelSelector with specialization-based routing

---

## Description

### Objective

Enable ModelSelector to route tasks to appropriate models based on specialization profile (backend/frontend/infrastructure), combining domain knowledge with existing retry escalation logic.

### Problem Statement

Currently, model selection only considers task type and retry count. A frontend task with 3 CSS files gets the same model as a complex backend distributed systems change. The specialization system already detects task domains — we should use this signal for smarter model routing.

### Implementation

#### Changes Required

1. **ModelSelector.select()** - Add optional specialization_profile and file_count parameters
2. **LLMRequest** - Add context field for metadata
3. **Agent** - Pass specialization profile and file count in request context
4. **Backends** - Extract and forward specialization context to model selector

#### Routing Logic

- Backend/infrastructure + file_count > 10 → premium model
- Frontend/docs → default model (no escalation)
- Retry escalation (>=3) always wins
- Backward compatible with existing behavior

### Files Modified

- `src/agent_framework/llm/model_selector.py` (~40 lines)
- `src/agent_framework/llm/base.py` (~5 lines)
- `src/agent_framework/core/agent.py` (~15 lines)
- `src/agent_framework/llm/claude_cli_backend.py` (~10 lines)
- `src/agent_framework/llm/litellm_backend.py` (~10 lines)
- `tests/unit/test_model_selector.py` (~60 lines)

**Total:** ~140 lines across 6 files

---

## Acceptance Criteria

- [ ] ModelSelector accepts optional specialization_profile parameter
- [ ] Backend/infrastructure tasks with high file count use premium model
- [ ] Frontend/docs tasks use default model
- [ ] Existing retry escalation logic preserved
- [ ] All existing tests pass
- [ ] New tests cover specialization-based routing

---

## Testing Strategy

- Unit tests for specialization routing logic
- Backward compatibility tests (no profile provided)
- Integration test: task → specialization detection → model selection

---

## References

- Implementation plan: `IMPLEMENTATION_PLAN.md`
- Related: Specialization system (`engineer_specialization.py`)
- GitHub Repo: harrisonju123/agent-framework
- Branch: agent/architect-1/task-ef764e36

---

## Labels

- `model-routing`
- `specialization`
- `enhancement`
- `backend`

---

## Story Points

**Estimate:** 3 points (Medium complexity, well-scoped)
