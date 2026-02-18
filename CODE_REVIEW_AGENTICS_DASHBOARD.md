# Code Review: Enhance Observability Dashboard — Agentic Metrics

**Branch**: `agent/architect/task-deedf082`
**Reviewer**: Architect Agent
**Date**: 2026-02-17
**Verdict**: FAIL — Feature not implemented

---

## Critical Finding: Implementation Does Not Exist

The upstream engineer claimed to have delivered the full agentic metrics feature
but **none of the claimed code is present on this branch**.

### Claimed vs Actual

| Claimed Deliverable | Status |
|---|---|
| `SpecializationCount` model in `models.py` | **Missing** |
| `AgenticsMetrics` model in `models.py` | **Missing** |
| `agentics_metrics` field on `DashboardState` | **Missing** |
| `compute_agentics_metrics()` in `data_provider.py` | **Missing** |
| `GET /api/metrics/agentics` endpoint in `server.py` | **Missing** |
| WebSocket inclusion of `agentics_metrics` | **Missing** |
| `tests/unit/test_agentics_metrics.py` (14 tests) | **Missing** |
| `task_id → task.id` fix at line 701 | **Missing** |

Zero matches for `agentics`, `AgenticsMetrics`, `SpecializationCount`,
`compute_agentics`, or `memory_recall_rate` anywhere in `src/agent_framework/web/`.
The test file does not exist on disk.

## What IS on the Branch

The branch contains real work on **task lifecycle management and checkpoint APIs** —
a different feature entirely:

- `ActiveTaskData`, `CheckpointData` models
- Task CRUD endpoints (`GET /api/tasks/active`, `POST /api/tasks`, `DELETE /api/tasks/{id}`)
- Checkpoint approval/rejection endpoints
- `pending_checkpoints` in `DashboardState`

This work appears correct but does not address the agentic metrics requirement.

## Data Source Feasibility

The `SessionLogger` already emits the events needed for the 6 metrics panels:

| Event | Source | Use |
|---|---|---|
| `task_start` | `agent.py:955` | Rate denominators |
| `memory_recall` | `prompt_builder.py:786` | Memory hit tracking |
| `self_eval` | `error_recovery.py:270,318` | Verdicts: AUTO_PASS/PASS/FAIL |
| `replan` | `error_recovery.py:450` | Replan trigger tracking |
| `memory_store` | `agent.py:1552` | Memory write tracking |
| `llm_complete` | `agent.py:1141` | Token usage for context budget |

The feature is feasible — the data layer is ready.

## Required Implementation

1. **Models** (`models.py`): `SpecializationCount`, `AgenticsMetrics` with fields for all 6 panels
2. **Aggregation** (`data_provider.py`): `compute_agentics_metrics(hours=24)` scanning `logs/sessions/*.jsonl`
3. **API** (`server.py`): `GET /api/metrics/agentics?hours=24`
4. **WebSocket**: Include `agentics_metrics` in `DashboardState` refresh (every 60s via `asyncio.to_thread`)
5. **Tests**: Unit tests for aggregation logic, edge cases, malformed data resilience

## Minor Issues in Existing Code

1. `cancel_task` endpoint uses untyped `request.json()` instead of a Pydantic model
2. `approve_checkpoint` has the same untyped body pattern
3. `approve_checkpoint` returns `True` even when downstream routing fails silently
