# CLI/Web Consolidation & Feature Parity Analysis

**Date:** 2026-02-19
**Goal:** Consolidate CLI tools, improve UX, and achieve feature parity between CLI and web interface

---

## Executive Summary

The agent framework currently has **25+ CLI commands** and a **web dashboard with 30+ API endpoints**. This analysis identifies:

1. **Feature gaps** between CLI and web
2. **Consolidation opportunities** (redundant/underused commands)
3. **Recommended refactoring** to improve discoverability and UX
4. **Implementation roadmap** for achieving CLI/web parity

---

## Current CLI Commands (25 commands)

### Core Operations (8 commands)
| Command | Description | Web Equivalent |
|---------|-------------|----------------|
| `agent init` | Initialize new project | ❌ No (Setup UI exists but not full init) |
| `agent start` | Start agents with options | ✅ `/api/agents/start-all` |
| `agent stop` | Stop agents | ✅ `/api/agents/stop-all` |
| `agent up` | Start agents + dashboard + watchdog | ⚠️ Partial (`/api/agents/start-all`) |
| `agent down` | Stop everything | ⚠️ Partial (`/api/agents/stop-all`) |
| `agent restart` | Restart agents | ✅ `/api/agents/{id}/restart` |
| `agent status` | Show agent status (with --watch for TUI) | ✅ `/api/agents` + WebSocket |
| `agent analytics` | Show analytics TUI dashboard | ⚠️ Partial (`/api/agentic-metrics`) |

### Task/Queue Management (10 commands)
| Command | Description | Web Equivalent |
|---------|-------------|----------------|
| `agent work` | Interactive mode to describe work | ✅ `/api/operations/work` |
| `agent pull` | Pull JIRA tickets to queue | ❌ No |
| `agent run TICKET` | Work on specific JIRA ticket | ✅ `/api/operations/run-ticket` |
| `agent analyze` | Analyze repo and create epic | ✅ `/api/operations/analyze` |
| `agent summary EPIC` | Show epic progress summary | ❌ No |
| `agent retry TASK` | Retry failed task | ✅ `/api/tasks/{id}/retry` |
| `agent cancel TASK` | Cancel task | ✅ `/api/tasks/{id}/cancel` |
| `agent guide TASK` | Inject human guidance into failed task | ❌ No |
| `agent clear` | Clear queues | ❌ No |
| `agent purge` | Full workspace reset | ❌ No |

### System Health & Control (4 commands)
| Command | Description | Web Equivalent |
|---------|-------------|----------------|
| `agent doctor` | Run health checks | ✅ `/api/health` |
| `agent check` | Run circuit breaker checks | ⚠️ Partial (part of `/api/health`) |
| `agent pause` | Pause agent processing | ✅ `/api/system/pause` |
| `agent resume` | Resume agent processing | ✅ `/api/system/resume` |

### UI/Dashboards (2 commands)
| Command | Description | Web Equivalent |
|---------|-------------|----------------|
| `agent dashboard` | Start web dashboard server | N/A (the server itself) |
| `agent status --watch` | Live TUI dashboard | ✅ Web dashboard UI |

### Advanced/Power User (3 commands)
| Command | Description | Web Equivalent |
|---------|-------------|----------------|
| `agent cleanup-worktrees` | Clean stale git worktrees | ❌ No |
| `agent apply-pattern` | Apply pattern across repos | ❌ No |
| `agent team` (subgroup) | Agent Teams integration | ⚠️ Partial (`/api/teams`, `/api/tasks/{id}/escalate-to-team`) |

### Team Commands (subgroup: `agent team`)
| Command | Description | Web Equivalent |
|---------|-------------|----------------|
| `agent team start` | Launch interactive team session | ❌ No |
| `agent team escalate TASK` | Escalate failed task to team | ✅ `/api/tasks/{id}/escalate-to-team` |
| `agent team status` | List team sessions | ✅ `/api/teams` |
| `agent team handoff TEAM` | Queue team's work to autonomous pipeline | ❌ No |

---

## Current Web API Endpoints (30+ endpoints)

### Agents (6 endpoints)
- `GET /api/agents` - List all agents
- `POST /api/agents/start-all` - Start all agents
- `POST /api/agents/stop-all` - Stop all agents
- `POST /api/agents/{id}/start` - Start specific agent
- `POST /api/agents/{id}/stop` - Stop specific agent
- `POST /api/agents/{id}/restart` - Restart specific agent

### Tasks & Queues (7 endpoints)
- `GET /api/queues` - Get queue stats
- `GET /api/tasks/failed` - List failed tasks
- `GET /api/tasks/active` - List active tasks
- `POST /api/tasks` - Create new task
- `POST /api/tasks/{id}/retry` - Retry failed task
- `POST /api/tasks/{id}/cancel` - Cancel task
- `POST /api/tasks/{id}/escalate-to-team` - Escalate to team

### Operations (3 endpoints)
- `POST /api/operations/work` - Queue planning task (like `agent work`)
- `POST /api/operations/analyze` - Analyze repository
- `POST /api/operations/run-ticket` - Process JIRA ticket

### System (4 endpoints)
- `GET /api/health` - System health status
- `POST /api/system/pause` - Pause processing
- `POST /api/system/resume` - Resume processing
- `GET /api/system/status` - System status

### Setup & Config (4 endpoints)
- `POST /api/setup/validate-jira` - Validate JIRA credentials
- `POST /api/setup/validate-github` - Validate GitHub credentials
- `POST /api/setup/save-config` - Save configuration
- `GET /api/setup/status` - Get setup status

### Monitoring (6 endpoints)
- `GET /api/events` - Get recent activity events
- `GET /api/agentic-metrics` - Get agent performance metrics
- `GET /api/logs` - Get agent logs
- `GET /api/logs/{agent_id}` - Get specific agent logs
- `GET /api/logs/claude-cli` - Get Claude CLI logs
- `GET /api/logs/claude-cli/{task_id}` - Get task-specific logs

### Teams (2 endpoints)
- `GET /api/teams` - List team sessions
- `POST /api/tasks/{id}/escalate-to-team` - Escalate task to team

### Error Handling (1 endpoint)
- `POST /api/errors/translate` - Translate error messages

---

## Feature Gaps Analysis

### 🔴 CLI-only Features (No Web Equivalent)

**High Priority:**
1. ❌ **`agent init`** - Project initialization wizard
2. ❌ **`agent pull`** - Pull JIRA tickets to queue
3. ❌ **`agent summary EPIC`** - Epic progress summary
4. ❌ **`agent guide TASK --hint`** - Human guidance injection
5. ❌ **`agent clear`** - Clear queues
6. ❌ **`agent purge`** - Full workspace reset
7. ❌ **`agent team start`** - Launch interactive team session
8. ❌ **`agent team handoff`** - Handoff team work to pipeline

**Medium Priority:**
9. ❌ **`agent cleanup-worktrees`** - Git worktree management
10. ❌ **`agent apply-pattern`** - Cross-repo pattern application
11. ⚠️ **`agent up` (full feature)** - One-command startup (agents + dashboard + watchdog)
12. ⚠️ **`agent analytics` (TUI)** - Rich terminal analytics dashboard

**Low Priority:**
13. ⚠️ **`agent check --fix`** - Circuit breaker with auto-fix option

### 🔵 Web-only Features (No CLI Equivalent)

**High Priority:**
1. ❌ **Setup wizard UI** - Interactive onboarding (`/api/setup/*`)
2. ❌ **Real-time WebSocket updates** - Live agent status streaming
3. ❌ **Insights page** - Performance trends, failure analysis

**Medium Priority:**
4. ❌ **Task detail view** - Rich UI for inspecting tasks
5. ❌ **Agent card visualizations** - Current phase, progress indicators
6. ❌ **Interactive logs viewer** - Searchable, filterable logs
7. ❌ **Config editor UI** - In-browser configuration editing

---

## Consolidation Opportunities

### 🟡 Redundant Commands to Merge

1. **`agent start` vs `agent up`**
   - **Current:** Two separate commands with overlapping functionality
   - **Proposed:** Make `agent up` the primary command; keep `agent start` for backwards compat
   - **Rationale:** `agent up` is more intuitive and handles dashboard + watchdog

2. **`agent stop` vs `agent down`**
   - **Current:** Both stop agents; `down` also stops dashboard
   - **Proposed:** Merge into `agent down`; deprecate `agent stop`
   - **Rationale:** Consistent with Docker's `up`/`down` pattern

3. **`agent status` vs `agent status --watch` vs `agent dashboard` vs `agent analytics`**
   - **Current:** 4 different ways to monitor agents
   - **Proposed:** Unified monitoring commands:
     - `agent monitor` (live TUI, default)
     - `agent monitor --analytics` (analytics view)
     - `agent monitor --web` (open web dashboard)
     - `agent status` (one-time snapshot)

### 🟢 Commands to Keep As-Is

1. **`agent work`** - Core user workflow, well-designed
2. **`agent retry`** - Essential task management
3. **`agent cancel`** - Essential task management
4. **`agent doctor`** - Clear purpose, diagnostic tool
5. **`agent pause/resume`** - Simple, focused control commands

### 🔴 Commands to Consider Deprecating or Moving

1. **`agent apply-pattern`** - Very specialized, low usage
   - **Proposed:** Move to a plugin or separate tool
   - **Alternative:** Make it part of `agent team` for power users

2. **`agent cleanup-worktrees`** - Maintenance task
   - **Proposed:** Make it part of `agent doctor` or automated

3. **`agent check`** - Overlaps with `agent doctor`
   - **Proposed:** Merge into `agent doctor` as `agent doctor --circuit-breaker`

---

## Recommended CLI Structure (After Refactoring)

### Primary Commands (12)
```
agent init                    # Initialize project
agent up [--port 8080]        # Start everything (agents + dashboard + watchdog)
agent down                    # Stop everything
agent work                    # Queue new work (interactive)
agent monitor                 # Live TUI monitoring (default)
  --web                       # Open web dashboard instead
  --analytics                 # Show analytics view
agent doctor                  # Health checks & diagnostics
  --fix                       # Auto-fix issues
  --circuit-breaker           # Run circuit breaker checks
agent status                  # One-time status snapshot
agent pause                   # Pause processing
agent resume                  # Resume processing
agent purge                   # Full workspace reset
  --keep-memory
  --keep-indexes
  --keep-worktrees
```

### Task Management (subgroup: `agent task`)
```
agent task list               # List all tasks (pending, active, failed)
agent task retry TASK         # Retry failed task
agent task cancel TASK        # Cancel task
agent task guide TASK         # Inject human guidance
agent task summary EPIC       # Epic progress summary
```

### Queue Management (subgroup: `agent queue`)
```
agent queue pull              # Pull JIRA tickets
agent queue clear             # Clear queues
agent queue stats             # Show queue statistics
```

### Advanced Operations (subgroup: `agent ops`)
```
agent ops analyze REPO        # Analyze repository
agent ops run-ticket TICKET   # Process specific ticket
agent ops apply-pattern       # Cross-repo pattern application
```

### Team Commands (subgroup: `agent team`)
```
agent team start              # Launch team session
agent team escalate TASK      # Escalate task to team
agent team handoff TEAM       # Handoff team work
agent team status             # List team sessions
```

### Agent Control (subgroup: `agent agent`)
```
agent agent list              # List agents
agent agent start [AGENT]     # Start specific agent
agent agent stop [AGENT]      # Stop specific agent
agent agent restart [AGENT]   # Restart specific agent
agent agent logs [AGENT]      # View agent logs
```

---

## Implementation Roadmap

### Phase 1: CLI Refactoring (Week 1-2)
**Goal:** Reorganize CLI into logical subgroups without breaking changes

1. **Create new subgroup structure**
   - Add `agent task` subgroup
   - Add `agent queue` subgroup
   - Add `agent ops` subgroup
   - Keep backwards compatibility (e.g., `agent retry` → `agent task retry`)

2. **Consolidate monitoring commands**
   - Rename `agent status --watch` to `agent monitor`
   - Add `agent monitor --analytics` (launch analytics TUI)
   - Add `agent monitor --web` (open web dashboard)

3. **Tests & Documentation**
   - Update CLI tests for new structure
   - Update README with new command structure
   - Add deprecation warnings for old commands

**Deliverable:** PR #1 - CLI structure refactoring

---

### Phase 2: Web Feature Parity - Backend (Week 3-4)
**Goal:** Add missing CLI features to web API

#### New API Endpoints to Add:

**Queue Management:**
```python
POST /api/queue/pull          # Pull JIRA tickets
DELETE /api/queue/clear       # Clear queues
GET /api/queue/stats          # Queue statistics
```

**Task Management:**
```python
GET /api/tasks/epic/{epic_key}    # Epic summary
POST /api/tasks/{id}/guide        # Human guidance
```

**System Management:**
```python
POST /api/system/purge            # Workspace reset
  ?keep_memory=true
  ?keep_indexes=true
  ?keep_worktrees=true
POST /api/system/init             # Project initialization
```

**Team Management:**
```python
POST /api/teams/start              # Launch team session
POST /api/teams/{id}/handoff       # Handoff to autonomous
```

**Logs & Monitoring:**
```python
GET /api/logs/{agent_id}/stream    # Server-sent events for live logs
```

**Deliverable:** PR #2 - Web API parity (backend)

---

### Phase 3: Web Feature Parity - Frontend (Week 5-6)
**Goal:** Build UI components for newly added API endpoints

#### New UI Components:

1. **Queue Management Page** (`QueuePage.vue`)
   - Button: "Pull JIRA Tickets"
   - Button: "Clear Queues" (with confirmation)
   - Table: Queue statistics by agent

2. **Epic Summary View** (`EpicSummaryPage.vue`)
   - Input: Epic key
   - Table: All tickets in epic with status, PR links
   - Section: Failed tasks with error details

3. **Task Guidance Modal** (`TaskGuidanceModal.vue`)
   - Input: Human guidance text
   - Button: "Inject Guidance & Retry"
   - Display: Previous errors and retry count

4. **System Settings Page** (`SettingsPage.vue` enhancements)
   - Button: "Purge Workspace" (with checkboxes for keep options)
   - Button: "Run Health Checks" (shows results)
   - Button: "Cleanup Worktrees"

5. **Team Session UI** (`TeamsPage.vue` enhancements)
   - Button: "Launch New Team Session"
   - Modal: Team template selector (full, review, debug)
   - Button: "Handoff to Autonomous" (per team session)

6. **Live Logs Viewer** (`LogsPage.vue` enhancements)
   - Dropdown: Select agent
   - Toggle: "Auto-scroll"
   - Button: "Download logs"
   - Use Server-Sent Events for live streaming

**Deliverable:** PR #3 - Web UI parity (frontend)

---

### Phase 4: Testing & Consolidation Cleanup (Week 7)
**Goal:** Clean up deprecated commands, add integration tests

1. **Remove/deprecate redundant commands**
   - Deprecate `agent start` (keep for compat, redirect to `agent up`)
   - Deprecate `agent stop` (keep for compat, redirect to `agent down`)
   - Deprecate `agent check` (merge into `agent doctor`)

2. **Integration tests**
   - Test CLI → Web API equivalence
   - Test backwards compatibility of deprecated commands
   - Test new subgroup commands

3. **Documentation**
   - Update migration guide for users
   - Update API documentation
   - Create CLI cheat sheet

**Deliverable:** PR #4 - Cleanup & deprecations

---

## Success Metrics

### User Experience
- ✅ All CLI commands have web equivalents (and vice versa)
- ✅ CLI command count reduced from 25 to ~15 primary + subgroups
- ✅ Improved discoverability via logical grouping
- ✅ Zero breaking changes for existing users

### Technical Debt
- ✅ Reduced code duplication between CLI and web
- ✅ Consistent API patterns across all endpoints
- ✅ Comprehensive test coverage for new features
- ✅ Deprecated commands removed in v2.0.0

### Feature Completeness
- ✅ 100% CLI/Web parity for core workflows
- ✅ Real-time updates via WebSocket for all monitoring
- ✅ Power user features accessible in both CLI and web

---

## Risk Assessment

### Low Risk ✅
- Adding new API endpoints (non-breaking)
- Adding new subgroups with backwards compat
- Building new UI components

### Medium Risk ⚠️
- Deprecating commands (requires clear migration path)
- Merging `agent start`/`up` and `agent stop`/`down` (user confusion)
- WebSocket implementation for live logs (performance concerns)

### High Risk 🔴
- Removing commands entirely (wait for v2.0.0)
- Changing CLI flag names (breaking change)

---

## Open Questions

1. **Should `agent apply-pattern` be kept or moved to a plugin?**
   - Recommendation: Move to `agent ops apply-pattern` for now, consider plugin in v2.0.0

2. **How to handle `agent team start` in web UI?**
   - Recommendation: Add "Launch Team Session" button that opens a new Claude CLI window

3. **Should we support SSE (Server-Sent Events) for live logs?**
   - Recommendation: Yes, for better UX than polling

4. **Migration strategy for deprecated commands?**
   - Recommendation: Keep for 6 months with deprecation warnings, remove in v2.0.0

---

## Appendix: Full Command Mapping

| Old CLI Command | New CLI Command | Web API Endpoint |
|----------------|-----------------|------------------|
| `agent init` | `agent init` | `POST /api/system/init` |
| `agent start` | `agent up` | `POST /api/agents/start-all` |
| `agent stop` | `agent down` | `POST /api/agents/stop-all` |
| `agent restart` | `agent agent restart` | `POST /api/agents/{id}/restart` |
| `agent status` | `agent status` | `GET /api/agents` |
| `agent status --watch` | `agent monitor` | WebSocket `/api/agents` |
| `agent analytics` | `agent monitor --analytics` | `GET /api/agentic-metrics` |
| `agent dashboard` | `agent monitor --web` | N/A (server) |
| `agent work` | `agent work` | `POST /api/operations/work` |
| `agent pull` | `agent queue pull` | `POST /api/queue/pull` |
| `agent run` | `agent ops run-ticket` | `POST /api/operations/run-ticket` |
| `agent analyze` | `agent ops analyze` | `POST /api/operations/analyze` |
| `agent summary` | `agent task summary` | `GET /api/tasks/epic/{key}` |
| `agent retry` | `agent task retry` | `POST /api/tasks/{id}/retry` |
| `agent cancel` | `agent task cancel` | `POST /api/tasks/{id}/cancel` |
| `agent guide` | `agent task guide` | `POST /api/tasks/{id}/guide` |
| `agent clear` | `agent queue clear` | `DELETE /api/queue/clear` |
| `agent purge` | `agent purge` | `POST /api/system/purge` |
| `agent doctor` | `agent doctor` | `GET /api/health` |
| `agent check` | `agent doctor --circuit-breaker` | `GET /api/health` |
| `agent pause` | `agent pause` | `POST /api/system/pause` |
| `agent resume` | `agent resume` | `POST /api/system/resume` |
| `agent cleanup-worktrees` | `agent ops cleanup-worktrees` | `POST /api/system/cleanup-worktrees` |
| `agent apply-pattern` | `agent ops apply-pattern` | `POST /api/operations/apply-pattern` |
| `agent team start` | `agent team start` | `POST /api/teams/start` |
| `agent team escalate` | `agent team escalate` | `POST /api/tasks/{id}/escalate-to-team` |
| `agent team status` | `agent team status` | `GET /api/teams` |
| `agent team handoff` | `agent team handoff` | `POST /api/teams/{id}/handoff` |

---

**End of Analysis**
