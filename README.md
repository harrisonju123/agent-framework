# Agent Framework

Multi-agent system for autonomous software development. Agents plan, implement, test, review, and ship code from JIRA tickets — with zero human intervention.

![Agent Dashboard](img.png)

## How It Works

```
You describe what to build (or point at a JIRA epic)
  |
  v
Architect plans, routes, and breaks down work
  |
  v
Engineer implements --> QA verifies + reviews --> PRs created
  |
  v
JIRA updated, ready for merge
```

Three self-sufficient agents form a complete autonomous engineering team. Fewer agents = fewer queue hops = faster execution = more autonomy.

With **Team Mode** enabled, each ticket runs as a single Claude Agent Teams session — the full architect/engineer/QA workflow happens in one pass instead of bouncing between queues.

## Quick Start

```bash
pip install -e .

# Build MCP servers (JIRA + GitHub access during execution)
cd mcp-servers/jira && npm install && npm run build && cd ../..
cd mcp-servers/github && npm install && npm run build && cd ../..

# Configure credentials
cp .env.example .env   # Edit with JIRA_SERVER, JIRA_EMAIL, JIRA_API_TOKEN, GITHUB_TOKEN

# Validate setup
agent doctor
```

### Process a JIRA Epic

```bash
# Start agents
agent start --replicas 3

# Fire off an epic — parallel worktrees, each ticket gets its own team
agent work --epic ME-443 --parallel -w standard --no-dashboard

# Monitor progress
agent summary --epic ME-443
agent status --watch
```

### Describe What to Build

```bash
agent work
# > What would you like to work on? Add retry logic to the payment webhook handler
# > Which repository? justworkshr/pto
# > Workflow? standard
```

### Work a Single Ticket

```bash
agent run PROJ-123
agent run PROJ-123 --agent engineer
```

## Team Mode

Team mode uses [Claude Agent Teams](https://docs.anthropic.com/en/docs/claude-code) to run multi-agent workflows in a single session. Instead of serial queue hops (engineer finishes → QA picks up → architect reviews), a team session handles the full workflow atomically.

### Autonomous Teams (Pipeline)

Enable in `config/agent-framework.yaml`:

```yaml
team_mode:
  enabled: true
  min_workflow: standard   # standard and above trigger teams
```

| Workflow | Lead | Teammates | Behavior |
|----------|------|-----------|----------|
| `simple` | engineer | none | Single-agent, no team |
| `standard` | engineer | QA | Engineer implements, QA validates in same session |
| `full` | architect | engineer, QA | Architect plans, engineer implements, QA validates |

Teammate prompts come from `config/agents.yaml` — no duplication.

### Interactive Teams

For hands-on work where you want to steer the team:

```bash
# Full team on a repo
agent team start --template full --repo justworkshr/pto

# Full team with epic context loaded
agent team start --template full --repo justworkshr/pto --epic ME-443

# Code review team (2 specialized reviewers)
agent team start --template review --repo justworkshr/pto

# Debug a failed task
agent team escalate task-impl-1234
```

Templates: `full` (Architect + Engineer + QA), `review` (QA + Security + Performance), `debug` (2 investigators).

## Workflow Modes

| Mode | Flow | Best For |
|------|------|----------|
| **simple** | Architect → Engineer | Bug fixes, small changes |
| **standard** | Architect → Engineer → QA | Features needing verification |
| **full** | Architect → Engineer → QA → Architect | Complex features |

### Failure Loop

When QA finds issues, tasks loop back to Engineer automatically:

```
QA finds issues → Engineer fixes → QA re-verifies
After 5 retries → escalate to Architect for replanning
```

## CLI Reference

### Core

| Command | Description |
|---------|-------------|
| `agent work` | Interactive: describe goal, pick repo, choose workflow |
| `agent work --epic PROJ-100` | Process all tickets in a JIRA epic |
| `agent work --epic PROJ-100 --parallel` | Process epic tickets in parallel worktrees |
| `agent run PROJ-123` | Work on a single JIRA ticket |
| `agent pull --project PROJ` | Pull unassigned tickets from JIRA backlog |

### Operations

| Command | Description |
|---------|-------------|
| `agent start` | Start all agents |
| `agent start --replicas 4` | Start N replicas per agent type (parallel processing) |
| `agent stop` | Stop agents gracefully |
| `agent pause` / `agent resume` | Pause/resume task processing |
| `agent status --watch` | Live terminal dashboard |
| `agent summary --epic PROJ-100` | Epic progress with PR links and errors |
| `agent retry PROJ-104` | Retry a failed task |
| `agent retry --all` | Retry all failed tasks |

### Teams

| Command | Description |
|---------|-------------|
| `agent team start -t full -r owner/repo` | Launch interactive team session |
| `agent team start -t full -r owner/repo -e PROJ-100` | Team session with epic context |
| `agent team escalate TASK-ID` | Debug a failed task with a team |
| `agent team status` | List active team sessions |
| `agent team handoff TEAM-NAME` | Push team output to autonomous pipeline |

### Analysis & Health

| Command | Description |
|---------|-------------|
| `agent doctor` | Validate configuration and connectivity |
| `agent analyze --repo owner/repo` | Scan repo for issues, create JIRA epic |
| `agent analyze --repo R --focus "auth flow"` | Focused analysis on specific areas |
| `agent check --fix` | Run circuit breaker checks, auto-fix |
| `agent dashboard` | Web dashboard with setup wizard |
| `agent cleanup-worktrees` | Remove stale git worktrees |

## Configuration

### `config/agent-framework.yaml`

```yaml
llm:
  mode: claude_cli
  use_mcp: true
  mcp_config_path: /path/to/config/mcp-config.json

team_mode:
  enabled: true          # Use Agent Teams for multi-agent workflows
  min_workflow: standard  # simple = never, standard = eng+QA, full = arch+eng+QA

task:
  poll_interval: 30
  max_retries: 5

multi_repo:
  workspace_root: ~/.agent-workspaces
  worktree:
    enabled: true        # Isolated branches per task
    cleanup_on_complete: true

repositories:
  - github_repo: owner/repo
    jira_project: PROJ
```

### `config/agents.yaml`

Defines agent roles, prompts, and permissions. Each agent has a queue, prompt, and JIRA/GitHub permissions. See `config/agents.yaml` for the full definitions.

### `.env`

```
JIRA_SERVER=https://your-org.atlassian.net
JIRA_EMAIL=your@email.com
JIRA_API_TOKEN=your-token
GITHUB_TOKEN=your-github-token
```

## Architecture

```
src/agent_framework/
├── cli/           CLI commands, TUI dashboard, team commands
├── core/          Agent loop, orchestrator, task model, team composer
├── llm/           Claude CLI backend, LiteLLM backend, model selection
├── queue/         File-based task queues with locking
├── integrations/  JIRA and GitHub clients
├── safeguards/    Circuit breaker, watchdog, retry/escalation
├── workspace/     Multi-repo manager, git worktrees
├── web/           Web dashboard (FastAPI + Vue.js)
└── utils/         Logging, validation, subprocess, error handling
```

### Agent Types

| Agent | Role |
|-------|------|
| **Architect** | Plans, routes, analyzes repos, breaks down work, creates tickets, reviews architecture, creates PRs (full workflow) |
| **Engineer** | Implements code, writes tests, creates PRs (simple workflow), self-heals via test-runner teammate |
| **QA** | Single quality gate — linting, testing, security scanning, code review, PR creation (standard workflow) |

### MCP Integration

Agents access JIRA and GitHub in real-time during execution via MCP servers:

- **JIRA**: Search, create, transition, comment on issues and epics
- **GitHub**: Create branches, commit, push, create PRs, add comments

### Safeguards

1. Retry limits (max 5 per task)
2. Escalation loop prevention
3. Circuit breaker (8 health checks)
4. Task timeouts (per task type)
5. Watchdog auto-restart
6. Graceful shutdown (SIGTERM then SIGKILL)
7. Stale lock recovery
8. Queue size limits
9. Task age limits
10. Worktree safety (won't delete unpushed work)

## Troubleshooting

```bash
agent doctor              # Check config, credentials, connectivity
agent check --fix         # Run safety checks and auto-fix
agent cleanup-worktrees   # Remove stale worktrees
```

| Problem | Fix |
|---------|-----|
| JIRA auth failed | New token at https://id.atlassian.com/manage-profile/security/api-tokens |
| GitHub auth failed | New token at https://github.com/settings/tokens (needs `repo` scope) |
| Stale locks | Watchdog auto-cleans, or `rm -rf .agent-communication/locks/*.lock` |
| Agents not responding | `agent stop && agent start` |
| MCP tool name conflict | Framework uses `--strict-mcp-config` to avoid global config collisions |

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
