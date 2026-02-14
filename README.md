# Agent Framework

Multi-agent system for autonomous software development. Agents plan, implement, test, review, and ship code — with zero human intervention. Works with or without JIRA.

![Agent Dashboard](img.png)

## How It Works

```
You describe what to build
  |
  v
Architect plans, routes, and breaks down work
  |
  v
Engineer implements --> QA verifies + reviews --> PRs created
```

Three self-sufficient agents form a complete autonomous engineering team. Fewer agents = fewer queue hops = faster execution = more autonomy.

With **Team Mode** enabled, each ticket runs as a single Claude Agent Teams session — the full architect/engineer/QA workflow happens in one pass instead of bouncing between queues.

## Quick Start

```bash
pip install -e .

# Set up your config
cp config/agent-framework.yaml.example config/agent-framework.yaml
```

Edit `config/agent-framework.yaml` and add your repositories (see [Configuration](#configuration) below), then:

```bash
# Build MCP servers (JIRA + GitHub access during execution)
cd mcp-servers/jira && npm install && npm run build && cd ../..
cd mcp-servers/github && npm install && npm run build && cd ../..

# Configure credentials (optional — only needed for JIRA/GitHub MCP tools)
cp .env.example .env   # Edit with JIRA_SERVER, JIRA_EMAIL, JIRA_API_TOKEN, GITHUB_TOKEN

# Validate setup
agent doctor
```

### Describe What to Build

```bash
agent start
agent work
# > What would you like to work on? Add retry logic to the payment webhook handler
# > Which repository? your-org/your-repo
```

The framework queues a planning task to the Architect, who breaks it down, chains to Engineer for implementation, then QA for review — all locally via file queues.

### Process a JIRA Epic (optional)

If your repo has a `jira_project` configured:

```bash
agent work --epic PROJ-100 --parallel
agent summary --epic PROJ-100
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

## Workflow Checkpoints

Add configurable pause points for human approval at high-stakes workflow steps:

```yaml
workflows:
  production-deploy:
    steps:
      engineer:
        agent: engineer
        checkpoint:
          message: "Review implementation before production deployment"
          reason: "High-risk production changes require approval"
        next:
          - target: qa
```

```bash
# List tasks awaiting approval
agent approve

# Approve a checkpoint to continue workflow
agent approve chain-abc123-engineer -m "Reviewed, looks good"
```

See [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md) for full documentation.

## CLI Reference

### Core

| Command | Description |
|---------|-------------|
| `agent work` | Interactive: describe goal, pick repo, choose workflow |
| `agent work --epic PROJ-100` | Process all tickets in a JIRA epic |
| `agent work --epic PROJ-100 --parallel` | Process epic tickets in parallel worktrees |
| `agent run PROJ-123` | Work on a single JIRA ticket |
| `agent pull --project PROJ` | Pull unassigned tickets from JIRA backlog |
| `agent approve` | List tasks awaiting checkpoint approval |
| `agent approve <task-id>` | Approve a checkpoint to continue workflow |

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

This file is **not tracked by git** — it contains your repo list and local settings. Copy the example to get started:

```bash
cp config/agent-framework.yaml.example config/agent-framework.yaml
```

The key section is `repositories` — add the repos you want agents to work on:

```yaml
repositories:
  # Local-only repo (no JIRA)
  - github_repo: your-org/your-app
    display_name: Your App

  # Repo with JIRA integration
  - github_repo: your-org/another-repo
    jira_project: PROJ
    display_name: Another Repo
```

`jira_project` is optional. Repos without it use local-only task tracking — tasks flow through the same architect → engineer → QA pipeline, just without JIRA ticket creation or status sync.

Other settings you may want to adjust:

```yaml
llm:
  mode: claude_cli                          # or "litellm" for API calls
  claude_cli_default_model: claude-sonnet-4-5-20250929
  use_mcp: true                             # JIRA/GitHub access during execution
  mcp_config_path: config/mcp-config.json

team_mode:
  enabled: true       # Multi-agent Claude Teams sessions

multi_repo:
  workspace_root: ~/.agent-workspaces
  worktree:
    enabled: true     # Isolated git worktrees per task
```

See `config/agent-framework.yaml.example` for all available options.

### `config/agents.yaml`

Defines agent roles, prompts, and permissions. Each agent has a queue, prompt, and JIRA/GitHub permissions. See `config/agents.yaml` for the full definitions.

### `.env` (optional)

Only needed if you're using JIRA or GitHub MCP integrations:

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
| Missing config | `cp config/agent-framework.yaml.example config/agent-framework.yaml` |
| Tasks stuck as in_progress | Agent auto-recovers orphaned tasks on next poll cycle |
| Stale locks | Watchdog auto-cleans, or `rm -rf .agent-communication/locks/*.lock` |
| Agents not responding | `agent stop && agent start` |
| JIRA auth failed | New token at https://id.atlassian.com/manage-profile/security/api-tokens |
| GitHub auth failed | New token at https://github.com/settings/tokens (needs `repo` scope) |
| MCP tool name conflict | Framework uses `--strict-mcp-config` to avoid global config collisions |

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
