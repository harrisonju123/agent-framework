# Agent Framework

Multi-agent system for autonomous software development with JIRA and GitHub integration.

![Agent Dashboard](img.png)

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Web Dashboard](#web-dashboard)
- [Workflow Modes](#workflow-modes)
- [Configuration](#configuration)
- [CLI Commands](#cli-commands)
- [Architecture](#architecture)
- [MCP Integration](#mcp-integration)
- [Safeguards](#safeguards)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## Features

- **Web Dashboard**: Real-time browser UI with agent cards, activity feed, and one-click controls
- **Interactive Mode**: Describe what you want to build, select a repo, agents handle the rest
- **Workflow Modes**: Simple (direct), Standard (with QA), or Full (architect-led) workflows
- **Agent Scaling**: Run 1-50 replicas per agent type for parallel processing
- **Rich Logging**: Context-aware logs with JIRA keys, phases, and visual indicators
- **Multi-Repository**: Work across multiple repos from a central orchestrator
- **Git Worktrees**: Isolated workspaces so agents don't interfere with your work
- **Docker Sandbox**: Isolated test execution in containers
- **MCP Integration**: Real-time JIRA/GitHub access during agent execution
- **Smart Model Selection**: Automatic Haiku/Sonnet/Opus routing based on task complexity
- **Robust Safeguards**: Circuit breaker, watchdog, retry limits, timeout handling

## Requirements

- Python 3.10+
- Node.js 18+ (for MCP servers and dashboard)
- Docker (for sandbox test execution)
- Claude CLI or Anthropic API key
- Git, JIRA API token, GitHub token

## Installation

```bash
# Clone and install
git clone https://github.com/your-org/agent-framework.git
cd agent-framework
pip install -e .

# Build MCP servers
cd mcp-servers/jira && npm install && npm run build && cd ../..
cd mcp-servers/github && npm install && npm run build && cd ../..

# Build web dashboard (optional)
cd src/agent_framework/web/frontend && npm install && npm run build && cd ../../../..

# Configure
cp config/agents.yaml.example config/agents.yaml
cp config/jira.yaml.example config/jira.yaml
cp config/github.yaml.example config/github.yaml
cp config/agent-framework.yaml.example config/agent-framework.yaml
cp config/mcp-config.json.example config/mcp-config.json
cp .env.example .env

# Edit .env with your credentials (JIRA_EMAIL, JIRA_API_TOKEN, GITHUB_TOKEN)

# Load environment
source scripts/setup-env.sh
```

## Quick Start

### Interactive Mode (Recommended)

```bash
source scripts/setup-env.sh
agent work
```

You'll be prompted for:
1. What you want to build
2. Which repository to work on
3. Workflow mode (simple/standard/full)

The system creates JIRA tickets and queues tasks for agents automatically.

### Run a Specific Ticket

```bash
# Work on a single JIRA ticket
agent run PROJ-123

# Assign to a specific agent type
agent run PROJ-123 --agent engineer
```

### Traditional Mode

```bash
# Pull from JIRA backlog
agent pull --project PROJ

# Start agents (basic)
agent start

# Start with scaling and debug logging
agent start --replicas 4 --log-level DEBUG

# Stop agents
agent stop

# Monitor via CLI
agent status --watch
```

## Web Dashboard

The web dashboard provides real-time visibility into agent operations through your browser.

### Starting the Dashboard

```bash
# Start the dashboard server
agent dashboard

# Custom port
agent dashboard --port 8080

# Allow external access
agent dashboard --host 0.0.0.0
```

Open `http://localhost:3000` in your browser.

### Dashboard Features

**Agent Cards**
- Visual status for each agent (idle, working, dead)
- Current task and progress phases
- Elapsed time on active tasks
- Hover for task details

**Controls**
- **Pause/Resume**: Temporarily halt task processing
- **Start All / Stop All**: Bulk agent management with confirmation
- **New Work**: Create tasks via goal description (opens modal)
- **Analyze Repo**: Run repository analysis (opens modal)

**Monitoring**
- **Queue Status**: Pending task counts per agent type
- **Health Checks**: System health with pass/fail indicators
- **Uptime**: Server uptime and WebSocket connection status
- **Recent Activity**: Live feed of task starts, completions, and failures
- **Failed Tasks**: List of failed tasks with retry option

**Live Logs** (collapsible)
- Real-time log streaming per agent
- Filter by agent type
- Auto-scroll with manual pause

### WebSocket Connection

The dashboard maintains a WebSocket connection for real-time updates. If disconnected, it automatically reconnects with exponential backoff (up to 10 attempts).

## Workflow Modes

Choose the right workflow based on task complexity:

| Mode | Flow | Best For |
|------|------|----------|
| **simple** | Engineer -> Done | Bug fixes, small changes |
| **standard** | Engineer -> QA -> Done | Medium features with testing |
| **full** | Architect -> Engineer -> QA -> Review | Complex features, architectural changes |

### Simple Workflow
- Engineer implements and creates PR directly
- No QA verification step
- JIRA ticket moved to Done

### Standard Workflow
- Engineer implements and commits
- QA verifies and creates PR
- Good balance of speed and quality

### Full Workflow
- Architect creates detailed implementation plan
- Engineer follows plan, creates QA task
- QA verifies, Architect reviews and creates PR
- Maximum oversight for complex changes

## Configuration

### Framework Settings (`config/agent-framework.yaml`)

```yaml
workspace: "."

llm:
  mode: claude_cli          # or "litellm" for API
  use_mcp: true             # Enable real-time JIRA/GitHub
  mcp_config_path: /path/to/config/mcp-config.json

task:
  poll_interval: 30
  max_retries: 5
  timeout: 1800

safeguards:
  max_queue_size: 100
  heartbeat_timeout: 600    # 10 min for long MCP tasks
  watchdog_interval: 60

repositories:
  - github_repo: owner/repo
    jira_project: PROJ
    display_name: My Project
```

### Agent Definitions (`config/agents.yaml`)

```yaml
agents:
  - id: engineer
    name: Software Engineer
    queue: engineer
    enabled: true
    prompt: "Your agent prompt..."
    can_commit: true
    can_create_pr: true
    jira_can_update_status: true
    jira_allowed_transitions:
      - "In Progress"
      - "Done"
```

### Environment Variables

Required in `.env`:
```
JIRA_SERVER=https://your-org.atlassian.net
JIRA_EMAIL=your@email.com
JIRA_API_TOKEN=your-token
GITHUB_TOKEN=your-github-token
```

Load before running: `source scripts/setup-env.sh`

## CLI Commands

| Command | Description |
|---------|-------------|
| `agent work` | Interactive mode - describe goal, select repo, choose workflow |
| `agent run PROJ-123` | Work on a specific JIRA ticket |
| `agent dashboard` | Start web dashboard server |
| `agent status --watch` | Live CLI dashboard |
| `agent status` | One-time status snapshot |
| `agent pull --project PROJ` | Pull JIRA backlog tickets |
| `agent start` | Start all agents (1 of each type) |
| `agent start --replicas N` | Start N replicas per agent (1-50) |
| `agent start --log-level LEVEL` | Set log level (DEBUG/INFO/WARNING/ERROR) |
| `agent analyze --repo owner/repo` | Analyze repo for issues, create JIRA epic |
| `agent analyze --repo R --focus "..."` | Focused analysis on specific areas |
| `agent stop` | Stop agents gracefully |
| `agent check` | Run safety checks |
| `agent check --fix` | Auto-fix issues |
| `agent cleanup-worktrees` | Remove stale git worktrees |

## Architecture

```
agent-framework/
├── src/agent_framework/
│   ├── cli/                # CLI commands and TUI dashboard
│   ├── core/               # Task model, orchestrator, agent loop
│   ├── queue/              # File-based queues with locking
│   ├── llm/                # Claude CLI and LiteLLM backends
│   ├── integrations/       # JIRA and GitHub clients
│   ├── safeguards/         # Circuit breaker, watchdog, retry logic
│   ├── sandbox/            # Docker test execution
│   ├── utils/              # Rich logging with context
│   ├── web/                # Web dashboard (FastAPI + Vue.js)
│   │   ├── server.py       # API endpoints and WebSocket
│   │   ├── data_provider.py # Real-time data aggregation
│   │   └── frontend/       # Vue.js SPA
│   └── workspace/          # Git operations, multi-repo manager
├── mcp-servers/
│   ├── jira/               # JIRA MCP server
│   └── github/             # GitHub MCP server
└── config/                 # YAML configuration files
```

### Agent Types

| Agent | Role |
|-------|------|
| **Software Engineer** | Implements features and bug fixes |
| **QA Engineer** | Verifies implementations and creates PRs |
| **Technical Architect** | Plans complex features, reviews PRs |
| **Product Owner** | Manages backlog and priorities |
| **Testing Agent** | Runs test suites in sandbox |
| **Static Analysis Agent** | Code quality and security scanning |
| **Code Reviewer** | Reviews pull requests |
| **Repository Analyzer** | Scans repos for issues and tech debt |

### Agent Scaling

Run multiple replicas of each agent type for parallel processing:

```bash
# Single instance of each agent (default)
agent start

# 4 replicas per agent type (32 total agents)
agent start --replicas 4

# Maximum parallelism (up to 50 per type)
agent start --replicas 50
```

**How it works:**
- All replicas share the same queue (e.g., `engineer-1`, `engineer-2` both poll `engineer` queue)
- File-based locking prevents race conditions
- Git worktrees provide isolated workspaces
- Each replica has unique logger (process-safe)

**Cost efficiency:**
- Idle agents cost $0 (no token usage)
- Only active agents consume tokens
- Scale up during high load, scale down when idle

### Multi-Repository Support

Agents run from one location but work across multiple repositories:
- Tasks contain repository context (`github_repo`, `jira_project`)
- Repos cloned to `~/.agent-workspaces/owner/repo`
- Single set of logs and queues for easy monitoring

### Repository Analysis

The `analyze` command scans repositories for issues (security, performance, code quality) and creates JIRA epics with file-grouped subtasks:

```bash
# Basic analysis - runs static analyzers and creates JIRA epic
agent analyze --repo justworkshr/pto

# Focused analysis - target specific code flows or issue types
agent analyze --repo justworkshr/pto --focus "review PTO accrual flow for tech debt"

# Focus on security in specific areas
agent analyze --repo justworkshr/pto --focus "authentication and authorization code, security vulnerabilities"

# Preview without creating JIRA tickets
agent analyze --repo justworkshr/pto --dry-run

# Combine with severity filtering
agent analyze --repo justworkshr/pto --severity medium --max-issues 30
```

### Git Worktree Support

Worktrees provide isolated workspaces so agents don't interfere with your work or each other:

```
~/.agent-workspaces/
├── owner/repo/                    # Shared clone (base repo)
└── worktrees/owner/repo/
    ├── engineer-task-abc1/        # Agent 1's isolated workspace
    └── qa-task-def2/              # Agent 2's isolated workspace
```

Enable in `config/agent-framework.yaml`:

```yaml
multi_repo:
  workspace_root: ~/.agent-workspaces
  worktree:
    enabled: true
    root: ~/.agent-workspaces/worktrees
    cleanup_on_complete: true      # Auto-remove after success
    cleanup_on_failure: false      # Keep for debugging
    max_age_hours: 24              # Auto-cleanup stale worktrees
    max_worktrees: 20              # LRU eviction when over limit
```

Cleanup orphaned worktrees:
```bash
agent cleanup-worktrees              # Remove stale worktrees
agent cleanup-worktrees --dry-run    # Preview what would be removed
agent cleanup-worktrees --force      # Remove all worktrees
```

### Enhanced Logging

Context-aware logs with JIRA keys, phases, and visual indicators:

```log
16:17:23 INFO     [engineer-1] [ME-422] Starting task: Implement rollback feature
16:17:23 INFO     [engineer-1] [analyzing] [ME-422] Phase: analyzing
16:17:23 INFO     [engineer-1] [executing_llm] [ME-422] Calling LLM (attempt 1)
16:18:45 INFO     [engineer-1] [ME-422] Tokens: 15,234 in + 3,456 out = 18,690 total (~$0.0328)
16:18:45 INFO     [engineer-1] [ME-422] Task completed in 82.3s (18,690 tokens)
```

**Log locations:**
- Console: Color output with ANSI codes (when interactive)
- Files: `logs/engineer-1.log` (plain text, no colors)

## MCP Integration

MCP (Model Context Protocol) enables real-time JIRA/GitHub access during agent execution.

### JIRA Tools

- `jira_search_issues` - JQL search
- `jira_get_issue` - Get issue details
- `jira_create_issue` - Create Story/Bug/Task
- `jira_create_epic` - Create epic
- `jira_create_subtask` - Create subtask
- `jira_transition_issue` - Change status
- `jira_add_comment` - Add comment

### GitHub Tools

- `github_create_branch` - Create branch from ref
- `github_create_pr` - Create pull request
- `github_update_pr` - Update PR title/body/state
- `github_clone_repo` - Clone to local path
- `github_commit_changes` - Stage and commit
- `github_push_branch` - Push to remote
- `github_add_pr_comment` - Comment on PR
- `github_get_pr_by_branch` - Find PR by branch
- `github_link_pr_to_jira` - Link PR to JIRA

### Configuration

```json
// config/mcp-config.json
{
  "mcpServers": {
    "jira": {
      "command": "node",
      "args": ["/path/to/mcp-servers/jira/build/index.js"],
      "env": {
        "JIRA_SERVER": "${JIRA_SERVER}",
        "JIRA_EMAIL": "${JIRA_EMAIL}",
        "JIRA_API_TOKEN": "${JIRA_API_TOKEN}"
      }
    },
    "github": {
      "command": "node",
      "args": ["/path/to/mcp-servers/github/build/index.js"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

## Safeguards

10 layers of protection:

1. **Retry Limits**: Max 5 retries per task
2. **Escalation Loop Prevention**: Escalations cannot create escalations
3. **Circuit Breaker**: 8 safety checks on queue health
4. **Circular Dependency Detection**: Prevents deadlocks
5. **Task Timeout**: 30 min default with configurable LLM timeout
6. **Watchdog**: Auto-restarts crashed agents
7. **Graceful Shutdown**: SIGTERM before SIGKILL
8. **Stale Lock Recovery**: Removes locks from dead processes
9. **Queue Size Limits**: 100 tasks per agent
10. **Task Age Limits**: Archive tasks >7 days old

Run checks: `agent check` or `agent check --fix`

## Documentation

Detailed technical documentation is available in the `docs/` directory:

- **[MCP Architecture](docs/MCP_ARCHITECTURE.md)** - Model Context Protocol integration details
- **[MCP Setup](docs/MCP_SETUP.md)** - Setting up JIRA and GitHub MCP servers
- **[Code Review Workflow](docs/CODE_REVIEW_WORKFLOW.md)** - How PRs are queued and picked up by reviewers

## Troubleshooting

### Environment variables not loaded
```bash
# Check if set
echo $JIRA_EMAIL

# Load them
source scripts/setup-env.sh
```

### Stale locks blocking tasks
```bash
# Check locks
ls -la .agent-communication/locks/

# Watchdog auto-removes stale locks, or manually:
rm -rf .agent-communication/locks/<task-id>.lock
```

### Agents not responding
```bash
# Check processes
ps aux | grep run_agent

# Check heartbeats
ls -la .agent-communication/heartbeats/

# Restart
agent stop && agent start
```

### MCP config not found
Ensure `mcp_config_path` in `agent-framework.yaml` is an absolute path and the file exists.

### Dashboard not loading
```bash
# Rebuild frontend
cd src/agent_framework/web/frontend
npm install && npm run build

# Check if port is in use
lsof -i :3000
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
pytest --cov=agent_framework

# Type checking
mypy src/

# Linting
ruff check src/
ruff format src/

# Frontend development
cd src/agent_framework/web/frontend
npm run dev  # Hot-reload dev server
```

## License

MIT
