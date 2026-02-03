# Agent Framework

Open source AI agent framework with JIRA and GitHub integration. Configurable multi-agent system for autonomous task execution with robust safeguards.

## Features

- **Interactive Mode**: Describe what you want to build, select a repo, and let agents handle the rest
- **Real-Time Activity Dashboard**: Live TUI showing agent status, current tasks, progress phases, and recent activity
- **Multi-Repository Support**: Work across multiple repositories from a central orchestrator
- **Intelligent Epic Creation**: Product Owner agent analyzes codebases and creates smart JIRA breakdowns
- Pull unassigned JIRA tickets from backlog
- Process tickets using Claude (LiteLLM API or Claude CLI subprocess)
- Automatic git workflows: branch creation, commits, PR opening
- JIRA integration: status transitions, commenting, ticket linking, epic/subtask creation
- Configurable agent roles via YAML
- File-based task queues with atomic locking
- Automatic recovery from failures with watchdog
- Circuit breaker with 8 safety checks
- Cost optimization with automatic model selection (Haiku/Sonnet/Opus)

## Requirements

- Python 3.10+
- **Node.js 18+** (for MCP servers)
- Claude CLI (for `claude_cli` mode) OR Anthropic API key (for `litellm` mode)
- Git
- JIRA account with API token
- GitHub account with personal access token

## Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/your-org/agent-framework.git
cd agent-framework

# Install Python package
pip install -e .

# Install MCP servers (for real-time JIRA/GitHub integration)
cd mcp-servers/jira
npm install
npm run build

cd ../github
npm install
npm run build
cd ../..
```

### Setup

```bash
# Initialize project
agent init

# Copy and configure settings
cp config/agents.yaml.example config/agents.yaml
cp config/jira.yaml.example config/jira.yaml
cp config/github.yaml.example config/github.yaml
cp config/agent-framework.yaml.example config/agent-framework.yaml
cp .env.example .env

# Copy MCP configuration (optional, for real-time integration)
cp config/mcp-config.json.example config/mcp-config.json

# Edit .env with your credentials
# - JIRA_EMAIL
# - JIRA_API_TOKEN
# - GITHUB_TOKEN

# Load environment variables
source scripts/setup-env.sh
```

**Important:** The configuration files use environment variable substitution (`${JIRA_EMAIL}`, `${JIRA_API_TOKEN}`, etc.). You must load the `.env` file into your shell environment before running commands:

```bash
# Option 1: Source setup script before each command
source scripts/setup-env.sh
agent run PROJ-123

# Option 2: Load .env directly (one-time)
set -a && source .env && set +a
agent run PROJ-123

# Option 3: Add to your shell profile for automatic loading
# Add to ~/.zshrc or ~/.bashrc:
if [ -f ~/path/to/agent-framework/scripts/setup-env.sh ]; then
    source ~/path/to/agent-framework/scripts/setup-env.sh
fi
```

### Basic Usage

#### Interactive Mode (Recommended)

The easiest way to get started - describe what you want to build and let the agents handle the rest:

```bash
# Load environment variables (required before running commands)
source scripts/setup-env.sh

# Start interactive mode (launches live dashboard)
agent work

# Skip dashboard if you prefer logs
agent work --no-dashboard

# You'll be prompted:
# 1. What would you like to work on?
#    > Add multi-factor authentication to login flow
#
# 2. Which repository?
#    1. justworkshr/pto (PTO project)
#    2. justworkshr/international (INTL project)
#    > 1
#
# The system will:
# - Create a planning task for the Product Owner agent
# - Product Owner will clone/analyze the codebase
# - Create a JIRA epic with intelligent subtask breakdown
# - Queue tasks for architect, engineer, and QA agents
# - All work happens autonomously in the background
```

**What Happens After `agent work`:**

1. **Product Owner** analyzes your goal and the target codebase
2. Creates a JIRA epic in the appropriate project (e.g., PTO-1234)
3. Breaks down into subtasks based on complexity:
   - Simple fixes: Engineer → QA
   - Medium features: Architect → Engineer → QA
   - Complex features: Architect → Multiple Engineers → QA
4. Other agents pick up tasks and execute autonomously
5. Work happens in `~/.agent-workspaces/owner/repo` (not your framework directory)

**Live Dashboard:**

After queuing your task, `agent work` automatically launches a live dashboard showing:
- Real-time agent status (Idle/Working/Dead)
- Current task and phase (e.g., "Executing LLM", "Creating PR")
- Elapsed time for current tasks
- Recent completions and failures
- Queue statistics

Press Ctrl+C to exit the dashboard (agents continue running).

#### Traditional Ticket Mode

Work with existing JIRA tickets:

```bash
# Load environment variables (required before running commands)
source scripts/setup-env.sh

# Pull tickets from JIRA backlog (assigns to architect for planning)
agent pull --project PROJ

# Work on specific ticket (assigns to architect by default)
agent run PROJ-123

# Assign to specific agent (skip planning phase)
agent run PROJ-123 --agent engineer

# Start all agents with watchdog
agent start

# Check system status
agent status

# Run safety checks
agent check

# Stop agents gracefully
agent stop
```

**Workflow Notes:**
- By default, `agent run` and `agent pull` assign tickets to the **architect** for planning
- The architect creates a task for the engineer with `depends_on` set
- The engineer's task is blocked until the architect completes planning
- Use `--agent engineer` to skip planning and go directly to implementation

## Multi-Repository Support

The framework supports working across multiple repositories from a central location. This is ideal for microservices architectures or managing work across multiple projects.

### Architecture: Central Orchestrator with Multi-Repo Task Context

**How It Works:**
- Agents run from **ONE** location (your agent-framework directory)
- All communication happens in one `.agent-communication/` directory
- Tasks contain repository context (`github_repo`, `jira_project`)
- Agents use `MultiRepoManager` to clone/work in target repos as needed
- Target repos are cloned to `~/.agent-workspaces/owner/repo`

**Benefits:**
- ✅ Single set of logs, queues, locks for easy monitoring
- ✅ No conflicts from multiple agent instances
- ✅ Doesn't pollute target repos with framework artifacts
- ✅ Portable config using `owner/repo` format
- ✅ Can work on multiple repos simultaneously

**Configuration:**

1. Register repositories in `config/agent-framework.yaml`:

```yaml
repositories:
  - github_repo: justworkshr/pto
    jira_project: PTO
    display_name: PTO Service

  - github_repo: justworkshr/api-gateway
    jira_project: API
    display_name: API Gateway
```

2. Use interactive mode to work on any registered repo:

```bash
agent work
# Select which repo to work on
# Agents automatically clone and work in the right place
```

### Multi-Repo Pattern Application

Apply code patterns from one repo to multiple others:

```bash
agent apply-pattern \
  --reference justworkshr/pto \
  --files "src/auth/mfa.go,src/auth/mfa_test.go" \
  --targets "justworkshr/international,justworkshr/api-gateway" \
  --description "Add MFA support to login endpoints"
```

This will:
1. Clone all target repos to `~/.agent-workspaces/`
2. Read reference implementation from PTO repo
3. Use Claude to implement the same pattern in each target
4. Create PRs in each target repository

## Architecture

### Project Structure

```
agent-framework/
├── src/agent_framework/
│   ├── cli/                # Click CLI commands
│   ├── core/               # Task model, agent loop, orchestrator
│   ├── queue/              # File-based queue with mkdir locks
│   ├── llm/                # LiteLLM + Claude CLI backends
│   ├── integrations/       # JIRA + GitHub clients
│   ├── safeguards/         # Retry, escalation, watchdog, circuit breaker
│   └── workspace/          # Git operations
├── config/
│   ├── agents.yaml         # Agent definitions
│   ├── jira.yaml           # JIRA connection
│   ├── github.yaml         # GitHub settings
│   └── agent-framework.yaml # Main framework config
└── tests/
    ├── unit/               # Unit tests
    └── integration/        # Integration tests
```

### Workflow

The framework implements a multi-agent workflow with automatic task handoff:

1. **Planning Phase (Architect)**
   - `agent pull` or `agent run` assigns tickets to **architect** by default
   - Architect reviews ticket → creates detailed implementation plan
   - Architect creates follow-up task for engineer with plan in description
   - Architect task is marked complete

2. **Implementation Phase (Engineer)**
   - Engineer's task has `depends_on: [architect_task_id]`
   - Queue automatically waits until architect completes planning
   - Engineer picks up task → reviews architect's plan
   - Engineer implements following the plan → commits changes
   - Engineer creates follow-up task for QA

3. **Verification Phase (QA)**
   - QA task has `depends_on: [engineer_task_id]`
   - QA picks up task → runs tests and verifies implementation
   - If tests pass: creates PR request task for architect
   - If tests fail: creates fix task for engineer

4. **Review Phase (Architect)**
   - Architect reviews PR → approves or requests changes
   - Architect creates git branch → pushes to remote
   - Architect opens PR with JIRA link
   - Architect transitions JIRA to "Code Review"

**Key Features:**
- `depends_on` field enforces task order (architect → engineer → QA)
- Tasks are blocked until dependencies complete
- Each agent creates follow-up tasks for the next phase
- Full audit trail in task JSON files

## Configuration

### Agent Definitions (config/agents.yaml)

```yaml
agents:
  - id: engineer
    name: Software Engineer
    queue: engineer
    enabled: true
    prompt: "Implement features following existing patterns. Write tests."
    jira_can_update_status: true
    can_commit: true
```

### JIRA Integration (config/jira.yaml)

```yaml
jira:
  server: https://your-org.atlassian.net
  email: ${JIRA_EMAIL}
  api_token: ${JIRA_API_TOKEN}
  project: PROJ
  backlog_filter: "status = 'To Do' AND assignee is EMPTY"
  transitions:
    in_progress: "21"
    code_review: "31"
    done: "41"
```

### GitHub Integration (config/github.yaml)

```yaml
github:
  token: ${GITHUB_TOKEN}
  owner: your-org
  repo: your-repo
  branch_pattern: "feature/{ticket_id}-{slug}"
  pr_title_pattern: "[{ticket_id}] {title}"
```

### Framework Settings (config/agent-framework.yaml)

```yaml
workspace: "."
llm:
  mode: claude_cli  # or "litellm"
  claude_cli_executable: claude
task:
  poll_interval: 30
  max_retries: 5
  timeout: 1800
safeguards:
  max_queue_size: 100
  max_escalations: 50
  heartbeat_timeout: 90
  watchdog_interval: 60

# Multi-Repository Configuration
multi_repo:
  workspace_root: "~/.agent-workspaces"  # Where to clone repos

# Repository Registry (for interactive mode)
repositories:
  - github_repo: justworkshr/pto
    jira_project: PTO
    display_name: PTO Service

  - github_repo: justworkshr/international
    jira_project: INTL
    display_name: International Service
```

## LLM Backends

### Claude CLI (Default)

Uses Claude Code subprocess for Max subscription users:

```yaml
llm:
  mode: claude_cli
  claude_cli_executable: claude
  claude_cli_cheap_model: haiku
  claude_cli_default_model: sonnet
  claude_cli_premium_model: opus
```

### LiteLLM (API)

Uses Anthropic API via LiteLLM:

```yaml
llm:
  mode: litellm
  litellm_api_key: ${ANTHROPIC_API_KEY}
  litellm_cheap_model: claude-3-5-haiku-20241022
  litellm_default_model: claude-sonnet-4-20250514
  litellm_premium_model: claude-opus-4-20250514
```

### Model Selection

Automatic model routing based on task type and retry count:

- **Haiku** (cheap): testing, verification, fixes, docs
- **Sonnet** (default): implementation, architecture, planning
- **Opus** (premium): escalations, tasks with 3+ retries

## MCP Integration

The framework supports MCP (Model Context Protocol) for real-time JIRA and GitHub integration during agent execution.

### What is MCP?

MCP enables agents to interact with JIRA and GitHub **during** task execution, not just after. This provides:

- **Real-time access** - Query issues, create tickets, manage PRs while working
- **Transparent workflows** - All operations visible in agent conversations
- **Better error handling** - Agents can see and respond to integration errors
- **Flexible workflows** - Agents decide when to use integrations, not the framework

### Enabling MCP

**Requirements:**
- Node.js 18+ installed
- Claude CLI mode (`mode: claude_cli`)
- MCP servers built (see Installation)

**Configuration:**

```yaml
# config/agent-framework.yaml
llm:
  mode: claude_cli  # Required for MCP
  use_mcp: true
  mcp_config_path: ${PWD}/config/mcp-config.json
```

The MCP configuration (`config/mcp-config.json`) defines JIRA and GitHub MCP servers:

```json
{
  "mcpServers": {
    "jira": {
      "command": "node",
      "args": ["${PWD}/mcp-servers/jira/build/index.js"],
      "env": {
        "JIRA_SERVER": "${JIRA_SERVER}",
        "JIRA_EMAIL": "${JIRA_EMAIL}",
        "JIRA_API_TOKEN": "${JIRA_API_TOKEN}"
      }
    },
    "github": {
      "command": "node",
      "args": ["${PWD}/mcp-servers/github/build/index.js"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

### Available MCP Tools

**JIRA Tools:**
- `jira_search_issues` - Search with JQL
- `jira_get_issue` - Get issue details
- `jira_create_issue` - Create Story/Bug/Task
- `jira_create_epic` - Create epic
- `jira_create_subtask` - Create subtask
- `jira_transition_issue` - Change status
- `jira_add_comment` - Add comment
- `jira_create_epic_with_subtasks` - Batch create

**GitHub Tools:**
- `github_create_branch` - Create branch
- `github_create_pr` - Create pull request
- `github_add_pr_comment` - Comment on PR
- `github_get_pr_by_branch` - Find PR by branch
- `github_link_pr_to_jira` - Link PR to JIRA ticket

### Legacy Mode

Without MCPs (default), the framework uses a post-LLM workflow where git/GitHub/JIRA operations happen after agent completion. Set `use_mcp: false` to use this mode.

### Documentation

- **Setup Guide:** [docs/MCP_SETUP.md](docs/MCP_SETUP.md)
- **Architecture:** [docs/MCP_ARCHITECTURE.md](docs/MCP_ARCHITECTURE.md)
- **JIRA MCP:** [mcp-servers/jira/README.md](mcp-servers/jira/README.md)
- **GitHub MCP:** [mcp-servers/github/README.md](mcp-servers/github/README.md)

## Safeguards

Ported from the original Bash implementation with 10 layers of safety:

1. **Retry Limits**: Max 5 retries per task
2. **Escalation Loop Prevention**: Escalations CANNOT create more escalations
3. **Circuit Breaker**: 8 safety checks (queue size, creation rate, etc.)
4. **Circular Dependency Detection**: Prevents Task A → B → A deadlocks
5. **Task Timeout**: Default 30 min per task
6. **Watchdog Auto-Recovery**: Restarts crashed agents
7. **Stale Lock Recovery**: Removes locks from dead agents
8. **Queue Size Limits**: 100 tasks per agent max
9. **Escalation Count Limits**: 50 escalations max
10. **Task Age Limits**: Archive tasks >7 days old

### Circuit Breaker

Run safety checks manually:

```bash
# Run all checks
agent check

# Auto-fix detected issues
agent check --fix
```

## Orchestrator

### Starting Agents

```bash
# Start all enabled agents with watchdog
agent start

# Start without watchdog
agent start --no-watchdog
```

### Monitoring

```bash
# Live dashboard with auto-refresh (recommended)
agent status --watch

# One-time snapshot
agent status

# View logs
tail -f logs/engineer.log
tail -f logs/watchdog.log
```

**Dashboard Features:**
- 🔄 Real-time agent status (Working/Idle/Dead)
- ⏱️ Task elapsed time and progress phases
- ✓ Recent completions with duration
- ✗ Recent failures with retry counts
- 📊 Queue statistics per agent
- ↻ Auto-refreshes every 2 seconds

### Stopping Agents

```bash
# Graceful shutdown (SIGTERM)
agent stop

# Force shutdown (SIGKILL)
agent stop --force
```

## Live Activity Dashboard

The framework includes a real-time TUI dashboard for monitoring agent activity.

### Features

- **Agent Status Table**: Shows all agents with current status, activity, and elapsed time
- **Recent Activity**: Last 5 events (starts, completions, failures) with timestamps and durations
- **Queue Statistics**: Pending task counts per agent
- **Auto-Refresh**: Updates every 2 seconds
- **Phase Tracking**: See detailed progress phases:
  - Analyzing, Planning
  - Executing LLM (when Claude is processing)
  - Implementing, Testing
  - Committing, Creating PR, Updating JIRA

### Usage

**Automatic Launch (Recommended):**
```bash
agent work
# Dashboard appears automatically after queuing task
```

**Manual Launch:**
```bash
# Watch mode with live updates
agent status --watch

# One-time snapshot
agent status
```

**Dashboard Display:**
```
╭─────────────────────── Agent Activity Dashboard ────────────────────────╮
│ 🤖 Agent Activity Dashboard • Updates every 2s • Uptime: 2m 15s         │
├──────────────────────────────────────────────────────────────────────────┤
│ ┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓  │
│ ┃ Agent         ┃ Status     ┃ Current Activity      ┃ Elapsed      ┃  │
│ ┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩  │
│ │ Product Owner │ 🔄 Working │ Executing Llm         │ 2m 34s       │  │
│ │ Engineer      │ ⏸  Idle    │ Waiting for tasks     │ -            │  │
│ └───────────────┴────────────┴───────────────────────┴──────────────┘  │
│                                                                          │
│ Recent Activity:                                                        │
│ ✓ 14:28:30 - product-owner completed: Create epic (3m 22s)            │
│ ✗ 14:23:10 - engineer failed: Fix auth bug (retry 2/5)                │
╰──────────────────────────────────────────────────────────────────────────╯
```

**Status Indicators:**
- ⏸ **Idle** (yellow) - Agent waiting for tasks
- 🔄 **Working** (green) - Agent processing a task
- ❌ **Dead** (red) - Agent not responding (watchdog will restart)

**Keyboard:**
- `Ctrl+C` - Exit dashboard (agents keep running)

See [DASHBOARD_QUICKSTART.md](DASHBOARD_QUICKSTART.md) for detailed guide.

## Development

### Install Dev Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
pytest -v
pytest --cov=agent_framework
```

### Type Checking

```bash
mypy src/
```

### Linting

```bash
ruff check src/
ruff format src/
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `agent work` | **Interactive mode with live dashboard**: Describe goal, select repo |
| `agent work --no-dashboard` | Interactive mode without dashboard |
| `agent status --watch` | **Live dashboard**: Real-time agent activity monitor |
| `agent status` | One-time snapshot of agent status |
| `agent init` | Initialize project with config templates |
| `agent pull --project PROJ` | Pull unassigned JIRA tickets |
| `agent run PROJ-123` | Work on specific ticket |
| `agent start` | Start all agents + watchdog |
| `agent stop` | Stop all agents gracefully |
| `agent check` | Run circuit breaker safety checks |
| `agent check --fix` | Auto-fix detected issues |
| `agent apply-pattern` | Apply code patterns across multiple repos |

## Troubleshooting

### JIRA authentication failed (HTTP 404)?

If you get `JiraError HTTP 404` or `X-Seraph-Loginreason: AUTHENTICATED_FAILED`, your environment variables aren't loaded:

```bash
# Check if variables are set
echo $JIRA_EMAIL
echo $JIRA_API_TOKEN

# If empty, load them
source scripts/setup-env.sh

# Or load .env directly
set -a && source .env && set +a

# Try again
agent run PROJ-123
```

### Agent keeps finding task but not processing it?

This usually means a **stale lock** from a crashed agent. The system has automatic stale lock detection that checks if the lock's PID is still alive and removes it if not.

```bash
# Check for stale locks
ls -la .agent-communication/locks/

# The agent will automatically remove stale locks after detecting them
# Check logs for "Removing stale lock" messages
tail -f logs/*.log

# If a lock is stuck, you can manually remove it
rm -rf .agent-communication/locks/<task-id>.lock

# Always stop agents gracefully to avoid stale locks
agent stop  # uses SIGTERM for graceful shutdown
```

### Agents not processing tasks?

```bash
# Check if agents are running
ps aux | grep run_agent

# Check heartbeats
ls -la .agent-communication/heartbeats/

# Check logs
tail -f logs/*.log
```

### Dashboard not showing agent activity?

```bash
# Check if agents are running
ps aux | grep run_agent

# Check activity files
ls -la .agent-communication/activity/

# View activity directly
cat .agent-communication/activity/engineer.json | jq .

# Restart agents
agent stop && agent start
```

### Too many escalations?

```bash
# Run circuit breaker checks
agent check

# Review escalated tasks
find .agent-communication/queues -name "*escalation*.json"
```

### High API costs?

```bash
# Check model usage in logs
grep "Processing task" logs/*.log | grep "model"

# Force cheaper model in config
# Edit config/agent-framework.yaml:
# llm.claude_cli_default_model: haiku
```

## Original Bash Implementation

The `scripts/` directory contains the original Bash implementation that this framework is based on. It includes:

- `scripts/start-async-agents.sh` - Original multi-agent startup
- `scripts/async-agent-runner.sh` - Core polling loop
- `scripts/watchdog.sh` - Process monitoring
- `scripts/circuit-breaker.sh` - Safety checks
- `scripts/status-async-agents.sh` - Status dashboard

This Python framework preserves all the safeguards and patterns from the Bash system while adding better configurability and integration capabilities.

## Requirements

- Python 3.10+
- Claude CLI (for claude_cli mode) OR Anthropic API key (for litellm mode)
- Git
- JIRA account with API token
- GitHub account with personal access token

## License

MIT

## Contributing

Contributions welcome! This project aims to be a robust, production-ready agent framework.

## Credits

Built based on the async agent system developed during SnapEdge platform development. Ported from Bash to Python with enhanced features and configurability.
