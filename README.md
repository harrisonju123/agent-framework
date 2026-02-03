# Agent Framework

Open source AI agent framework with JIRA and GitHub integration. Configurable multi-agent system for autonomous task execution with robust safeguards.

## Features

- Pull unassigned JIRA tickets from backlog
- Process tickets using Claude (LiteLLM API or Claude CLI subprocess)
- Automatic git workflows: branch creation, commits, PR opening
- JIRA integration: status transitions, commenting, ticket linking
- Configurable agent roles via YAML
- File-based task queues with atomic locking
- Automatic recovery from failures with watchdog
- Circuit breaker with 8 safety checks
- Cost optimization with automatic model selection (Haiku/Sonnet/Opus)

## Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/your-org/agent-framework.git
cd agent-framework

# Install package
pip install -e .
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

# Edit .env with your credentials
# - JIRA_EMAIL
# - JIRA_API_TOKEN
# - GITHUB_TOKEN
```

### Basic Usage

```bash
# Pull tickets from JIRA backlog
agent pull --project PROJ

# Work on specific ticket
agent run PROJ-123

# Start agents with watchdog
agent start

# Check system status
agent status

# Run safety checks
agent check

# Stop agents gracefully
agent stop
```

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

1. `agent pull` queries JIRA for unassigned tickets → creates task JSON files
2. Agent picks up task → transitions JIRA to "In Progress"
3. Agent sends task to LLM → receives code changes
4. Agent creates git branch → commits changes → pushes to remote
5. Agent opens PR with JIRA link
6. Agent transitions JIRA to "Code Review"

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
# Check agent health and queue status
agent status

# View logs
tail -f logs/engineer.log
tail -f logs/watchdog.log
```

### Stopping Agents

```bash
# Graceful shutdown (SIGTERM)
agent stop

# Force shutdown (SIGKILL)
agent stop --force
```

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
| `agent init` | Initialize project with config templates |
| `agent pull --project PROJ` | Pull unassigned JIRA tickets |
| `agent run PROJ-123` | Work on specific ticket |
| `agent start` | Start all agents + watchdog |
| `agent stop` | Stop all agents gracefully |
| `agent status` | Show agent health and queue stats |
| `agent check` | Run circuit breaker safety checks |
| `agent check --fix` | Auto-fix detected issues |

## Troubleshooting

### Agents not processing tasks?

```bash
# Check if agents are running
ps aux | grep run_agent

# Check heartbeats
ls -la .agent-communication/heartbeats/

# Check logs
tail -f logs/*.log
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
