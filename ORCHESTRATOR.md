# Agent Framework Orchestrator

The orchestrator enables running multiple AI agents concurrently with full process management, health monitoring, and safety checks.

## Features Implemented

### Process Management (`orchestrator.py`)
- Spawn multiple agent subprocesses
- PID tracking and stale PID detection
- Graceful shutdown (SIGTERM → SIGKILL)
- Signal handling for clean exit
- 0.5s staggered startup
- Lock file cleanup
- Reset in-progress tasks on shutdown

### Watchdog (`watchdog.py`)
- Heartbeat monitoring (90s timeout)
- Auto-restart dead agents
- Zombie process detection and cleanup
- Reset orphaned in-progress tasks
- 60s check interval

### Circuit Breaker (`circuit_breaker.py`)
8 safety checks:
1. Queue size limits (100 per agent)
2. Escalation count (max 50)
3. Circular dependency detection
4. Stale task detection (>7 days)
5. Task creation rate monitoring
6. Stuck task detection (3+ retries)
7. Duplicate ID detection
8. Escalation retry prevention

Auto-fix mode:
- Archive stale tasks
- Reset escalation retries

### Post-LLM Workflow (`agent.py`)
After Claude completes a task:
1. Create git branch: `feature/{ticket-id}-{slug}`
2. Stage and commit changes
3. Push to GitHub
4. Open PR with JIRA link
5. Transition JIRA ticket to "Code Review"

## CLI Commands

```bash
# Start all agents with watchdog
agent start

# Start without watchdog
agent start --no-watchdog

# Stop agents gracefully
agent stop

# Force stop
agent stop --force

# Check system status
agent status

# Run circuit breaker checks
agent check

# Auto-fix issues
agent check --fix

# Pull JIRA tickets
agent pull --project PROJ

# Work on specific ticket
agent run PROJ-123
```

## Configuration

**agents.yaml**
```yaml
agents:
  - id: engineer
    queue: engineer
    enabled: true
    prompt: "Implement features..."
    jira_can_update_status: true
    can_commit: true
```

**agent-framework.yaml**
```yaml
workspace: "."
llm:
  mode: claude_cli
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

## Architecture

```
Orchestrator
├── Spawns agent subprocesses
│   └── python -m agent_framework.run_agent <agent_id>
├── Spawns watchdog subprocess
│   └── python -m agent_framework.run_watchdog
└── Manages PIDs in pids.txt

Agent Process
├── Polls task queue every 30s
├── Acquires lock on task
├── Calls Claude CLI subprocess
├── On success: git + PR + JIRA workflow
└── Writes heartbeat every iteration

Watchdog Process
├── Checks heartbeats every 60s
├── Detects dead agents (>90s)
├── Kills zombies
├── Resets orphaned tasks
└── Restarts dead agents
```

## File Structure

```
workspace/
├── .agent-communication/
│   ├── queues/
│   │   ├── engineer/           # Task JSON files
│   │   ├── qa/
│   │   └── architect/
│   ├── completed/              # Completed tasks
│   ├── locks/                  # Task locks (mkdir-based)
│   ├── heartbeats/             # Agent heartbeat timestamps
│   ├── pids.txt                # Process IDs
│   └── archived/               # Stale tasks
└── logs/
    ├── engineer.log
    ├── qa.log
    └── watchdog.log
```

## Safety Guarantees

1. **No infinite loops**: Escalations cannot create more escalations
2. **No resource exhaustion**: Queue size limits enforced
3. **No zombie processes**: Watchdog kills and restarts
4. **No lost tasks**: Orphaned tasks reset to pending
5. **No duplicate work**: Atomic mkdir-based locking
6. **No stale data**: Old tasks archived automatically
7. **No circular deps**: Graph cycle detection
8. **Graceful shutdown**: SIGTERM with timeout

## Workflow Example

```bash
# 1. Initialize project
agent init

# 2. Configure
cp config/agents.yaml.example config/agents.yaml
cp config/jira.yaml.example config/jira.yaml
cp config/github.yaml.example config/github.yaml

# Edit configs with credentials
export JIRA_EMAIL="you@company.com"
export JIRA_API_TOKEN="token"
export GITHUB_TOKEN="token"

# 3. Pull tickets from JIRA
agent pull --project PROJ

# 4. Start agents
agent start

# 5. Monitor
agent status
agent check

# 6. Stop
agent stop
```

## Verification

Test the orchestrator:
```bash
# Start agents
agent start

# Check PIDs exist
cat .agent-communication/pids.txt

# Check agents are running
ps aux | grep run_agent

# Check heartbeats are fresh
ls -la .agent-communication/heartbeats/

# Kill an agent manually
kill <pid>

# Watchdog should restart it within 60s

# Stop all agents
agent stop
```

## Troubleshooting

**Agents won't start**
- Check logs in `logs/<agent-id>.log`
- Verify agent config in `config/agents.yaml`
- Ensure workspace is a git repo

**Watchdog not restarting agents**
- Check watchdog log: `logs/watchdog.log`
- Verify heartbeat files exist
- Check heartbeat timeout settings

**Circuit breaker failing**
- Run `agent check` to see specific issues
- Use `agent check --fix` to auto-fix
- Check queue sizes with `agent status`

**Tasks stuck in pending**
- Check dependencies with circuit breaker
- Verify agents are running
- Check for stale locks

## Dependencies

All dependencies in `pyproject.toml`:
- subprocess (stdlib) - process management
- signal (stdlib) - graceful shutdown
- pathlib (stdlib) - file operations
- click - CLI framework
- rich - console output
- pydantic - config validation

No external process managers needed (no systemd, supervisor, etc)
