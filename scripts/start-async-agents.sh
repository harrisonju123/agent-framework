#!/bin/bash
set -euo pipefail

# Async Agent System Launcher
# Starts all 4 agents in background processes
#
# Environment Variables:
#   ASYNC_AGENT_WORKSPACE - Workspace directory (default: $HOME/async-workspace)
#   FORCE_MODEL - Force specific model for all tasks (haiku/sonnet/opus)
#                 By default, model is auto-selected based on task type:
#                   - haiku: testing, verification, fixes, documentation
#                   - sonnet: implementation, architecture, planning, review
#                   - opus: tasks with 3+ retries (difficult tasks)
#   POLL_INTERVAL - Seconds between queue polls (default: 30)
#   TASK_TIMEOUT - Max seconds per task (default: 1800 = 30min)
#   MAX_RETRIES - Max retry attempts before marking failed (default: 5)

WORKSPACE="${ASYNC_AGENT_WORKSPACE:-$HOME/async-workspace}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER_SCRIPT="$SCRIPT_DIR/async-agent-runner.sh"
WATCHDOG_SCRIPT="$SCRIPT_DIR/watchdog.sh"

# Parse command-line arguments
ENABLE_WATCHDOG=false
for arg in "$@"; do
    case $arg in
        --watchdog|--with-watchdog)
            ENABLE_WATCHDOG=true
            shift
            ;;
    esac
done

echo "🚀 Starting Async Agent System"
echo "================================"
echo "Workspace: $WORKSPACE"
echo ""

# Setup directory structure
echo "Setting up directories..."
mkdir -p "$WORKSPACE/.agent-communication/queues/agent-d"
mkdir -p "$WORKSPACE/.agent-communication/queues/agent-a"
mkdir -p "$WORKSPACE/.agent-communication/queues/agent-b"
mkdir -p "$WORKSPACE/.agent-communication/queues/agent-c"
mkdir -p "$WORKSPACE/.agent-communication/completed"
mkdir -p "$WORKSPACE/.agent-communication/locks"
mkdir -p "$WORKSPACE/logs"

# Ensure we're in the workspace directory
cd "$WORKSPACE"

# Create initial bootstrap task for Agent D
BOOTSTRAP_FILE="$WORKSPACE/.agent-communication/queues/agent-d/task-bootstrap.json"
if [[ ! -f "$BOOTSTRAP_FILE" ]]; then
    echo "Creating bootstrap task for Agent D..."
    cat > "$BOOTSTRAP_FILE.tmp" << EOF
{
  "id": "task-bootstrap",
  "type": "initialize",
  "status": "pending",
  "priority": 0,
  "created_by": "system",
  "assigned_to": "agent-d",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "depends_on": [],
  "blocks": [],
  "title": "Initialize product roadmap",
  "description": "Read INITIATIVES.md to understand the product vision. Create the first product requirement task for Agent A based on the highest priority initiative. This is the bootstrap task that kicks off the entire development process.",
  "acceptance_criteria": [
    "Read INITIATIVES.md completely",
    "Identify highest priority initiative",
    "Create at least one product_requirement task in agent-a's queue"
  ],
  "context": {
    "workspace": "$WORKSPACE",
    "instructions_file": ".agent-communication/agent-d-instructions.md"
  }
}
EOF
    mv "$BOOTSTRAP_FILE.tmp" "$BOOTSTRAP_FILE"
    echo "✓ Bootstrap task created"
fi

# Check for already-running agents
PID_FILE="$WORKSPACE/.agent-communication/pids.txt"
if [[ -f "$PID_FILE" ]]; then
    echo "⚠️  Found existing PID file. Checking for running agents..."
    any_running=false
    while IFS=: read -r agent pid; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "  ❌ $agent (PID: $pid) is already running"
            any_running=true
        fi
    done < "$PID_FILE"

    if $any_running; then
        echo ""
        echo "Error: Some agents are already running. Stop them first with:"
        echo "  bash $SCRIPT_DIR/stop-async-agents.sh"
        exit 1
    else
        echo "  ✓ PIDs in file are stale, proceeding..."
        rm -f "$PID_FILE"
    fi
fi

# Verify runner script exists
if [[ ! -f "$RUNNER_SCRIPT" ]]; then
    echo "❌ Error: Runner script not found at: $RUNNER_SCRIPT"
    exit 1
fi

# Launch all agents in background
echo ""
echo "Launching agents..."

export ASYNC_AGENT_WORKSPACE="$WORKSPACE"

# Agent D (CPO)
nohup bash "$RUNNER_SCRIPT" agent-d > "$WORKSPACE/logs/agent-d.log" 2>&1 &
AGENT_D_PID=$!
echo "📋 Agent D (CPO) started - PID: $AGENT_D_PID"
sleep 0.5  # Throttle startup

# Agent A (CTO)
nohup bash "$RUNNER_SCRIPT" agent-a > "$WORKSPACE/logs/agent-a.log" 2>&1 &
AGENT_A_PID=$!
echo "🧠 Agent A (CTO) started - PID: $AGENT_A_PID"
sleep 0.5

# Agent B (Engineer)
nohup bash "$RUNNER_SCRIPT" agent-b > "$WORKSPACE/logs/agent-b.log" 2>&1 &
AGENT_B_PID=$!
echo "⚙️  Agent B (Engineer) started - PID: $AGENT_B_PID"
sleep 0.5

# Agent C (QA)
nohup bash "$RUNNER_SCRIPT" agent-c > "$WORKSPACE/logs/agent-c.log" 2>&1 &
AGENT_C_PID=$!
echo "🔍 Agent C (QA) started - PID: $AGENT_C_PID"

# Verify all agents are still running after brief delay
echo ""
echo "Verifying agents started successfully..."
sleep 2

all_started=true
for agent_info in "agent-d:$AGENT_D_PID" "agent-a:$AGENT_A_PID" "agent-b:$AGENT_B_PID" "agent-c:$AGENT_C_PID"; do
    IFS=: read -r agent pid <<< "$agent_info"
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "  ❌ $agent (PID: $pid) failed to start or crashed immediately"
        all_started=false
    else
        echo "  ✓ $agent (PID: $pid) running"
    fi
done

if ! $all_started; then
    echo ""
    echo "❌ Some agents failed to start. Check logs:"
    echo "  tail -50 $WORKSPACE/logs/agent-*.log"
    exit 1
fi

# Save PIDs to file for stop script
cat > "$PID_FILE" << EOF
agent-d:$AGENT_D_PID
agent-a:$AGENT_A_PID
agent-b:$AGENT_B_PID
agent-c:$AGENT_C_PID
EOF

# Launch watchdog if enabled
if $ENABLE_WATCHDOG; then
    echo ""
    echo "Launching watchdog..."
    nohup bash "$WATCHDOG_SCRIPT" > "$WORKSPACE/logs/watchdog.log" 2>&1 &
    WATCHDOG_PID=$!
    sleep 1

    if kill -0 "$WATCHDOG_PID" 2>/dev/null; then
        echo "🐕 Watchdog started - PID: $WATCHDOG_PID"
        echo "watchdog:$WATCHDOG_PID" >> "$PID_FILE"
    else
        echo "  ⚠️  Watchdog failed to start (check logs/watchdog.log)"
    fi
fi

echo ""
echo "✅ All agents running!"
if $ENABLE_WATCHDOG; then
    echo "   (with watchdog monitoring)"
fi
echo ""
echo "Monitor with:"
echo "  tail -f $WORKSPACE/.agent-communication/log.jsonl"
echo "  tail -f $WORKSPACE/logs/agent-*.log"
if $ENABLE_WATCHDOG; then
    echo "  tail -f $WORKSPACE/logs/watchdog.log"
fi
echo ""
echo "Stop with:"
echo "  bash $SCRIPT_DIR/stop-async-agents.sh"
echo ""
echo "Check status:"
echo "  ls -la $WORKSPACE/.agent-communication/queues/*/"
echo "  ls -la $WORKSPACE/.agent-communication/completed/"
