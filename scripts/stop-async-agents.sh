#!/bin/bash
set -euo pipefail

# Async Agent System Shutdown
# Gracefully stops all running agents

WORKSPACE="${ASYNC_AGENT_WORKSPACE:-$HOME/async-workspace}"
PID_FILE="$WORKSPACE/.agent-communication/pids.txt"

echo "🛑 Stopping Async Agent System"
echo "================================"
echo "Workspace: $WORKSPACE"
echo ""

if [[ ! -f "$PID_FILE" ]]; then
    echo "No PID file found at: $PID_FILE"
    echo "Agents may not be running or were started manually."
    exit 1
fi

# Read PIDs and attempt graceful shutdown
while IFS=: read -r agent pid; do
    if [[ -z "$pid" ]]; then
        continue
    fi

    # Check if process is still running
    if kill -0 "$pid" 2>/dev/null; then
        echo "Stopping $agent (PID: $pid)..."
        kill -TERM "$pid" 2>/dev/null || true
    else
        echo "$agent (PID: $pid) - already stopped"
    fi
done < "$PID_FILE"

# Wait for processes to terminate gracefully
echo ""
echo "Waiting for agents to shut down gracefully (max 10 seconds)..."
sleep 2

# Check if all processes stopped
all_stopped=true
while IFS=: read -r agent pid; do
    if [[ -z "$pid" ]]; then
        continue
    fi

    if kill -0 "$pid" 2>/dev/null; then
        echo "⚠️  $agent (PID: $pid) still running, waiting..."
        all_stopped=false
    fi
done < "$PID_FILE"

if ! $all_stopped; then
    sleep 3
    # Force kill if still running
    while IFS=: read -r agent pid; do
        if [[ -z "$pid" ]]; then
            continue
        fi

        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing $agent (PID: $pid)..."
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done < "$PID_FILE"
fi

# Clean up lock files
echo ""
echo "Cleaning up lock files..."
if [[ -d "$WORKSPACE/.agent-communication/locks" ]]; then
    find "$WORKSPACE/.agent-communication/locks" -type d -name "*.lock" -exec rm -rf {} + 2>/dev/null || true
fi

# Reset any in-progress tasks to pending status
echo "Checking for abandoned in-progress tasks..."
abandoned_count=0
for queue_dir in "$WORKSPACE/.agent-communication/queues"/*; do
    if [[ -d "$queue_dir" ]]; then
        # Use find to avoid shell expansion issues with *.json
        while IFS= read -r task_file; do
            [[ ! -f "$task_file" ]] && continue
            status=$(jq -r '.status' "$task_file" 2>/dev/null || echo "")
            if [[ "$status" == "in_progress" ]]; then
                task_id=$(basename "$task_file" .json)
                echo "  Resetting $task_id to pending..."
                jq '.status = "pending" | del(.started_at) | del(.started_by)' "$task_file" > "$task_file.tmp"
                mv "$task_file.tmp" "$task_file"
                ((abandoned_count++))
            fi
        done < <(find "$queue_dir" -name "*.json" -type f 2>/dev/null)
    fi
done

if [[ $abandoned_count -gt 0 ]]; then
    echo "  Reset $abandoned_count abandoned tasks to pending status"
else
    echo "  No abandoned tasks found"
fi

# Remove PID file
rm -f "$PID_FILE"

echo ""
echo "✅ All agents stopped"
echo ""
echo "Queue status:"
for queue_dir in "$WORKSPACE/.agent-communication/queues"/*; do
    if [[ -d "$queue_dir" ]]; then
        agent=$(basename "$queue_dir")
        count=$(ls "$queue_dir"/*.json 2>/dev/null | wc -l | tr -d ' ')
        echo "  $agent: $count pending tasks"
    fi
done

echo ""
echo "Completed tasks: $(ls "$WORKSPACE/.agent-communication/completed"/*.json 2>/dev/null | wc -l | tr -d ' ')"
