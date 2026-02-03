#!/bin/bash
set -euo pipefail

# Watchdog - Monitors agent heartbeats and restarts crashed agents
# Run this alongside the agents to provide automatic crash recovery

WORKSPACE="${ASYNC_AGENT_WORKSPACE:-$HOME/async-workspace}"
HEARTBEAT_DIR="$WORKSPACE/.agent-communication/heartbeats"
QUEUE_DIR="$WORKSPACE/.agent-communication/queues"
PIDS_FILE="$WORKSPACE/.agent-communication/pids.txt"
LOG_FILE="$WORKSPACE/.agent-communication/logs/watchdog.jsonl"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

HEARTBEAT_TIMEOUT="${HEARTBEAT_TIMEOUT:-90}"  # seconds - if heartbeat older than this, agent is dead
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"  # seconds - how often to check heartbeats

AGENTS=("agent-d" "agent-a" "agent-b" "agent-c")

log() {
    local level=$1
    shift
    local msg="$*"
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "{\"timestamp\":\"$timestamp\",\"agent\":\"watchdog\",\"level\":\"$level\",\"message\":\"$msg\"}" >> "$LOG_FILE"
}

cleanup() {
    log "INFO" "Watchdog shutting down (signal received)"
    exit 0
}

trap cleanup TERM INT EXIT

get_agent_pid() {
    local agent_id=$1
    # Read PID from pids.txt file
    if [[ -f "$PIDS_FILE" ]]; then
        grep "^$agent_id:" "$PIDS_FILE" 2>/dev/null | cut -d: -f2 || echo ""
    else
        echo ""
    fi
}

is_process_alive() {
    local pid=$1
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    return 1
}

reset_in_progress_tasks() {
    local agent_id=$1
    local queue="$QUEUE_DIR/$agent_id"

    # Find all in_progress tasks for this agent and reset to pending
    while IFS= read -r task_file; do
        [[ ! -f "$task_file" ]] && continue

        status=$(jq -r '.status' "$task_file" 2>/dev/null || echo "")
        if [[ "$status" == "in_progress" ]]; then
            task_id=$(jq -r '.id' "$task_file")
            log "WARN" "Resetting in_progress task $task_id to pending (agent $agent_id crashed)"

            # Reset to pending
            tmp_file="$task_file.tmp"
            jq '.status = "pending" | del(.started_at) | del(.started_by) | .retry_count = (.retry_count // 0) + 1 | .last_failed_at = now' "$task_file" > "$tmp_file"
            mv "$tmp_file" "$task_file"
        fi
    done < <(find "$queue" -name "*.json" -type f 2>/dev/null)
}

restart_agent() {
    local agent_id=$1
    log "INFO" "Restarting agent: $agent_id"

    # Reset any in-progress tasks
    reset_in_progress_tasks "$agent_id"

    # Launch agent in background
    nohup bash "$SCRIPT_DIR/async-agent-runner.sh" "$agent_id" > "$WORKSPACE/$agent_id.log" 2>&1 &
    local new_pid=$!

    # Update PID file
    if [[ -f "$PIDS_FILE" ]]; then
        # Remove old entry
        sed -i.bak "/^$agent_id:/d" "$PIDS_FILE" 2>/dev/null || true
    fi
    echo "$agent_id:$new_pid" >> "$PIDS_FILE"

    log "INFO" "Agent $agent_id restarted with PID $new_pid"
}

check_agent_health() {
    local agent_id=$1
    local heartbeat_file="$HEARTBEAT_DIR/$agent_id"

    # Check if heartbeat file exists
    if [[ ! -f "$heartbeat_file" ]]; then
        log "WARN" "No heartbeat file for $agent_id - agent may not be running"
        return 1
    fi

    # Read last heartbeat timestamp
    local last_heartbeat=$(cat "$heartbeat_file")
    local now=$(date +%s)
    local age=$((now - last_heartbeat))

    if [[ $age -gt $HEARTBEAT_TIMEOUT ]]; then
        log "ERROR" "Agent $agent_id heartbeat is $age seconds old (timeout: $HEARTBEAT_TIMEOUT)"
        return 1
    fi

    # Heartbeat is fresh
    log "DEBUG" "Agent $agent_id heartbeat is fresh ($age seconds old)"
    return 0
}

# Setup
mkdir -p "$HEARTBEAT_DIR" "$(dirname "$LOG_FILE")"
log "INFO" "Watchdog starting (checking every ${CHECK_INTERVAL}s, timeout: ${HEARTBEAT_TIMEOUT}s)"

# Main monitoring loop
while true; do
    for agent_id in "${AGENTS[@]}"; do
        if ! check_agent_health "$agent_id"; then
            # Agent is dead or unresponsive
            agent_pid=$(get_agent_pid "$agent_id")

            if [[ -n "$agent_pid" ]]; then
                if is_process_alive "$agent_pid"; then
                    # Process exists but heartbeat is stale - kill zombie process
                    log "WARN" "Killing zombie process $agent_pid for $agent_id"
                    kill -9 "$agent_pid" 2>/dev/null || true
                    sleep 2
                else
                    log "WARN" "Agent $agent_id process $agent_pid already dead"
                fi
            fi

            # Restart the agent
            restart_agent "$agent_id"
        fi
    done

    sleep "$CHECK_INTERVAL"
done
