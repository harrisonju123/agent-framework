#!/bin/bash
set -euo pipefail

# Async Agent Runner
# Continuously polls a task queue and executes tasks for a single agent

AGENT_ID=$1  # agent-d, agent-a, agent-b, agent-c
if [[ -z "${AGENT_ID:-}" ]]; then
    echo "Usage: $0 <agent-id>"
    echo "Example: $0 agent-b"
    exit 1
fi

WORKSPACE="${ASYNC_AGENT_WORKSPACE:-$HOME/async-workspace}"
QUEUE_DIR="$WORKSPACE/.agent-communication/queues/$AGENT_ID"
LOCK_DIR="$WORKSPACE/.agent-communication/locks"
COMPLETED_DIR="$WORKSPACE/.agent-communication/completed"
HEARTBEAT_DIR="$WORKSPACE/.agent-communication/heartbeats"
LOG_FILE="$WORKSPACE/.agent-communication/logs/$AGENT_ID.jsonl"  # Per-agent log file
POLL_INTERVAL="${POLL_INTERVAL:-30}"  # seconds
TASK_TIMEOUT="${TASK_TIMEOUT:-1800}"  # 30 minutes default timeout for tasks
MAX_RETRIES="${MAX_RETRIES:-5}"  # Maximum retry attempts before marking failed
FORCE_MODEL="${FORCE_MODEL:-}"  # Optional: force specific model (haiku/sonnet/opus)

# Track current task for cleanup on signal
current_task_id=""

log() {
    local level=$1
    shift
    local msg="$*"
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    # Write to per-agent log file (no shared file = no race condition)
    echo "{\"timestamp\":\"$timestamp\",\"agent\":\"$AGENT_ID\",\"level\":\"$level\",\"message\":\"$msg\"}" >> "$LOG_FILE"
}

write_heartbeat() {
    # Write current Unix timestamp to heartbeat file
    # Watchdog uses this to detect crashed agents
    echo "$(date +%s)" > "$HEARTBEAT_DIR/$AGENT_ID"
}

cleanup() {
    log "INFO" "Shutting down $AGENT_ID (signal received)"
    if [[ -n "$current_task_id" ]]; then
        log "WARN" "Releasing lock for current task: $current_task_id"
        release_lock "$current_task_id"
    fi
    # Write final heartbeat before exit (signals graceful shutdown)
    write_heartbeat
    exit 0
}

# Trap signals for graceful shutdown
trap cleanup TERM INT EXIT

find_next_task() {
    # Find oldest pending task with no unmet dependencies
    # Sort by filename (which includes timestamp for chronological ordering)
    for task_file in $(ls -t "$QUEUE_DIR"/*.json 2>/dev/null | tail -r 2>/dev/null || ls "$QUEUE_DIR"/*.json 2>/dev/null | sort); do
        # Skip if file doesn't exist (race condition)
        [[ ! -f "$task_file" ]] && continue

        # Parse JSON safely
        if ! jq empty "$task_file" 2>/dev/null; then
            log "ERROR" "Invalid JSON in $task_file, skipping"
            continue
        fi

        status=$(jq -r '.status' "$task_file" 2>/dev/null || echo "")

        # Only process pending tasks
        if [[ "$status" != "pending" ]]; then
            continue
        fi

        # Check exponential backoff for failed tasks
        retry_count=$(jq -r '.retry_count // 0' "$task_file" 2>/dev/null)
        if [[ $retry_count -gt 0 ]]; then
            last_failed_at=$(jq -r '.last_failed_at // 0' "$task_file" 2>/dev/null)
            now=$(date +%s)

            # Calculate backoff: 30 * 2^(retry_count-1), max 240 seconds
            backoff=$((30 * (2 ** (retry_count - 1))))
            if [[ $backoff -gt 240 ]]; then
                backoff=240
            fi

            # Check if enough time has passed since last failure
            # Strip decimal part from timestamp for bash arithmetic
            time_since_failure=$((now - ${last_failed_at%.*}))
            if [[ $time_since_failure -lt $backoff ]]; then
                log "DEBUG" "Task $(basename "$task_file" .json) waiting for backoff: $time_since_failure/$backoff seconds"
                continue
            fi
        fi

        # Check if dependencies are met
        deps_met=true
        while IFS= read -r dep; do
            if [[ -n "$dep" && "$dep" != "null" ]]; then
                if [[ ! -f "$COMPLETED_DIR/$dep.json" ]]; then
                    deps_met=false
                    log "DEBUG" "Task $(basename "$task_file" .json) waiting for dependency: $dep"
                    break
                fi
            fi
        done < <(jq -r '.depends_on[]?' "$task_file" 2>/dev/null)

        if $deps_met; then
            echo "$task_file"
            return 0
        fi
    done
    return 1
}

acquire_lock() {
    local task_id=$1
    local lock_file="$LOCK_DIR/$task_id.lock"

    # Check if lock exists and is stale
    if [[ -d "$lock_file" ]]; then
        if [[ -f "$lock_file/pid" ]]; then
            lock_pid=$(cat "$lock_file/pid")
            # Check if process is still running
            if ! kill -0 "$lock_pid" 2>/dev/null; then
                log "WARN" "Removing stale lock for $task_id (pid $lock_pid no longer exists)"
                rm -rf "$lock_file"
            else
                log "DEBUG" "Task $task_id is locked by pid $lock_pid"
                return 1
            fi
        else
            # Lock exists but no PID file - remove it
            log "WARN" "Removing invalid lock for $task_id (no PID file)"
            rm -rf "$lock_file"
        fi
    fi

    # Try to acquire lock (mkdir is atomic)
    if mkdir "$lock_file" 2>/dev/null; then
        echo $$ > "$lock_file/pid"
        log "DEBUG" "Acquired lock for $task_id (pid $$)"
        return 0
    fi

    return 1
}

release_lock() {
    local task_id=$1
    local lock_file="$LOCK_DIR/$task_id.lock"
    if [[ -d "$lock_file" ]]; then
        rm -rf "$lock_file"
        log "DEBUG" "Released lock for $task_id"
    fi
}

mark_in_progress() {
    local task_file=$1
    local tmp_file="$task_file.tmp"

    # Atomic update: write to temp file, then rename
    jq '.status = "in_progress" | .started_at = now | .started_by = env.AGENT_ID' "$task_file" > "$tmp_file"
    mv "$tmp_file" "$task_file"
}

mark_completed() {
    local task_file=$1
    local task_id=$(basename "$task_file" .json)
    local tmp_file="$COMPLETED_DIR/$task_id.json.tmp"
    local final_file="$COMPLETED_DIR/$task_id.json"

    # Atomic update: write to temp file, then rename
    jq '.status = "completed" | .completed_at = now | .completed_by = env.AGENT_ID' "$task_file" > "$tmp_file"
    mv "$tmp_file" "$final_file"

    # Remove from queue
    rm -f "$task_file"
}

reset_task_to_pending() {
    local task_file=$1
    local tmp_file="$task_file.tmp"

    # Reset task to pending status so it can be retried
    # Track last_failed_at for exponential backoff
    jq '.status = "pending" | del(.started_at) | del(.started_by) | .retry_count = (.retry_count // 0) + 1 | .last_failed_at = now' "$task_file" > "$tmp_file"
    mv "$tmp_file" "$task_file"
}

mark_failed() {
    local task_file=$1
    local task_id=$(basename "$task_file" .json)
    local tmp_file="$task_file.tmp"

    # Mark task as permanently failed
    jq '.status = "failed" | .failed_at = now | .failed_by = env.AGENT_ID' "$task_file" > "$tmp_file"
    mv "$tmp_file" "$task_file"

    log "ERROR" "Task $task_id marked as FAILED after exceeding max retries"
}

create_escalation() {
    local task_file=$1
    local task_id=$(basename "$task_file" .json)
    local escalation_id="escalation-$(date +%s)-$task_id"

    # Escalation goes to agent-d (CPO) by default
    local escalation_queue="$WORKSPACE/.agent-communication/queues/agent-d"
    local escalation_file="$escalation_queue/$escalation_id.json"
    local tmp_file="$escalation_file.tmp"

    # Read original task details
    local task_title=$(jq -r '.title' "$task_file")
    local task_type=$(jq -r '.type' "$task_file")
    local retry_count=$(jq -r '.retry_count // 0' "$task_file")

    # Create escalation task
    cat > "$tmp_file" << EOF
{
  "id": "$escalation_id",
  "type": "escalation",
  "status": "pending",
  "priority": 0,
  "created_by": "$AGENT_ID",
  "assigned_to": "agent-d",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "depends_on": [],
  "blocks": [],
  "title": "ESCALATION: Task failed after $retry_count retries",
  "description": "Task $task_id failed after $retry_count retry attempts and has been marked as failed. This requires human intervention or product decision.\\n\\nOriginal task: $task_title\\nTask type: $task_type\\n\\nPlease review the failed task and decide next steps.",
  "failed_task_id": "$task_id",
  "needs_human_review": true,
  "context": {
    "original_task_file": "$task_file"
  }
}
EOF

    # Atomic move
    mv "$tmp_file" "$escalation_file"

    log "WARN" "Created escalation task $escalation_id for failed task $task_id"
}

# Setup
mkdir -p "$QUEUE_DIR" "$LOCK_DIR" "$COMPLETED_DIR" "$HEARTBEAT_DIR" "$(dirname "$LOG_FILE")"
log "INFO" "Starting $AGENT_ID runner (pid $$)"
write_heartbeat  # Initial heartbeat on startup

# Main polling loop
while true; do
    # Write heartbeat every iteration (watchdog monitors this)
    write_heartbeat

    if task_file=$(find_next_task); then
        task_id=$(jq -r '.id' "$task_file")
        task_title=$(jq -r '.title' "$task_file")
        log "INFO" "Found task: $task_id - $task_title"

        if acquire_lock "$task_id"; then
            current_task_id="$task_id"  # Track for signal handler cleanup
            mark_in_progress "$task_file"

            # Build prompt from task
            PROMPT=$(cat << EOF
You are $AGENT_ID working on an asynchronous task.

TASK DETAILS:
$(cat "$task_file")

YOUR RESPONSIBILITIES:
1. Complete the task described above
2. Create any follow-up tasks by writing JSON files to other agents' queues:
   - Write to: $WORKSPACE/.agent-communication/queues/<agent-id>/task-<id>.json.tmp
   - Then rename to: task-<id>.json (atomic operation)
   - Task JSON format must match the schema shown above
3. Use unique task IDs (timestamp or UUID)
4. Set depends_on array for tasks that depend on this one completing

QUEUE DIRECTORIES:
- agent-d queue: $WORKSPACE/.agent-communication/queues/agent-d/
- agent-a queue: $WORKSPACE/.agent-communication/queues/agent-a/
- agent-b queue: $WORKSPACE/.agent-communication/queues/agent-b/
- agent-c queue: $WORKSPACE/.agent-communication/queues/agent-c/

IMPORTANT:
- Create tasks for ANY agent as needed (not just hierarchical flow)
- Use atomic writes (write to .tmp file, then mv to final name)
- Task status should be "pending" when you create it
- This task will be automatically marked as completed when you're done
EOF
)

            # Select model based on task type and complexity
            # Haiku: fast/cheap for simple tasks
            # Sonnet: balanced for most work
            # Opus: complex/difficult tasks (high retry count)
            select_model() {
                local task_type=$1
                local retry_count=$2

                # Check for forced model override
                if [[ -n "$FORCE_MODEL" ]]; then
                    echo "$FORCE_MODEL"
                    return
                fi

                # Escalate to stronger model if task keeps failing
                if [[ $retry_count -ge 3 ]]; then
                    echo "opus"
                    return
                fi

                # Select based on task type
                case "$task_type" in
                    escalation)
                        # Escalations are important - use best model
                        # But limit retries to prevent infinite loops
                        echo "opus"
                        ;;
                    testing|verification|fix|bugfix|bug-fix)
                        echo "haiku"
                        ;;
                    coordination|status_report|documentation)
                        echo "haiku"
                        ;;
                    implementation|architecture|planning|review|enhancement)
                        echo "sonnet"
                        ;;
                    *)
                        echo "sonnet"  # Default to sonnet for unknown types
                        ;;
                esac
            }

            task_type=$(jq -r '.type // "unknown"' "$task_file")
            retry_count=$(jq -r '.retry_count // 0' "$task_file")
            MODEL=$(select_model "$task_type" "$retry_count")

            log "INFO" "Processing task $task_id with Claude ($MODEL model, type: $task_type)..."

            # Check if timeout command exists (GNU coreutils)
            if command -v timeout &> /dev/null; then
                TIMEOUT_CMD="timeout $TASK_TIMEOUT"
            elif command -v gtimeout &> /dev/null; then
                TIMEOUT_CMD="gtimeout $TASK_TIMEOUT"
            else
                TIMEOUT_CMD=""
                log "WARN" "timeout command not found, running without timeout limit"
            fi

            if echo "$PROMPT" | $TIMEOUT_CMD claude \
                --model "$MODEL" \
                --dangerously-skip-permissions \
                --max-turns 999; then

                mark_completed "$task_file"
                current_task_id=""  # Clear current task
                release_lock "$task_id"
                log "INFO" "Completed task: $task_id"
            else
                exit_code=$?
                if [[ $exit_code -eq 124 ]]; then
                    # Timeout occurred
                    log "ERROR" "Task $task_id timed out after ${TASK_TIMEOUT}s"
                else
                    # Claude failed
                    log "ERROR" "Claude failed for task $task_id (exit code: $exit_code)"
                fi

                # Check if task has exceeded max retries
                retry_count=$(jq -r '.retry_count // 0' "$task_file")
                task_type=$(jq -r '.type // "unknown"' "$task_file")

                if [[ $retry_count -ge $MAX_RETRIES ]]; then
                    # Max retries exceeded - mark as failed
                    log "ERROR" "Task $task_id has failed $retry_count times (max: $MAX_RETRIES)"
                    mark_failed "$task_file"

                    # CRITICAL: Prevent infinite loop - escalations should NOT create more escalations
                    if [[ "$task_type" == "escalation" ]]; then
                        log "ERROR" "Escalation task $task_id failed after $retry_count retries - NOT creating another escalation (would cause infinite loop)"
                        log "ERROR" "This escalation requires immediate human intervention"
                    else
                        create_escalation "$task_file"
                    fi
                else
                    # Reset task to pending so it can be retried
                    log "WARN" "Resetting task $task_id to pending status (retry $((retry_count + 1))/$MAX_RETRIES)"
                    reset_task_to_pending "$task_file"
                fi

                current_task_id=""  # Clear current task
                release_lock "$task_id"
            fi
        else
            log "DEBUG" "Could not acquire lock for $task_id, will retry later"
        fi
    else
        log "DEBUG" "No tasks available, sleeping for ${POLL_INTERVAL}s..."
    fi

    sleep "$POLL_INTERVAL"
done
