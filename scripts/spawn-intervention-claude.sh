#!/bin/bash
# Spawn a Claude Code instance to help with manual intervention in the async agent system
# Usage: ASYNC_AGENT_WORKSPACE=/path/to/workspace bash scripts/spawn-intervention-claude.sh [task-id]
set -euo pipefail

WORKSPACE="${ASYNC_AGENT_WORKSPACE:-$HOME/async-workspace}"
TASK_ID="${1:-}"

# Build context for Claude
build_context() {
    echo "You are helping with manual intervention in an async agent system."
    echo ""
    echo "WORKSPACE: $WORKSPACE"
    echo ""
    echo "SYSTEM OVERVIEW:"
    echo "- 4 concurrent agents: agent-d (CPO), agent-a (CTO), agent-b (Engineer), agent-c (QA)"
    echo "- Task queues: $WORKSPACE/.agent-communication/queues/{agent-d,agent-a,agent-b,agent-c}/"
    echo "- Completed tasks: $WORKSPACE/.agent-communication/completed/"
    echo "- Logs: $WORKSPACE/.agent-communication/logs/"
    echo ""
    echo "AVAILABLE TOOLS:"
    echo "1. View system status:"
    echo "   ASYNC_AGENT_WORKSPACE=$WORKSPACE bash scripts/status-async-agents.sh"
    echo ""
    echo "2. Investigate stalled tasks:"
    echo "   ASYNC_AGENT_WORKSPACE=$WORKSPACE bash scripts/investigate-stalled-tasks.sh"
    echo ""
    echo "3. View task file:"
    echo "   cat $WORKSPACE/.agent-communication/queues/{agent-id}/task-{id}.json"
    echo ""
    echo "4. Change task status (blocked → pending):"
    echo "   jq '.status = \"pending\"' task.json > tmp && mv tmp task.json"
    echo ""
    echo "5. Reset retry count:"
    echo "   jq '.retry_count = 0 | del(.last_failed_at)' task.json > tmp && mv tmp task.json"
    echo ""
    echo "6. Create completed marker for missing dependency:"
    echo "   echo '{\"id\":\"task-id\",\"status\":\"completed\"}' > $WORKSPACE/.agent-communication/completed/task-id.json"
    echo ""
    echo "7. View agent logs:"
    echo "   tail -50 $WORKSPACE/.agent-communication/logs/agent-{d,a,b,c}.jsonl"
    echo ""

    if [[ -n "$TASK_ID" ]]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "SPECIFIC TASK CONTEXT: $TASK_ID"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        # Find the task file
        local task_file=""
        for agent_queue in "$WORKSPACE/.agent-communication/queues"/*/; do
            if [[ -f "$agent_queue/$TASK_ID.json" ]]; then
                task_file="$agent_queue/$TASK_ID.json"
                break
            fi
        done

        if [[ -n "$task_file" ]]; then
            echo "Task file: $task_file"
            echo ""
            echo "TASK DETAILS:"
            cat "$task_file" | jq .
            echo ""
            echo "TASK ANALYSIS:"

            # Show dependencies
            local deps=$(jq -r '.depends_on // [] | .[]' "$task_file" 2>/dev/null)
            if [[ -n "$deps" ]]; then
                echo ""
                echo "Dependencies:"
                while IFS= read -r dep; do
                    [[ -z "$dep" ]] && continue

                    # Check if dependency exists
                    if [[ -f "$WORKSPACE/.agent-communication/completed/$dep.json" ]]; then
                        echo "  ✓ $dep (completed)"
                    else
                        local found=false
                        for q in "$WORKSPACE/.agent-communication/queues"/*/; do
                            if [[ -f "$q/$dep.json" ]]; then
                                local status=$(jq -r '.status' "$q/$dep.json" 2>/dev/null)
                                echo "  → $dep (status: $status, in queue: $(basename "$q"))"
                                found=true
                                break
                            fi
                        done
                        if ! $found; then
                            echo "  ✗ $dep (MISSING - blocks task forever!)"
                        fi
                    fi
                done <<< "$deps"
            fi

            # Show blocked_by
            local blocked_by=$(jq -r '.blocked_by // ""' "$task_file" 2>/dev/null)
            if [[ -n "$blocked_by" && "$blocked_by" != "null" ]]; then
                echo ""
                echo "Blocked by: $blocked_by"
            fi

            # Show retry info
            local retry_count=$(jq -r '.retry_count // 0' "$task_file" 2>/dev/null)
            if [[ $retry_count -gt 0 ]]; then
                echo ""
                echo "Retry count: $retry_count"
                local last_failed_at=$(jq -r '.last_failed_at // 0' "$task_file" 2>/dev/null)
                if [[ "$last_failed_at" != "0" && "$last_failed_at" != "null" ]]; then
                    echo "Last failed: $(date -r ${last_failed_at%.*} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo 'unknown')"
                fi
            fi
        else
            echo "ERROR: Task $TASK_ID not found in any queue"
            echo ""
            echo "Searching completed directory..."
            if [[ -f "$WORKSPACE/.agent-communication/completed/$TASK_ID.json" ]]; then
                echo "Task is completed:"
                cat "$WORKSPACE/.agent-communication/completed/$TASK_ID.json" | jq .
            else
                echo "Task not found anywhere. It may have been deleted or never existed."
            fi
        fi
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "COMMON INTERVENTION TASKS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1. Fix 'blocked' status tasks:"
    echo "   - Find: ASYNC_AGENT_WORKSPACE=$WORKSPACE bash scripts/investigate-stalled-tasks.sh"
    echo "   - Fix: jq '.status = \"pending\"' task.json > tmp && mv tmp task.json"
    echo ""
    echo "2. Create missing dependency to unblock tasks:"
    echo "   - Identify missing dep from investigation script"
    echo "   - Create marker: echo '{\"id\":\"dep-id\",\"status\":\"completed\"}' > completed/dep-id.json"
    echo ""
    echo "3. Reset task that's stuck in retry loop:"
    echo "   - Reset: jq '.retry_count = 0 | del(.last_failed_at)' task.json > tmp && mv tmp task.json"
    echo ""
    echo "4. Manually complete a task (if work was done outside system):"
    echo "   - Move to completed: mv queues/agent-x/task.json completed/"
    echo "   - Update status: jq '.status = \"completed\" | .completed_at = now' completed/task.json > tmp && mv tmp completed/task.json"
    echo ""
    echo "5. Remove circular dependencies:"
    echo "   - Edit task: jq 'del(.depends_on[] | select(. == \"problematic-dep\"))' task.json > tmp && mv tmp task.json"
    echo ""
    echo "6. Change task priority:"
    echo "   - Boost: jq '.priority = 0' task.json > tmp && mv tmp task.json"
    echo ""
}

# Generate context and show it
CONTEXT=$(build_context)
echo "$CONTEXT"
echo ""

# Ask if user wants to spawn Claude Code
read -p "Spawn Claude Code with this context? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Create a temp file with the context
CONTEXT_FILE=$(mktemp)
echo "$CONTEXT" > "$CONTEXT_FILE"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Spawning Claude Code..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Change to workspace directory so Claude can easily access files
cd "$WORKSPACE"

# Spawn Claude Code with context
exec claude --directory "$WORKSPACE" -- "$(cat "$CONTEXT_FILE")

You are now in the async agent system workspace. You have full access to all task files, logs, and completed tasks.

CURRENT TASK: Help investigate and fix issues in the async agent system.

Start by running the investigation script to see what's blocked:
ASYNC_AGENT_WORKSPACE=$WORKSPACE bash $(dirname "$0")/investigate-stalled-tasks.sh

Then help the user fix any issues you find. You can modify task files, create completed markers, reset retry counts, and more.

Ask the user what they want to do, or suggest fixes based on what you find."
