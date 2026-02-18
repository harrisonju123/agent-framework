#!/bin/bash
set -euo pipefail

# Async Agent System Status Dashboard
# Displays comprehensive status of all agents and tasks

WORKSPACE="${ASYNC_AGENT_WORKSPACE:-$HOME/async-workspace}"
PID_FILE="$WORKSPACE/.agent-communication/pids.txt"
HEARTBEAT_DIR="$WORKSPACE/.agent-communication/heartbeats"
QUEUE_DIR="$WORKSPACE/.agent-communication/queues"
COMPLETED_DIR="$WORKSPACE/.agent-communication/completed"
PAUSE_MARKER="$WORKSPACE/.agent-communication/.paused"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Parse arguments
SHOW_TASKS=false
SHOW_RECENT=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --tasks)
            SHOW_TASKS=true
            shift
            ;;
        --recent)
            SHOW_RECENT=$2
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --tasks          Show detailed task breakdown"
            echo "  --recent N       Show last N completed tasks (default: 5)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_header() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}                   ASYNC AGENT SYSTEM STATUS${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_system_info() {
    echo -e "${BLUE}Workspace:${NC} $WORKSPACE"

    # Check if paused
    if [[ -f "$PAUSE_MARKER" ]]; then
        echo -e "${YELLOW}Status: PAUSED${NC}"
        paused_at=$(jq -r '.paused_at' "$PAUSE_MARKER" 2>/dev/null || echo "unknown")
        echo -e "${BLUE}Paused at:${NC} $paused_at"
    else
        echo -e "${GREEN}Status: RUNNING${NC}"
    fi
    echo ""
}

get_agent_status() {
    local agent=$1
    local pid=$2

    # Check if process is running
    if ! kill -0 "$pid" 2>/dev/null; then
        echo -e "${RED}DEAD${NC}"
        return
    fi

    # Check heartbeat
    if [[ -f "$HEARTBEAT_DIR/$agent" ]]; then
        local heartbeat=$(cat "$HEARTBEAT_DIR/$agent")
        local now=$(date +%s)
        local age=$((now - heartbeat))

        if [[ $age -lt 60 ]]; then
            echo -e "${GREEN}HEALTHY${NC} (heartbeat: ${age}s ago)"
        elif [[ $age -lt 90 ]]; then
            echo -e "${YELLOW}STALE${NC} (heartbeat: ${age}s ago)"
        else
            echo -e "${RED}UNRESPONSIVE${NC} (heartbeat: ${age}s ago)"
        fi
    else
        echo -e "${YELLOW}NO HEARTBEAT${NC}"
    fi
}

get_task_duration() {
    local task_file=$1
    local started_at=$(jq -r '.started_at // 0' "$task_file" 2>/dev/null)

    if [[ "$started_at" == "0" || "$started_at" == "null" ]]; then
        echo "N/A"
        return
    fi

    local now=$(date +%s)
    # Handle decimal timestamps from jq's 'now' function
    local elapsed=$((now - ${started_at%.*}))

    if [[ $elapsed -lt 60 ]]; then
        echo "${elapsed}s"
    elif [[ $elapsed -lt 3600 ]]; then
        echo "$((elapsed / 60))m $((elapsed % 60))s"
    else
        echo "$((elapsed / 3600))h $((elapsed % 3600 / 60))m"
    fi
}

print_agent_status() {
    echo -e "${CYAN}Agent Status:${NC}"
    echo -e "${CYAN}───────────────────────────────────────────────────────────────────────────────${NC}"

    if [[ ! -f "$PID_FILE" ]]; then
        echo -e "${RED}No agents running (no PID file)${NC}"
        echo ""
        return
    fi

    printf "%-12s %-8s %-s\n" "Agent" "PID" "Status"
    echo "─────────────────────────────────────────────────────────────"

    while IFS=: read -r agent pid; do
        [[ -z "$pid" ]] && continue
        local status=$(get_agent_status "$agent" "$pid")
        printf "%-12s %-8s %s\n" "$agent" "$pid" "$status"
    done < "$PID_FILE"

    echo ""
}

count_tasks_by_status() {
    local queue=$1
    local status=$2
    local count=0

    while IFS= read -r task_file; do
        [[ ! -f "$task_file" ]] && continue
        task_status=$(jq -r '.status' "$task_file" 2>/dev/null || echo "")
        if [[ "$task_status" == "$status" ]]; then
            ((count++))
        fi
    done < <(find "$queue" -name "*.json" -type f 2>/dev/null)

    echo $count
}

count_escalations() {
    local queue=$1
    local count=0

    while IFS= read -r task_file; do
        [[ ! -f "$task_file" ]] && continue
        task_type=$(jq -r '.type' "$task_file" 2>/dev/null || echo "")
        needs_review=$(jq -r '.needs_human_review // false' "$task_file" 2>/dev/null || echo "false")
        if [[ "$task_type" == "escalation" ]] || [[ "$needs_review" == "true" ]]; then
            ((count++))
        fi
    done < <(find "$queue" -name "*.json" -type f 2>/dev/null)

    echo $count
}

print_queue_status() {
    echo -e "${CYAN}Queue Status:${NC}"
    echo -e "${CYAN}───────────────────────────────────────────────────────────────────────────────${NC}"

    printf "%-12s %-10s %-12s %-10s %-s\n" "Agent" "Pending" "In Progress" "Failed" "Escalations"
    echo "────────────────────────────────────────────────────────────────────────"

    local total_pending=0
    local total_in_progress=0
    local total_failed=0
    local total_escalations=0

    for queue in "$QUEUE_DIR"/*; do
        if [[ -d "$queue" ]]; then
            local agent=$(basename "$queue")
            local pending=$(count_tasks_by_status "$queue" "pending")
            local in_progress=$(count_tasks_by_status "$queue" "in_progress")
            local failed=$(count_tasks_by_status "$queue" "failed")
            local escalations=$(count_escalations "$queue")

            total_pending=$((total_pending + pending))
            total_in_progress=$((total_in_progress + in_progress))
            total_failed=$((total_failed + failed))
            total_escalations=$((total_escalations + escalations))

            # Color code based on status
            local pending_str="$pending"
            local in_progress_str="$in_progress"
            local failed_str="$failed"
            local escalations_str="$escalations"

            [[ $in_progress -gt 0 ]] && in_progress_str="${YELLOW}$in_progress${NC}"
            [[ $failed -gt 0 ]] && failed_str="${RED}$failed${NC}"
            [[ $escalations -gt 0 ]] && escalations_str="${MAGENTA}$escalations${NC}"

            printf "%-12s %-10s %-22s %-20s %-s\n" \
                "$agent" \
                "$pending_str" \
                "$in_progress_str" \
                "$failed_str" \
                "$escalations_str"
        fi
    done

    echo "────────────────────────────────────────────────────────────────────────"
    printf "%-12s %-10s %-12s %-10s %-s\n" \
        "TOTAL" \
        "$total_pending" \
        "$total_in_progress" \
        "$total_failed" \
        "$total_escalations"

    echo ""
}

print_in_progress_tasks() {
    echo -e "${CYAN}In-Progress Tasks:${NC}"
    echo -e "${CYAN}───────────────────────────────────────────────────────────────────────────────${NC}"

    local has_in_progress=false

    # Collect all in-progress tasks
    for queue in "$QUEUE_DIR"/*; do
        if [[ -d "$queue" ]]; then
            local agent=$(basename "$queue")

            while IFS= read -r task_file; do
                [[ ! -f "$task_file" ]] && continue

                local status=$(jq -r '.status' "$task_file" 2>/dev/null)
                if [[ "$status" == "in_progress" ]]; then
                    if ! $has_in_progress; then
                        # Print header on first in-progress task found
                        printf "%-12s %-24s %-12s %-s\n" "Agent" "Task ID" "Duration" "Title"
                        echo "────────────────────────────────────────────────────────────────────────"
                        has_in_progress=true
                    fi

                    local task_id=$(jq -r '.id' "$task_file" 2>/dev/null)
                    local title=$(jq -r '.title' "$task_file" 2>/dev/null)
                    local duration=$(get_task_duration "$task_file")

                    # Truncate title if too long
                    if [[ ${#title} -gt 40 ]]; then
                        title="${title:0:37}..."
                    fi

                    printf "%-12s %-24s ${YELLOW}%-12s${NC} %-s\n" \
                        "$agent" \
                        "$task_id" \
                        "$duration" \
                        "$title"
                fi
            done < <(find "$queue" -name "*.json" -type f 2>/dev/null)
        fi
    done

    if ! $has_in_progress; then
        echo "No tasks currently in progress"
    fi

    echo ""
}

print_completed_summary() {
    echo -e "${CYAN}Completed Tasks:${NC}"
    echo -e "${CYAN}───────────────────────────────────────────────────────────────────────────────${NC}"

    local completed_count=$(ls "$COMPLETED_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "Total completed: ${GREEN}$completed_count${NC}"

    if [[ $completed_count -gt 0 ]]; then
        echo ""
        echo "Recent completions (last $SHOW_RECENT):"
        for task in $(ls -t "$COMPLETED_DIR"/*.json 2>/dev/null | head -$SHOW_RECENT); do
            task_id=$(basename "$task" .json)
            title=$(jq -r '.title' "$task" 2>/dev/null || echo "Unknown")
            completed_at=$(jq -r '.completed_at' "$task" 2>/dev/null || echo "unknown")
            echo "  • $task_id"
            echo "    $title"
            echo "    Completed: $completed_at"
            echo ""
        done
    fi
}

print_task_details() {
    if ! $SHOW_TASKS; then
        return
    fi

    echo -e "${CYAN}Detailed Task Breakdown:${NC}"
    echo -e "${CYAN}───────────────────────────────────────────────────────────────────────────────${NC}"

    for queue in "$QUEUE_DIR"/*; do
        if [[ -d "$queue" ]]; then
            local agent=$(basename "$queue")
            local task_count=$(find "$queue" -name "*.json" -type f 2>/dev/null | wc -l | tr -d ' ')

            if [[ $task_count -gt 0 ]]; then
                echo -e "${BLUE}$agent Queue:${NC}"

                while IFS= read -r task_file; do
                    [[ ! -f "$task_file" ]] && continue

                    local task_id=$(jq -r '.id' "$task_file" 2>/dev/null)
                    local title=$(jq -r '.title' "$task_file" 2>/dev/null)
                    local status=$(jq -r '.status' "$task_file" 2>/dev/null)
                    local task_type=$(jq -r '.type' "$task_file" 2>/dev/null)

                    local status_color=$NC
                    case $status in
                        "pending") status_color=$NC ;;
                        "in_progress") status_color=$YELLOW ;;
                        "failed") status_color=$RED ;;
                    esac

                    # Show duration for in-progress tasks
                    if [[ "$status" == "in_progress" ]]; then
                        local duration=$(get_task_duration "$task_file")
                        echo -e "  [${status_color}$status${NC}] $task_id - $title ${YELLOW}($duration)${NC}"
                    else
                        echo -e "  [${status_color}$status${NC}] $task_id - $title"
                    fi
                done < <(find "$queue" -name "*.json" -type f 2>/dev/null)

                echo ""
            fi
        fi
    done
}

print_footer() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Commands:"
    echo "  Monitor logs:     tail -f $WORKSPACE/.agent-communication/logs/agent-*.jsonl"
    echo "  Review escal.:    bash $(dirname "$0")/review-escalations.sh"
    echo "  Pause system:     bash $(dirname "$0")/pause-agents.sh"
    echo "  Resume system:    bash $(dirname "$0")/resume-agents.sh"
}

# Main execution
print_header
print_system_info
print_agent_status
print_queue_status
print_in_progress_tasks
print_completed_summary
print_task_details
print_footer
