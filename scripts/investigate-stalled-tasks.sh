#!/bin/bash
# Investigate stalled tasks - diagnose why tasks aren't being processed
# Usage: ASYNC_AGENT_WORKSPACE=/path/to/workspace bash scripts/investigate-stalled-tasks.sh [--fix]
set -euo pipefail

WORKSPACE="${ASYNC_AGENT_WORKSPACE:-$HOME/async-workspace}"
QUEUE_DIR="$WORKSPACE/.agent-communication/queues"
COMPLETED_DIR="$WORKSPACE/.agent-communication/completed"

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

FIX_MODE=false
if [[ "${1:-}" == "--fix" ]]; then
    FIX_MODE=true
fi

echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}                    STALLED TASK INVESTIGATION REPORT${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "Workspace: $WORKSPACE"
echo -e "Time: $(date '+%Y-%m-%d %H:%M:%S')"
if $FIX_MODE; then
    echo -e "${YELLOW}FIX MODE ENABLED - Will attempt to fix issues${NC}"
fi
echo ""

# Track issues for summary
declare -a CRITICAL_ISSUES=()
declare -a WARNING_ISSUES=()
declare -a READY_SOON=()

# Check dependency status
check_dependency() {
    local dep=$1

    # Check completed
    if [[ -f "$COMPLETED_DIR/$dep.json" ]]; then
        echo "completed"
        return
    fi

    # Check all queues for pending/in_progress
    for agent_queue in "$QUEUE_DIR"/*/; do
        if [[ -f "$agent_queue/$dep.json" ]]; then
            local status=$(jq -r '.status // "unknown"' "$agent_queue/$dep.json" 2>/dev/null)
            echo "$status"
            return
        fi
    done

    echo "missing"
}

# Calculate backoff time remaining
get_backoff_info() {
    local task_file=$1
    local retry_count=$(jq -r '.retry_count // 0' "$task_file" 2>/dev/null)
    local last_failed_at=$(jq -r '.last_failed_at // 0' "$task_file" 2>/dev/null)

    if [[ $retry_count -eq 0 || "$last_failed_at" == "0" || "$last_failed_at" == "null" ]]; then
        echo "none|0|0"
        return
    fi

    local now=$(date +%s)
    local backoff=$((30 * (2 ** (retry_count - 1))))
    if [[ $backoff -gt 240 ]]; then
        backoff=240
    fi

    local time_since_failure=$((now - ${last_failed_at%.*}))
    local time_remaining=$((backoff - time_since_failure))

    if [[ $time_remaining -lt 0 ]]; then
        time_remaining=0
    fi

    echo "$retry_count|$backoff|$time_remaining"
}

# Format time nicely
format_time() {
    local seconds=$1
    if [[ $seconds -lt 60 ]]; then
        echo "${seconds}s"
    elif [[ $seconds -lt 3600 ]]; then
        echo "$((seconds / 60))m $((seconds % 60))s"
    else
        echo "$((seconds / 3600))h $((seconds % 3600 / 60))m"
    fi
}

# Analyze a single task
analyze_task() {
    local task_file=$1
    local agent=$(basename $(dirname "$task_file"))
    local task_id=$(jq -r '.id // "unknown"' "$task_file" 2>/dev/null)
    local status=$(jq -r '.status // "unknown"' "$task_file" 2>/dev/null)
    local title=$(jq -r '.title // "No title"' "$task_file" 2>/dev/null | cut -c1-60)
    local priority=$(jq -r '.priority // 99' "$task_file" 2>/dev/null)

    # Skip completed tasks
    [[ "$status" == "completed" ]] && return

    local issues=()
    local severity="green"
    local suggestions=()

    # Check for "blocked" status (should be pending)
    if [[ "$status" == "blocked" ]]; then
        issues+=("Status is 'blocked' - agent runner only processes 'pending' tasks")
        suggestions+=("Fix: Change status to 'pending'")
        severity="red"
        CRITICAL_ISSUES+=("$agent/$task_id: Status 'blocked' should be 'pending'")
    fi

    # Check backoff
    local backoff_info=$(get_backoff_info "$task_file")
    local retry_count=$(echo "$backoff_info" | cut -d'|' -f1)
    local backoff_time=$(echo "$backoff_info" | cut -d'|' -f2)
    local time_remaining=$(echo "$backoff_info" | cut -d'|' -f3)

    # Convert "none" to 0 for numeric comparison
    if [[ "$retry_count" == "none" ]]; then
        retry_count=0
    fi

    if [[ $retry_count -gt 0 ]]; then
        if [[ $time_remaining -gt 0 ]]; then
            issues+=("In exponential backoff: $retry_count retries, waiting $(format_time $time_remaining)")
            if [[ $retry_count -ge 5 ]]; then
                severity="red"
                CRITICAL_ISSUES+=("$agent/$task_id: Failed $retry_count times, still in backoff")
            else
                severity="yellow"
                WARNING_ISSUES+=("$agent/$task_id: In backoff ($retry_count retries)")
            fi
        else
            issues+=("Backoff elapsed: $retry_count retries, ready to retry")
            READY_SOON+=("$agent/$task_id: Backoff elapsed, ready for retry")
        fi
    fi

    # Check dependencies
    local deps=$(jq -r '.depends_on // [] | .[]' "$task_file" 2>/dev/null)
    local blocked_by=$(jq -r '.blocked_by // ""' "$task_file" 2>/dev/null)

    if [[ -n "$deps" ]]; then
        while IFS= read -r dep; do
            [[ -z "$dep" ]] && continue
            local dep_status=$(check_dependency "$dep")
            case "$dep_status" in
                "completed")
                    # Dependency satisfied
                    ;;
                "pending"|"in_progress")
                    issues+=("Waiting on dependency '$dep' (status: $dep_status)")
                    if [[ "$severity" != "red" ]]; then
                        severity="yellow"
                    fi
                    WARNING_ISSUES+=("$agent/$task_id: Waiting on $dep ($dep_status)")
                    ;;
                "missing")
                    issues+=("MISSING DEPENDENCY: '$dep' not found anywhere!")
                    suggestions+=("Fix: Create completed marker for '$dep' or remove from depends_on")
                    severity="red"
                    CRITICAL_ISSUES+=("$agent/$task_id: Missing dependency '$dep' BLOCKS FOREVER!")
                    ;;
                *)
                    issues+=("Dependency '$dep' has unusual status: $dep_status")
                    if [[ "$severity" != "red" ]]; then
                        severity="yellow"
                    fi
                    WARNING_ISSUES+=("$agent/$task_id: Dep '$dep' has status '$dep_status'")
                    ;;
            esac
        done <<< "$deps"
    fi

    # Check blocked_by field (different from depends_on)
    if [[ -n "$blocked_by" && "$blocked_by" != "null" ]]; then
        issues+=("Blocked by: $blocked_by")
        severity="yellow"
        WARNING_ISSUES+=("$agent/$task_id: Blocked by '$blocked_by'")
    fi

    # Only print if there are issues
    if [[ ${#issues[@]} -gt 0 ]]; then
        case "$severity" in
            "red") color=$RED ;;
            "yellow") color=$YELLOW ;;
            "green") color=$GREEN ;;
            *) color=$NC ;;
        esac

        echo -e "${color}───────────────────────────────────────────────────────────────────────────────${NC}"
        echo -e "${BOLD}Task:${NC} $task_id"
        echo -e "  ${CYAN}Queue:${NC} $agent | ${CYAN}Status:${NC} $status | ${CYAN}Priority:${NC} $priority"
        echo -e "  ${CYAN}Title:${NC} $title"

        echo -e "  ${BOLD}Issues:${NC}"
        for issue in "${issues[@]}"; do
            echo -e "    ${color}•${NC} $issue"
        done

        if [[ ${#suggestions[@]} -gt 0 ]]; then
            echo -e "  ${BOLD}Suggestions:${NC}"
            for suggestion in "${suggestions[@]}"; do
                echo -e "    ${GREEN}→${NC} $suggestion"
            done
        fi

        # Apply fixes if in fix mode
        if $FIX_MODE; then
            if [[ "$status" == "blocked" ]]; then
                echo -e "  ${YELLOW}FIXING: Changing status to 'pending'...${NC}"
                local tmp_file="$task_file.tmp"
                jq '.status = "pending"' "$task_file" > "$tmp_file"
                mv "$tmp_file" "$task_file"
                echo -e "  ${GREEN}FIXED!${NC}"
            fi
        fi

        echo ""
    fi
}

# Analyze all queues
echo -e "${BOLD}Analyzing tasks in all queues...${NC}"
echo ""

for agent_queue in "$QUEUE_DIR"/*/; do
    agent=$(basename "$agent_queue")
    task_count=$(find "$agent_queue" -name "*.json" -type f 2>/dev/null | wc -l | tr -d ' ')

    if [[ $task_count -gt 0 ]]; then
        echo -e "${BLUE}═══ Queue: $agent ($task_count tasks) ═══${NC}"

        while IFS= read -r task_file; do
            [[ ! -f "$task_file" ]] && continue
            analyze_task "$task_file"
        done < <(find "$agent_queue" -name "*.json" -type f 2>/dev/null | sort)
    fi
done

# Print summary
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}                              SUMMARY${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"

if [[ ${#CRITICAL_ISSUES[@]} -gt 0 ]]; then
    echo -e "${RED}${BOLD}CRITICAL ISSUES (${#CRITICAL_ISSUES[@]}):${NC}"
    for issue in "${CRITICAL_ISSUES[@]}"; do
        echo -e "  ${RED}✗${NC} $issue"
    done
    echo ""
fi

if [[ ${#WARNING_ISSUES[@]} -gt 0 ]]; then
    echo -e "${YELLOW}${BOLD}WARNINGS (${#WARNING_ISSUES[@]}):${NC}"
    for issue in "${WARNING_ISSUES[@]}"; do
        echo -e "  ${YELLOW}⚠${NC} $issue"
    done
    echo ""
fi

if [[ ${#READY_SOON[@]} -gt 0 ]]; then
    echo -e "${GREEN}${BOLD}READY FOR PROCESSING (${#READY_SOON[@]}):${NC}"
    for issue in "${READY_SOON[@]}"; do
        echo -e "  ${GREEN}✓${NC} $issue"
    done
    echo ""
fi

total_issues=$((${#CRITICAL_ISSUES[@]} + ${#WARNING_ISSUES[@]}))
if [[ $total_issues -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}No stalled tasks found! All tasks are processing normally.${NC}"
fi

echo ""
echo -e "${BOLD}Quick Fixes:${NC}"
echo "  • Change 'blocked' to 'pending': jq '.status = \"pending\"' task.json > tmp && mv tmp task.json"
echo "  • Reset retry count: jq '.retry_count = 0 | del(.last_failed_at)' task.json > tmp && mv tmp task.json"
echo "  • Create completed marker: echo '{\"id\":\"task-id\",\"status\":\"completed\"}' > completed/task-id.json"
echo ""
echo "  Or run with --fix flag to auto-fix simple issues:"
echo "  ASYNC_AGENT_WORKSPACE=$WORKSPACE bash $0 --fix"
