#!/bin/bash
set -euo pipefail

# Escalation Review Interface
# Allows humans to review and resolve escalated tasks

WORKSPACE="${ASYNC_AGENT_WORKSPACE:-$HOME/async-workspace}"
QUEUE_DIR="$WORKSPACE/.agent-communication/queues"
LOG_FILE="$WORKSPACE/.agent-communication/logs/human-review.jsonl"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log() {
    local level=$1
    shift
    local msg="$*"
    local timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "{\"timestamp\":\"$timestamp\",\"agent\":\"human-reviewer\",\"level\":\"$level\",\"message\":\"$msg\"}" >> "$LOG_FILE"
}

print_header() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}                    ESCALATION REVIEW INTERFACE${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

find_escalations() {
    # Find all escalation tasks across all queues
    local escalations=()

    for queue in "$QUEUE_DIR"/*; do
        if [[ -d "$queue" ]]; then
            while IFS= read -r task_file; do
                [[ ! -f "$task_file" ]] && continue

                # Check if it's an escalation task
                task_type=$(jq -r '.type' "$task_file" 2>/dev/null || echo "")
                needs_review=$(jq -r '.needs_human_review // false' "$task_file" 2>/dev/null || echo "false")

                if [[ "$task_type" == "escalation" ]] || [[ "$needs_review" == "true" ]]; then
                    escalations+=("$task_file")
                fi
            done < <(find "$queue" -name "*.json" -type f 2>/dev/null)
        fi
    done

    printf '%s\n' "${escalations[@]}"
}

display_escalation() {
    local task_file=$1
    local index=$2

    # Read task details
    local task_id=$(jq -r '.id' "$task_file")
    local title=$(jq -r '.title' "$task_file")
    local description=$(jq -r '.description' "$task_file")
    local created_by=$(jq -r '.created_by' "$task_file")
    local created_at=$(jq -r '.created_at' "$task_file")
    local failed_task_id=$(jq -r '.failed_task_id // "N/A"' "$task_file")

    echo -e "${YELLOW}───────────────────────────────────────────────────────────────────────────────${NC}"
    echo -e "${YELLOW}Escalation #$index${NC}"
    echo -e "${YELLOW}───────────────────────────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}Task ID:${NC}        $task_id"
    echo -e "${BLUE}Title:${NC}          $title"
    echo -e "${BLUE}Created by:${NC}     $created_by"
    echo -e "${BLUE}Created at:${NC}     $created_at"
    echo -e "${BLUE}Failed Task:${NC}    $failed_task_id"
    echo ""
    echo -e "${BLUE}Description:${NC}"
    echo "$description" | sed 's/^/  /'
    echo ""

    # Show original failed task if available
    if [[ "$failed_task_id" != "N/A" ]]; then
        local original_task_file=$(jq -r '.context.original_task_file // ""' "$task_file")
        if [[ -n "$original_task_file" && -f "$original_task_file" ]]; then
            echo -e "${BLUE}Original Task Details:${NC}"
            echo -e "  Status: $(jq -r '.status' "$original_task_file")"
            echo -e "  Retry Count: $(jq -r '.retry_count // 0' "$original_task_file")"
            echo ""
        fi
    fi
}

resolve_escalation() {
    local task_file=$1
    local resolution=$2
    local notes=$3

    local task_id=$(basename "$task_file" .json)
    local tmp_file="$task_file.tmp"

    # Mark escalation as resolved
    jq --arg resolution "$resolution" --arg notes "$notes" \
        '.status = "completed" | .resolved_at = now | .resolved_by = "human" | .resolution = $resolution | .resolution_notes = $notes' \
        "$task_file" > "$tmp_file"

    # Move to completed
    local completed_file="$WORKSPACE/.agent-communication/completed/$task_id.json"
    mv "$tmp_file" "$completed_file"

    # Remove from queue
    rm -f "$task_file"

    log "INFO" "Resolved escalation $task_id: $resolution"
    echo -e "${GREEN}✓ Escalation resolved and moved to completed/${NC}"
}

retry_failed_task() {
    local task_file=$1

    local failed_task_id=$(jq -r '.failed_task_id // ""' "$task_file")
    if [[ -z "$failed_task_id" || "$failed_task_id" == "null" ]]; then
        echo -e "${RED}Error: No failed task ID found in escalation${NC}"
        return 1
    fi

    # Find the original failed task
    local original_task_file=$(jq -r '.context.original_task_file // ""' "$task_file")
    if [[ ! -f "$original_task_file" ]]; then
        echo -e "${RED}Error: Original task file not found: $original_task_file${NC}"
        return 1
    fi

    # Reset task to pending with retry count reset
    local tmp_file="$original_task_file.tmp"
    jq '.status = "pending" | del(.started_at) | del(.started_by) | del(.failed_at) | del(.failed_by) | .retry_count = 0 | del(.last_failed_at)' \
        "$original_task_file" > "$tmp_file"
    mv "$tmp_file" "$original_task_file"

    echo -e "${GREEN}✓ Failed task reset to pending and will be retried${NC}"
    log "INFO" "Reset failed task $failed_task_id to pending (human intervention)"
}

review_escalation_interactive() {
    local task_file=$1
    local index=$2

    while true; do
        display_escalation "$task_file" "$index"

        echo -e "${CYAN}Actions:${NC}"
        echo "  [1] Resolve as 'Fixed' - Issue has been addressed"
        echo "  [2] Resolve as 'Wont Fix' - Issue acknowledged but not fixing"
        echo "  [3] Retry Failed Task - Reset failed task to retry"
        echo "  [4] Custom Resolution - Provide custom resolution notes"
        echo "  [5] Skip - Review later"
        echo ""
        read -p "Choose action [1-5]: " action

        case $action in
            1)
                read -p "Resolution notes (optional): " notes
                resolve_escalation "$task_file" "fixed" "$notes"
                return 0
                ;;
            2)
                read -p "Reason for not fixing: " reason
                resolve_escalation "$task_file" "wont_fix" "$reason"
                return 0
                ;;
            3)
                retry_failed_task "$task_file"
                read -p "Mark escalation as resolved? [y/N]: " mark_resolved
                if [[ "$mark_resolved" =~ ^[Yy]$ ]]; then
                    resolve_escalation "$task_file" "retrying" "Task reset to pending for retry"
                    return 0
                fi
                ;;
            4)
                read -p "Resolution status: " custom_status
                read -p "Resolution notes: " custom_notes
                resolve_escalation "$task_file" "$custom_status" "$custom_notes"
                return 0
                ;;
            5)
                echo -e "${YELLOW}Skipping escalation${NC}"
                return 1
                ;;
            *)
                echo -e "${RED}Invalid choice. Please select 1-5.${NC}"
                echo ""
                ;;
        esac
    done
}

# Main execution
print_header

# Find all escalations
echo "Searching for escalations..."
escalations=($(find_escalations))

if [[ ${#escalations[@]} -eq 0 ]]; then
    echo -e "${GREEN}No escalations found! All tasks are running smoothly.${NC}"
    echo ""
    exit 0
fi

echo -e "${YELLOW}Found ${#escalations[@]} escalation(s) requiring review${NC}"
echo ""

# Review each escalation
reviewed_count=0
skipped_count=0

for i in "${!escalations[@]}"; do
    task_file="${escalations[$i]}"
    index=$((i + 1))

    if review_escalation_interactive "$task_file" "$index"; then
        ((reviewed_count++))
    else
        ((skipped_count++))
    fi

    echo ""

    # Ask if user wants to continue
    if [[ $index -lt ${#escalations[@]} ]]; then
        read -p "Continue to next escalation? [Y/n]: " continue_review
        if [[ "$continue_review" =~ ^[Nn]$ ]]; then
            echo -e "${YELLOW}Review session ended${NC}"
            break
        fi
        echo ""
    fi
done

# Summary
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}Review Summary${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "Total escalations: ${#escalations[@]}"
echo -e "Reviewed: ${GREEN}$reviewed_count${NC}"
echo -e "Skipped: ${YELLOW}$skipped_count${NC}"
echo ""

log "INFO" "Review session completed: $reviewed_count resolved, $skipped_count skipped"
