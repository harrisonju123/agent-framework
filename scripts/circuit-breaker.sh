#!/bin/bash
# Circuit Breaker - Detects and prevents infinite loops in async agent system
# Usage: ASYNC_AGENT_WORKSPACE=/path bash scripts/circuit-breaker.sh [--fix]
set -euo pipefail

WORKSPACE="${ASYNC_AGENT_WORKSPACE:-$HOME/async-workspace}"
FIX_MODE=false
if [[ "${1:-}" == "--fix" ]]; then
    FIX_MODE=true
fi

# Thresholds (configurable via environment variables)
MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-100}"  # Max tasks per agent queue
MAX_ESCALATIONS="${MAX_ESCALATIONS:-50}"  # Max total escalations
MAX_TASK_AGE_DAYS="${MAX_TASK_AGE_DAYS:-7}"  # Archive tasks older than N days
MAX_CIRCULAR_DEPS="${MAX_CIRCULAR_DEPS:-5}"  # Max depth for circular dependency check

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}                        CIRCUIT BREAKER ANALYSIS${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

EXIT_CODE=0

# 1. Check queue sizes
echo -e "${BOLD}[1] Queue Size Check${NC}"
for agent_queue in "$WORKSPACE/.agent-communication/queues"/*/; do
    agent=$(basename "$agent_queue")
    count=$(find "$agent_queue" -name "*.json" -type f 2>/dev/null | wc -l | tr -d ' ')

    if [[ $count -gt $MAX_QUEUE_SIZE ]]; then
        echo -e "  ${RED}✗ $agent: $count tasks (exceeds limit: $MAX_QUEUE_SIZE)${NC}"
        EXIT_CODE=1
        if $FIX_MODE; then
            echo -e "    ${YELLOW}FIX: Archiving oldest tasks...${NC}"
            find "$agent_queue" -name "*.json" -type f -mtime +$MAX_TASK_AGE_DAYS -exec mv {} "$WORKSPACE/.agent-communication/archived/" \; 2>/dev/null || true
        fi
    else
        echo -e "  ${GREEN}✓ $agent: $count tasks${NC}"
    fi
done
echo ""

# 2. Check escalation count
echo -e "${BOLD}[2] Escalation Count Check${NC}"
escalation_count=$(find "$WORKSPACE/.agent-communication/queues" -name "escalation-*.json" 2>/dev/null | wc -l | tr -d ' ')
if [[ $escalation_count -gt $MAX_ESCALATIONS ]]; then
    echo -e "  ${RED}✗ $escalation_count escalations (exceeds limit: $MAX_ESCALATIONS)${NC}"
    echo -e "    ${YELLOW}WARNING: Too many escalations - system may need human intervention${NC}"
    EXIT_CODE=1
else
    echo -e "  ${GREEN}✓ $escalation_count escalations${NC}"
fi
echo ""

# 3. Check for escalation retry loops
echo -e "${BOLD}[3] Escalation Retry Loop Check${NC}"
escalation_retries=$(find "$WORKSPACE/.agent-communication/queues" -name "escalation-*.json" -exec jq -r 'select(.retry_count >= 1) | .id' {} \; 2>/dev/null | wc -l | tr -d ' ')
if [[ $escalation_retries -gt 0 ]]; then
    echo -e "  ${RED}✗ Found $escalation_retries escalations with retries (escalations should not retry!)${NC}"
    EXIT_CODE=1
    if $FIX_MODE; then
        echo -e "    ${YELLOW}FIX: Marking escalations as needs_human_review...${NC}"
        find "$WORKSPACE/.agent-communication/queues" -name "escalation-*.json" -exec sh -c '
            f="$1"
            jq ".retry_count = 0 | .needs_human_review = true | .status = \"pending\"" "$f" > "$f.tmp" && mv "$f.tmp" "$f"
        ' _ {} \; 2>/dev/null || true
    fi
else
    echo -e "  ${GREEN}✓ No escalations with retries${NC}"
fi
echo ""

# 4. Check for circular dependencies
echo -e "${BOLD}[4] Circular Dependency Check${NC}"
check_circular_deps() {
    local task_id=$1
    local depth=$2
    local visited=$3

    if [[ $depth -gt $MAX_CIRCULAR_DEPS ]]; then
        return 1  # Too deep, probably circular
    fi

    if [[ "$visited" =~ "$task_id" ]]; then
        echo "CIRCULAR: $visited -> $task_id"
        return 1
    fi

    # Find task file
    local task_file=$(find "$WORKSPACE/.agent-communication/queues" -name "$task_id.json" 2>/dev/null | head -1)
    if [[ -z "$task_file" ]]; then
        return 0  # Task doesn't exist or is completed
    fi

    # Check dependencies
    local deps=$(jq -r '.depends_on[]? // empty' "$task_file" 2>/dev/null)
    for dep in $deps; do
        if ! check_circular_deps "$dep" $((depth + 1)) "$visited -> $task_id"; then
            return 1
        fi
    done

    return 0
}

circular_found=0
while IFS= read -r task_file; do
    task_id=$(jq -r '.id' "$task_file" 2>/dev/null)
    if ! check_circular_deps "$task_id" 0 "" 2>&1 | grep -q "CIRCULAR"; then
        :
    else
        echo -e "  ${RED}✗ Circular dependency detected for task: $task_id${NC}"
        circular_found=1
        EXIT_CODE=1
    fi
done < <(find "$WORKSPACE/.agent-communication/queues" -name "*.json" -type f 2>/dev/null)

if [[ $circular_found -eq 0 ]]; then
    echo -e "  ${GREEN}✓ No circular dependencies detected${NC}"
fi
echo ""

# 5. Check for task creation rate (potential cascade)
echo -e "${BOLD}[5] Task Creation Rate Check${NC}"
recent_tasks=$(find "$WORKSPACE/.agent-communication/queues" -name "*.json" -type f -mmin -5 2>/dev/null | wc -l | tr -d ' ')
if [[ $recent_tasks -gt 50 ]]; then
    echo -e "  ${YELLOW}⚠ $recent_tasks tasks created in last 5 minutes (possible cascade)${NC}"
    echo -e "    ${YELLOW}Monitor closely for exponential growth${NC}"
else
    echo -e "  ${GREEN}✓ Task creation rate normal: $recent_tasks tasks in last 5 minutes${NC}"
fi
echo ""

# 6. Check for stuck tasks (same task retrying repeatedly)
echo -e "${BOLD}[6] Stuck Task Check${NC}"
stuck_count=0
while IFS= read -r task_file; do
    retry_count=$(jq -r '.retry_count // 0' "$task_file" 2>/dev/null)
    if [[ $retry_count -ge 3 ]]; then
        task_id=$(jq -r '.id' "$task_file" 2>/dev/null)
        task_type=$(jq -r '.type' "$task_file" 2>/dev/null)
        echo -e "  ${YELLOW}⚠ Task $task_id (type: $task_type) has $retry_count retries${NC}"
        stuck_count=$((stuck_count + 1))
    fi
done < <(find "$WORKSPACE/.agent-communication/queues" -name "*.json" -not -name "escalation-*" -type f 2>/dev/null)

if [[ $stuck_count -eq 0 ]]; then
    echo -e "  ${GREEN}✓ No stuck tasks (3+ retries)${NC}"
else
    echo -e "  ${YELLOW}⚠ Found $stuck_count tasks with 3+ retries${NC}"
fi
echo ""

# 7. Check for duplicate task IDs
echo -e "${BOLD}[7] Duplicate Task ID Check${NC}"
duplicates=$(find "$WORKSPACE/.agent-communication/queues" -name "*.json" -type f -exec basename {} \; 2>/dev/null | sort | uniq -d | wc -l | tr -d ' ')
if [[ $duplicates -gt 0 ]]; then
    echo -e "  ${RED}✗ Found $duplicates duplicate task IDs across queues${NC}"
    find "$WORKSPACE/.agent-communication/queues" -name "*.json" -type f -exec basename {} \; 2>/dev/null | sort | uniq -d | while read dup; do
        echo -e "    ${RED}Duplicate: $dup${NC}"
    done
    EXIT_CODE=1
else
    echo -e "  ${GREEN}✓ No duplicate task IDs${NC}"
fi
echo ""

# 8. Check for old stale tasks
echo -e "${BOLD}[8] Stale Task Check${NC}"
old_tasks=$(find "$WORKSPACE/.agent-communication/queues" -name "*.json" -type f -mtime +$MAX_TASK_AGE_DAYS 2>/dev/null | wc -l | tr -d ' ')
if [[ $old_tasks -gt 0 ]]; then
    echo -e "  ${YELLOW}⚠ Found $old_tasks tasks older than $MAX_TASK_AGE_DAYS days${NC}"
    if $FIX_MODE; then
        echo -e "    ${YELLOW}FIX: Archiving old tasks...${NC}"
        mkdir -p "$WORKSPACE/.agent-communication/archived"
        find "$WORKSPACE/.agent-communication/queues" -name "*.json" -type f -mtime +$MAX_TASK_AGE_DAYS -exec mv {} "$WORKSPACE/.agent-communication/archived/" \; 2>/dev/null || true
    fi
else
    echo -e "  ${GREEN}✓ No stale tasks${NC}"
fi
echo ""

# Summary
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}                              SUMMARY${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════════════════════${NC}"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}✓ All checks passed - no infinite loop risk detected${NC}"
else
    echo -e "${RED}${BOLD}✗ Circuit breaker detected potential issues${NC}"
    echo ""
    echo -e "${YELLOW}Recommendations:${NC}"
    echo "  1. Review escalations: ls -la $WORKSPACE/.agent-communication/queues/agent-d/escalation-*"
    echo "  2. Check stuck tasks: bash scripts/investigate-stalled-tasks.sh"
    echo "  3. Run with --fix to auto-resolve some issues"
    echo "  4. Consider stopping agents if growth is exponential"
fi

exit $EXIT_CODE
