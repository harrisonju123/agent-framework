#!/bin/bash
set -euo pipefail

# Aggregate Logs Script
# Combines and filters logs from all agents into a unified view

WORKSPACE="${ASYNC_AGENT_WORKSPACE:-$HOME/async-workspace}"
LOG_DIR="$WORKSPACE/.agent-communication/logs"

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Default values
SINCE=""
UNTIL=""
AGENT=""
LEVEL=""
FORMAT="table"
LIMIT=100

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
  --since TIME         Show logs since this time (e.g., "1 hour ago", "30 minutes ago")
  --until TIME         Show logs until this time
  --agent AGENT        Filter by agent name (e.g., agent-a, agent-b, watchdog)
  --level LEVEL        Filter by log level (DEBUG, INFO, WARN, ERROR)
  --format FORMAT      Output format: table (default) or json
  --limit N            Limit output to N most recent entries (default: 100, 0 for unlimited)
  --help               Show this help message

Examples:
  # Show all logs from last hour
  $0 --since "1 hour ago"

  # Show only errors
  $0 --level ERROR

  # Show only agent-b logs
  $0 --agent agent-b

  # Combine filters
  $0 --agent agent-a --level WARN --since "30 minutes ago"

  # Output as JSON for processing
  $0 --format json | jq .

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --since)
            SINCE="$2"
            shift 2
            ;;
        --until)
            UNTIL="$2"
            shift 2
            ;;
        --agent)
            AGENT="$2"
            shift 2
            ;;
        --level)
            LEVEL="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Convert time strings to Unix timestamps
time_to_timestamp() {
    local time_str="$1"
    date -j -f "%Y-%m-%dT%H:%M:%SZ" "$time_str" +%s 2>/dev/null || \
    date -d "$time_str" +%s 2>/dev/null || \
    echo "0"
}

# Parse "N units ago" format
parse_relative_time() {
    local time_str="$1"
    local now=$(date +%s)

    if [[ "$time_str" =~ ^([0-9]+)\ (second|minute|hour|day)s?\ ago$ ]]; then
        local num="${BASH_REMATCH[1]}"
        local unit="${BASH_REMATCH[2]}"

        case "$unit" in
            second) echo $((now - num)) ;;
            minute) echo $((now - num * 60)) ;;
            hour) echo $((now - num * 3600)) ;;
            day) echo $((now - num * 86400)) ;;
        esac
    else
        echo "0"
    fi
}

# Get timestamp thresholds
if [[ -n "$SINCE" ]]; then
    SINCE_TS=$(parse_relative_time "$SINCE")
    if [[ "$SINCE_TS" == "0" ]]; then
        SINCE_TS=$(time_to_timestamp "$SINCE")
    fi
else
    SINCE_TS=0
fi

if [[ -n "$UNTIL" ]]; then
    UNTIL_TS=$(parse_relative_time "$UNTIL")
    if [[ "$UNTIL_TS" == "0" ]]; then
        UNTIL_TS=$(time_to_timestamp "$UNTIL")
    fi
else
    UNTIL_TS=9999999999
fi

# Print header for table format
if [[ "$FORMAT" == "table" ]]; then
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    filters=""
    [[ -n "$SINCE" ]] && filters="$filters since: $SINCE"
    [[ -n "$UNTIL" ]] && filters="$filters until: $UNTIL"
    [[ -n "$AGENT" ]] && filters="$filters agent: $AGENT"
    [[ -n "$LEVEL" ]] && filters="$filters level: $LEVEL"
    if [[ -n "$filters" ]]; then
        echo -e "${CYAN}                    LOG AGGREGATION ($filters )${NC}"
    else
        echo -e "${CYAN}                    LOG AGGREGATION (all logs)${NC}"
    fi
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
fi

# Check if log directory exists
if [[ ! -d "$LOG_DIR" ]]; then
    echo "Error: Log directory not found: $LOG_DIR"
    exit 1
fi

# Collect all log files
log_files=("$LOG_DIR"/*.jsonl)

if [[ ${#log_files[@]} -eq 0 || ! -f "${log_files[0]}" ]]; then
    echo "No log files found in $LOG_DIR"
    exit 0
fi

# Process logs
{
    for log_file in "${log_files[@]}"; do
        [[ ! -f "$log_file" ]] && continue

        while IFS= read -r line; do
            # Skip empty lines
            [[ -z "$line" ]] && continue

            # Parse JSON
            timestamp=$(echo "$line" | jq -r '.timestamp // empty' 2>/dev/null)
            agent=$(echo "$line" | jq -r '.agent // empty' 2>/dev/null)
            level=$(echo "$line" | jq -r '.level // empty' 2>/dev/null)
            message=$(echo "$line" | jq -r '.message // empty' 2>/dev/null)

            # Skip if parsing failed
            [[ -z "$timestamp" ]] && continue

            # Apply agent filter
            if [[ -n "$AGENT" && "$agent" != "$AGENT" ]]; then
                continue
            fi

            # Apply level filter
            if [[ -n "$LEVEL" && "$level" != "$LEVEL" ]]; then
                continue
            fi

            # Apply time range filter
            if [[ "$SINCE_TS" != "0" || "$UNTIL_TS" != "9999999999" ]]; then
                log_ts=$(time_to_timestamp "$timestamp")
                if [[ "$log_ts" == "0" ]]; then
                    continue
                fi
                if [[ "$log_ts" -lt "$SINCE_TS" || "$log_ts" -gt "$UNTIL_TS" ]]; then
                    continue
                fi
            fi

            # Output based on format
            if [[ "$FORMAT" == "json" ]]; then
                echo "$line"
            else
                # Color code by level
                level_color=$NC
                case "$level" in
                    ERROR) level_color=$RED ;;
                    WARN) level_color=$YELLOW ;;
                    INFO) level_color=$GREEN ;;
                esac

                # Truncate message if too long
                if [[ ${#message} -gt 80 ]]; then
                    message="${message:0:77}..."
                fi

                printf "%-24s %-12s ${level_color}%-6s${NC} %s\n" \
                    "$timestamp" \
                    "$agent" \
                    "$level" \
                    "$message"
            fi
        done < "$log_file"
    done
} | sort -k1 | if [[ "$LIMIT" -gt 0 ]]; then tail -n "$LIMIT"; else cat; fi

# Print footer for table format
if [[ "$FORMAT" == "table" ]]; then
    echo ""
    echo -e "${CYAN}Showing most recent $LIMIT entries (use --limit 0 for unlimited)${NC}"
fi
