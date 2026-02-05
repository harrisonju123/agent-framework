# Fix: MCP Subprocess Race Condition and Duplicate Tools

## Problem Summary

When subprocess agents spawned via the Orchestrator, they failed with a 400 error: `"tools: Tool names must be unique"`. This was caused by two issues:

1. **Shared Cache File Race Condition**: Multiple processes were writing to the same MCP config cache file (~/.cache/agent-framework/mcp-config-expanded.json), causing race conditions where incomplete configs overwrote valid ones.

2. **Global MCP Config Conflict**: Claude CLI was loading BOTH the global MCP config (~/.claude/mcp_settings.json) AND the project-specific config, causing duplicate tool registrations.

## Solution Implemented

### 1. Process-Specific MCP Cache Files

**File**: `src/agent_framework/llm/claude_cli_backend.py`

**Changes**:
- Cache files now include PID and environment hash: `mcp-config-{PID}-{HASH}.json`
- Each process gets its own cache file, eliminating race conditions
- Hash ensures processes with different credentials get separate caches
- Added automatic cleanup of stale cache files from terminated processes

**Key methods added/modified**:
- `_expand_mcp_config()`: Now creates process-specific cache with env hash
- `_collect_env_vars()`: Collects all ${VAR} references in config
- `_expand_env_vars_recursive()`: Warns about undefined env vars instead of silently failing
- `_cleanup_stale_cache_files()`: Removes cache files from dead processes

### 2. Environment Variable Propagation

**File**: `src/agent_framework/core/orchestrator.py`

**Changes**:
- Explicitly passes MCP-related env vars to subprocess agents
- Ensures subprocesses have all required credentials

**Environment variables passed**:
- GITHUB_TOKEN
- JIRA_URL
- JIRA_EMAIL
- JIRA_API_TOKEN
- JIRA_SERVER

### 3. Strict MCP Config Mode

**File**: `src/agent_framework/llm/claude_cli_backend.py`

**Changes**:
- Added `--strict-mcp-config` flag to Claude CLI command
- Prevents loading of global ~/.claude/mcp_settings.json
- Ensures only project-specific MCP servers are loaded

## Verification

### Test Results

```bash
$ python test_subprocess.py
✓ Success: True
✓ Model: claude-haiku-4-5-20251001
✓ Latency: 2778ms
✓ Finish reason: stop

✅ Response:
   Subprocess test successful!
```

### Cache File Verification

```bash
$ ls -la ~/.cache/agent-framework/mcp-config-*.json
-rw-r--r--  1 hju  staff  1096 Feb  5 00:08 mcp-config-19514-f2cc4cb4.json
```

Each process creates its own cache file with:
- Process ID (19514)
- Environment hash (f2cc4cb4) - ensures different credentials = different cache

## Benefits

1. **No Race Conditions**: Each process has its own cache file
2. **Credential Isolation**: Different environments use different caches (via hash)
3. **Automatic Cleanup**: Stale caches removed on backend initialization
4. **Better Debugging**: Warning logs when env vars are missing
5. **No Duplicate Tools**: --strict-mcp-config prevents global config conflicts

## Backward Compatibility

- Main agent behavior unchanged
- If env vars missing, warnings logged but no crash
- Cache isolation prevents interference between processes
- Old shared cache files (mcp-config-expanded.json) can be safely deleted

## Files Modified

1. `src/agent_framework/llm/claude_cli_backend.py` (lines 1-350)
   - Added hashlib import
   - Added Set type hint
   - Modified `__init__` to call cleanup
   - Rewrote `_expand_mcp_config()` for process-specific caching
   - Added `_collect_env_vars()` method
   - Enhanced `_expand_env_vars_recursive()` with validation
   - Added `_cleanup_stale_cache_files()` method
   - Added `--strict-mcp-config` flag to command

2. `src/agent_framework/core/orchestrator.py` (lines 135-150)
   - Added MCP env var list
   - Added loop to pass MCP env vars to subprocesses

## Testing

Run the test script:
```bash
# Set required env vars
export GITHUB_TOKEN="your-token"
export JIRA_URL="https://your-jira.atlassian.net"
export JIRA_EMAIL="your-email@example.com"
export JIRA_API_TOKEN="your-jira-token"
export JIRA_SERVER="your-jira.atlassian.net"

# Run test
python test_subprocess.py
```

Expected: No 400 error, successful response from Claude CLI
