# MCP Architecture

This document describes the technical architecture of MCP (Model Context Protocol) integration in the agent framework.

## Overview

The agent framework uses MCP servers to provide real-time JIRA and GitHub access to Claude agents during task execution. This replaces the previous post-LLM workflow where integrations were only accessible after agent completion.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Framework                       │
│                                                          │
│  ┌─────────────┐                                        │
│  │ run_agent.py│                                        │
│  └──────┬──────┘                                        │
│         │                                                │
│         ├─ Creates ClaudeCLIBackend                     │
│         │  (with --mcp-config flag)                     │
│         │                                                │
│         ├─ Creates Agent                                │
│         │  (with mcp_enabled=true)                      │
│         │                                                │
│         └─ Agent._build_prompt()                        │
│            Adds MCP tool guidance to prompt             │
│                                                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ Spawns subprocess
                     ↓
         ┌───────────────────────┐
         │   Claude CLI Agent    │
         │   (with MCPs loaded)  │
         └───────┬───────────────┘
                 │
                 │ Uses MCP tools during execution
                 │
        ┌────────┴────────┐
        │                 │
        ↓                 ↓
  ┌──────────┐      ┌──────────┐
  │ JIRA MCP │      │GitHub MCP│
  │  Server  │      │  Server  │
  └────┬─────┘      └────┬─────┘
       │                 │
       ↓                 ↓
  ┌──────────┐      ┌──────────┐
  │   JIRA   │      │  GitHub  │
  │   API    │      │   API    │
  └──────────┘      └──────────┘
```

## Components

### 1. ClaudeCLIBackend

**File:** `src/agent_framework/llm/claude_cli_backend.py`

**Responsibilities:**
- Spawn Claude CLI subprocess with `--mcp-config` flag
- Expand environment variables in MCP config before passing to CLI
- Write expanded config to `~/.cache/agent-framework/mcp-config-expanded.json`

**Key Methods:**
- `__init__(mcp_config_path)` - Accept MCP config path
- `_expand_mcp_config()` - Expand `${VAR}` placeholders with env vars
- `_expand_env_vars_recursive()` - Recursively traverse and expand config
- `complete()` - Pass `--mcp-config` flag to Claude CLI

**Why Environment Variable Expansion?**

Claude CLI expects literal values, not `${VAR}` placeholders. The backend expands these before starting the CLI subprocess.

### 2. Agent

**File:** `src/agent_framework/core/agent.py`

**Responsibilities:**
- Store `mcp_enabled` flag and JIRA/GitHub configs
- Build prompts with MCP tool guidance
- Skip post-LLM workflow when MCPs enabled

**Key Changes:**
- Added `mcp_enabled`, `jira_config`, `github_config` to `__init__()`
- Enhanced `_build_prompt()` to add MCP guidance when enabled
- Updated `_handle_success()` to skip when MCPs enabled

**Prompt Engineering:**

When MCPs are enabled, agents receive detailed guidance:
- List of available JIRA tools with examples
- List of available GitHub tools with examples
- Branch naming patterns (from `github_config.branch_pattern`)
- PR title patterns (from `github_config.pr_title_pattern`)
- Error handling instructions
- Workflow coordination (git operations vs API operations)

### 3. Configuration

**Files:**
- `src/agent_framework/core/config.py` - Schema definitions
- `config/agent-framework.yaml` - User configuration
- `config/mcp-config.json` - MCP server definitions

**Schema Changes:**

```python
class LLMConfig(BaseModel):
    # ... existing fields ...
    mcp_config_path: Optional[str] = None
    use_mcp: bool = False
```

**Validation:**

`load_config()` validates that MCPs require Claude CLI mode:
```python
if config.llm.use_mcp and config.llm.mode != "claude_cli":
    raise ValueError("MCP integration requires Claude CLI mode")
```

### 4. MCP Servers

**Location:** `mcp-servers/jira/` and `mcp-servers/github/`

**Technology:**
- TypeScript (Node.js)
- `@modelcontextprotocol/sdk` for MCP protocol
- `jira-client` for JIRA API
- `@octokit/rest` for GitHub API
- `winston` for logging

**Architecture:**

Each MCP server is a standalone Node.js process that:
1. Connects to Claude CLI via stdio transport
2. Exposes tools via MCP protocol
3. Handles tool invocations by calling external APIs
4. Returns structured JSON responses
5. Logs all operations to `logs/mcp-{service}.log`

**Tool Response Format:**

Success:
```json
{
  "success": true,
  "data": {
    // Tool-specific data
  }
}
```

Error:
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Error description"
  }
}
```

## Data Flow

### Startup Sequence

1. User runs `agent start` or `agent run`
2. CLI loads `config/agent-framework.yaml`
3. If `use_mcp: true`, validates `mode: claude_cli`
4. CLI spawns `run_agent.py` subprocess for each agent
5. `run_agent.py` creates `ClaudeCLIBackend` with MCP config path
6. Backend expands environment variables in MCP config
7. Backend writes expanded config to temp file
8. Backend spawns Claude CLI with `--mcp-config <temp_file>`
9. Claude CLI starts MCP servers defined in config
10. Agent enters polling loop

### Task Execution with MCPs

1. Agent pops task from queue
2. Agent calls `_build_prompt()` which adds MCP guidance
3. Agent sends prompt to Claude CLI (via `llm.complete()`)
4. Claude CLI agent sees MCP tools in its tool list
5. Claude agent decides to use a tool (e.g., `jira_create_epic`)
6. Claude CLI sends tool invocation to JIRA MCP server
7. JIRA MCP server calls JIRA API
8. MCP server returns result to Claude CLI
9. Claude CLI continues execution with result
10. Agent receives final response from Claude CLI
11. Agent marks task complete (no post-LLM workflow)

### Task Execution without MCPs (Legacy)

1. Agent pops task from queue
2. Agent calls `_build_prompt()` (no MCP guidance)
3. Agent sends prompt to Claude CLI
4. Claude CLI agent completes task
5. Agent receives response
6. Agent runs `_handle_success()` post-LLM workflow:
   - Creates git branch
   - Commits changes
   - Pushes to GitHub
   - Creates PR via Python `github_client`
   - Updates JIRA via Python `jira_client`

## Feature Flag

The `use_mcp` flag in configuration controls behavior:

```yaml
llm:
  use_mcp: false  # Legacy mode (post-LLM workflow)
  # OR
  use_mcp: true   # MCP mode (real-time integration)
```

**Implementation:**

```python
if self._mcp_enabled:
    logger.debug("MCPs enabled - skipping post-LLM workflow")
    return
```

This provides instant rollback without code changes.

## Environment Variables

MCPs use the same environment variables as the Python clients:

- `JIRA_SERVER` - JIRA server hostname
- `JIRA_EMAIL` - JIRA user email
- `JIRA_API_TOKEN` - JIRA API token
- `GITHUB_TOKEN` - GitHub personal access token

Variables are loaded from `.env` and expanded by `ClaudeCLIBackend._expand_mcp_config()`.

## Logging

### Agent Logs

- Location: `logs/{agent_id}.log`
- Contains: Agent execution, task processing, errors
- Format: Standard Python logging

### MCP Server Logs

- Location: `logs/mcp-jira.log`, `logs/mcp-github.log`
- Contains: Tool invocations, API calls, errors
- Format: Winston JSON logging

**Example Log Entry:**
```json
{
  "level": "info",
  "message": "Tool called: jira_create_epic",
  "args": {
    "project": "PROJ",
    "title": "User Management",
    "description": "..."
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

## Error Handling

### MCP Server Errors

MCP servers catch exceptions and return structured errors:

```typescript
catch (error: any) {
  logger.error(`Tool failed: ${name}`, { error: error.message });

  const errorResponse = {
    success: false,
    error: {
      code: error.statusCode || "UNKNOWN_ERROR",
      message: error.message || "An unknown error occurred",
    },
  };

  return {
    content: [{ type: "text", text: JSON.stringify(errorResponse) }],
    isError: true,
  };
}
```

### Agent Error Handling

Agents receive error messages from MCP tools and can:
- Retry with corrected parameters
- Report failure to user
- Create follow-up tasks

Prompts include error handling guidance:
```
If a tool call fails:
1. Read the error message carefully
2. If rate limited, wait and retry
3. If authentication failed, report failure
4. If invalid input, correct and retry
```

## Security Considerations

### 1. Credential Storage

- Credentials stored in `.env` (git-ignored)
- Never logged by MCP servers (filtered)
- Passed to MCP servers via environment variables

### 2. MCP Config Permissions

The expanded MCP config file should be readable only by the user:
```bash
chmod 600 ~/.cache/agent-framework/mcp-config-expanded.json
```

### 3. MCP Server Validation

MCP servers validate all inputs:
- Required fields present
- Field types correct
- Invalid values rejected

## Performance

### Overhead

MCP calls add overhead compared to direct API calls:
- Network: ~10-50ms (local stdio transport)
- Parsing: ~5-10ms (JSON serialization)
- Total: ~50-100ms per MCP call

For typical workflows (5-10 MCP calls per task), overhead is ~0.5-1 second.

### Optimization

Future optimizations:
- Connection pooling in MCP servers
- Batch operations (e.g., `jira_create_epic_with_subtasks`)
- Caching of frequent queries

## Comparison: MCP vs Post-LLM

| Aspect | Post-LLM Workflow | MCP Workflow |
|--------|-------------------|--------------|
| Integration Access | Only after completion | During execution |
| Transparency | Hidden magic | Visible tool calls |
| Error Handling | Framework handles | Agent handles |
| Flexibility | Fixed workflow | Agent decides workflow |
| Code Complexity | Post-processing logic | Tool guidance prompts |
| Dependencies | Python clients | Node.js MCP servers |

## Future Enhancements

### Planned

1. **Additional MCP Servers**
   - Linear integration
   - Slack notifications
   - Notion documentation

2. **Enhanced Tooling**
   - `agent check --mcps` - Health check command
   - `agent init --with-mcps` - Setup wizard

3. **Performance**
   - Connection pooling
   - Request batching
   - Response caching

### Under Consideration

1. **LiteLLM Support**
   - Python MCP client wrapper
   - Enable MCPs for API-based workflows

2. **Transaction Guarantees**
   - Rollback support for failed operations
   - Saga pattern for multi-step workflows

3. **Observability**
   - Distributed tracing across MCP calls
   - Performance metrics dashboard

## References

- [MCP Specification](https://modelcontextprotocol.io/)
- [MCP SDK Documentation](https://github.com/modelcontextprotocol/sdk)
- [JIRA REST API](https://developer.atlassian.com/cloud/jira/platform/rest/v2/)
- [GitHub REST API](https://docs.github.com/en/rest)
