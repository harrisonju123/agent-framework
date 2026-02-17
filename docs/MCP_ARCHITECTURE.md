# MCP Architecture

## Overview

The agent framework uses MCP (Model Context Protocol) servers to give Claude agents real-time JIRA and GitHub access during task execution, replacing the previous post-LLM workflow where integrations only ran after agent completion.

## Architecture

```
┌─────────────────────────────────┐
│        Agent Framework          │
│                                 │
│  run_agent.py                   │
│    → ClaudeCLIBackend           │
│      (expands env vars in       │
│       MCP config, passes        │
│       --mcp-config to CLI)      │
│    → Agent (mcp_enabled=true)   │
│      (adds MCP tool guidance    │
│       to prompt)                │
└──────────┬──────────────────────┘
           │ Spawns subprocess
           ▼
   ┌───────────────────┐
   │  Claude CLI Agent  │
   │  (MCPs loaded)     │
   └───────┬───────────┘
           │ Uses MCP tools
     ┌─────┴─────┐
     ▼           ▼
  JIRA MCP    GitHub MCP
  Server      Server
     │           │
     ▼           ▼
  JIRA API    GitHub API
```

## Components

### ClaudeCLIBackend (`llm/claude_cli_backend.py`)

Spawns Claude CLI with `--mcp-config`. Expands `${VAR}` placeholders in MCP config with actual environment variable values before passing to CLI, since Claude CLI expects literal values.

### Agent (`core/agent.py`)

When `mcp_enabled=true`, the agent adds MCP tool guidance to prompts (available tools, branch naming patterns, error handling instructions) and skips the post-LLM workflow since the agent handles integrations directly.

### MCP Servers (`mcp-servers/jira/`, `mcp-servers/github/`)

Standalone Node.js processes using `@modelcontextprotocol/sdk`. Each server connects to Claude CLI via stdio, exposes tools via MCP protocol, and returns structured JSON responses. Logs to `logs/mcp-{service}.log`.

## Configuration

```yaml
# config/agent-framework.yaml
llm:
  use_mcp: true              # false = legacy post-LLM workflow
  mcp_config_path: config/mcp-config.json
  mode: claude_cli           # MCP requires Claude CLI mode
```

The `use_mcp` flag provides instant rollback without code changes.

## Environment Variables

- `JIRA_SERVER`, `JIRA_EMAIL`, `JIRA_API_TOKEN` — JIRA access
- `GITHUB_TOKEN` — GitHub access

Loaded from `.env` (git-ignored) and expanded by `ClaudeCLIBackend._expand_mcp_config()`.

## MCP vs Post-LLM Workflow

| Aspect | Post-LLM | MCP |
|--------|----------|-----|
| Integration access | After completion only | During execution |
| Error handling | Framework handles | Agent handles |
| Flexibility | Fixed workflow | Agent decides |
| Dependencies | Python clients | Node.js MCP servers |
