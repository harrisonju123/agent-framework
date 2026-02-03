# MCP Setup Guide

This guide explains how to set up and use MCP (Model Context Protocol) servers for JIRA and GitHub integration with the agent framework.

## Overview

MCP servers enable agents to interact with JIRA and GitHub in real-time during task execution, rather than only through post-processing. This provides:

- **Real-time access** - Query issues, create tickets, manage PRs during execution
- **Transparent workflows** - All operations visible in agent logs
- **Better error handling** - Agents can see and respond to integration errors
- **Simpler architecture** - No post-LLM magic, agents do everything explicitly

## Requirements

- **Node.js 18+** (for running MCP servers)
- **Claude CLI** (MCP integration requires `claude_cli` mode)
- Environment variables for JIRA and GitHub

## Installation

### 1. Install Node.js Dependencies

```bash
# Install JIRA MCP server
cd mcp-servers/jira
npm install
npm run build

# Install GitHub MCP server
cd ../github
npm install
npm run build
```

### 2. Configure MCP Servers

Copy the MCP configuration template:

```bash
cp config/mcp-config.json.example config/mcp-config.json
```

The configuration uses environment variables from your existing `.env` file:

```json
{
  "mcpServers": {
    "jira": {
      "command": "node",
      "args": ["${PWD}/mcp-servers/jira/build/index.js"],
      "env": {
        "JIRA_SERVER": "${JIRA_SERVER}",
        "JIRA_EMAIL": "${JIRA_EMAIL}",
        "JIRA_API_TOKEN": "${JIRA_API_TOKEN}",
        "LOG_PATH": "${PWD}/logs/mcp-jira.log"
      }
    },
    "github": {
      "command": "node",
      "args": ["${PWD}/mcp-servers/github/build/index.js"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}",
        "JIRA_SERVER": "${JIRA_SERVER}",
        "LOG_PATH": "${PWD}/logs/mcp-github.log"
      }
    }
  }
}
```

### 3. Enable MCP in Framework Configuration

Edit `config/agent-framework.yaml`:

```yaml
llm:
  mode: claude_cli  # Required for MCP
  use_mcp: true
  mcp_config_path: ${PWD}/config/mcp-config.json
```

## Verification

### Test MCP Servers

Test that MCP servers start correctly:

```bash
# Source environment variables
source scripts/setup-env.sh

# Test JIRA MCP server (Ctrl+C to exit)
cd mcp-servers/jira
npm start

# Test GitHub MCP server (Ctrl+C to exit)
cd ../github
npm start
```

If servers start without errors, MCP setup is complete.

### Test with Agent

Start an agent with MCP enabled:

```bash
# Ensure use_mcp: true in config/agent-framework.yaml
agent start --agent product-owner

# Watch logs to verify MCP tools are available
tail -f logs/product-owner.log

# Look for MCP tool usage in logs
# Should see: "Using tool: jira_create_epic", etc.
```

## Usage

### Available JIRA Tools

Agents have access to these JIRA tools:

- `jira_search_issues(jql, maxResults, fields)` - Search issues
- `jira_get_issue(issueKey)` - Get issue details
- `jira_create_issue(project, summary, description, issueType, labels)` - Create ticket
- `jira_create_epic(project, title, description)` - Create epic
- `jira_create_subtask(parentKey, summary, description)` - Create subtask
- `jira_transition_issue(issueKey, transitionName)` - Change status
- `jira_add_comment(issueKey, comment)` - Add comment
- `jira_update_field(issueKey, fieldName, value)` - Update custom field
- `jira_create_epic_with_subtasks(project, epicTitle, epicDescription, subtasks)` - Batch create

### Available GitHub Tools

Agents have access to these GitHub tools:

- `github_create_branch(owner, repo, branchName, fromBranch)` - Create branch
- `github_create_pr(owner, repo, title, body, head, base, draft, labels)` - Create PR
- `github_add_pr_comment(owner, repo, prNumber, body)` - Add PR comment
- `github_get_pr_by_branch(owner, repo, branchName)` - Find PR by branch
- `github_link_pr_to_jira(owner, repo, prNumber, jiraKey)` - Link PR to JIRA

### Example Workflow

When an agent receives a task with JIRA context:

1. Agent uses `jira_get_issue` to understand requirements
2. Agent makes code changes
3. Code is automatically committed by framework
4. Agent uses `github_create_pr` to create pull request
5. Agent uses `jira_transition_issue` to move ticket to "Code Review"
6. Agent uses `jira_add_comment` to link PR in JIRA

All operations are logged and visible in the agent's conversation.

## Troubleshooting

### MCP Server Not Starting

**Error:** `Missing required environment variables`

**Solution:** Ensure environment variables are set:
```bash
source scripts/setup-env.sh
echo $JIRA_SERVER
echo $GITHUB_TOKEN
```

### MCP Tools Not Available

**Error:** Agents don't see MCP tools

**Solution:**
1. Verify `use_mcp: true` in `config/agent-framework.yaml`
2. Verify `mode: claude_cli` (MCPs require Claude CLI mode)
3. Check MCP server logs: `tail -f logs/mcp-jira.log`

### Authentication Failures

**Error:** `401 Unauthorized` from JIRA or GitHub

**Solution:**
1. Verify credentials in `.env` file
2. Test credentials manually:
   ```bash
   # Test JIRA
   curl -u "$JIRA_EMAIL:$JIRA_API_TOKEN" "https://$JIRA_SERVER/rest/api/2/myself"

   # Test GitHub
   curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user
   ```

### Rate Limiting

**Error:** `Rate limit exceeded`

**Solution:**
- JIRA: Wait 60 seconds and retry
- GitHub: Check rate limit headers, typically resets hourly
- MCP servers include rate limiting protection

## Disabling MCPs

To temporarily disable MCPs and use the legacy post-LLM workflow:

```yaml
# config/agent-framework.yaml
llm:
  use_mcp: false
```

This provides instant rollback to the previous behavior.

## MCP Logs

MCP server logs are separate from agent logs:

- JIRA MCP: `logs/mcp-jira.log`
- GitHub MCP: `logs/mcp-github.log`

View logs:
```bash
tail -f logs/mcp-jira.log
tail -f logs/mcp-github.log
```

## Next Steps

- Read [MCP Architecture](MCP_ARCHITECTURE.md) for technical details
- See [JIRA MCP README](../mcp-servers/jira/README.md) for tool details
- See [GitHub MCP README](../mcp-servers/github/README.md) for tool details
