# GitHub MCP Server

MCP (Model Context Protocol) server for GitHub integration with the agent framework.

## Features

This MCP server provides the following GitHub operations:

- **github_create_branch** - Create new branch from base branch
- **github_create_pr** - Create pull request
- **github_add_pr_comment** - Add comment to pull request
- **github_get_pr_by_branch** - Find PR by branch name
- **github_link_pr_to_jira** - Update PR body with JIRA ticket link

## Setup

### Install Dependencies

```bash
npm install
```

### Build

```bash
npm run build
```

### Environment Variables

Required:
- `GITHUB_TOKEN` - GitHub personal access token

Optional:
- `JIRA_SERVER` - JIRA server hostname for PR linking (default: `jira.example.com`)
- `LOG_PATH` - Path to log file (default: `./logs/mcp-github.log`)
- `LOG_LEVEL` - Logging level (default: `info`)

### Test Standalone

```bash
export GITHUB_TOKEN=ghp_your_token

npm start
```

## Tool Return Values

All tools return structured responses:

### Success Response
```json
{
  "success": true,
  "data": {
    // Tool-specific data
  }
}
```

### Error Response
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Error description"
  }
}
```

## Examples

### Create Branch
```json
{
  "owner": "justworkshr",
  "repo": "pto",
  "branchName": "feature/PROJ-123-add-auth",
  "fromBranch": "main"
}
```

Returns:
```json
{
  "success": true,
  "data": {
    "name": "feature/PROJ-123-add-auth",
    "sha": "abc123...",
    "url": "https://github.com/justworkshr/pto/tree/feature/PROJ-123-add-auth"
  }
}
```

### Create PR
```json
{
  "owner": "justworkshr",
  "repo": "pto",
  "title": "[PROJ-123] Add authentication",
  "body": "This PR adds authentication feature",
  "head": "feature/PROJ-123-add-auth",
  "base": "main",
  "draft": false,
  "labels": ["enhancement"]
}
```

Returns:
```json
{
  "success": true,
  "data": {
    "number": 456,
    "url": "https://github.com/justworkshr/pto/pull/456",
    "html_url": "https://github.com/justworkshr/pto/pull/456"
  }
}
```

### Link PR to JIRA
```json
{
  "owner": "justworkshr",
  "repo": "pto",
  "prNumber": 456,
  "jiraKey": "PROJ-123"
}
```

Returns:
```json
{
  "success": true,
  "data": {
    "prNumber": 456,
    "jiraKey": "PROJ-123",
    "jiraLink": "https://jira.example.com/browse/PROJ-123"
  }
}
```

## Notes

Git operations (clone, commit, push) are handled by the Python framework's `MultiRepoManager`. This MCP server focuses on GitHub API operations only.

## Integration with Agent Framework

This MCP server is configured in `config/mcp-config.json` and automatically started by the Claude CLI when agents run with the `--mcp-config` flag.

See the main [MCP Setup Guide](../../docs/MCP_SETUP.md) for details.
