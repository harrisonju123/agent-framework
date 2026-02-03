# JIRA MCP Server

MCP (Model Context Protocol) server for JIRA integration with the agent framework.

## Features

This MCP server provides the following JIRA operations:

- **jira_search_issues** - Search issues using JQL
- **jira_get_issue** - Get detailed issue information
- **jira_create_issue** - Create Story, Bug, or Task
- **jira_create_epic** - Create Epic
- **jira_create_subtask** - Create subtask under parent issue
- **jira_transition_issue** - Change issue status
- **jira_add_comment** - Add comment to issue
- **jira_update_field** - Update custom field
- **jira_create_epic_with_subtasks** - Batch create epic with subtasks

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
- `JIRA_SERVER` - JIRA server hostname (e.g., `company.atlassian.net`)
- `JIRA_EMAIL` - JIRA user email
- `JIRA_API_TOKEN` - JIRA API token

Optional:
- `LOG_PATH` - Path to log file (default: `./logs/mcp-jira.log`)
- `LOG_LEVEL` - Logging level (default: `info`)

### Test Standalone

```bash
export JIRA_SERVER=company.atlassian.net
export JIRA_EMAIL=user@example.com
export JIRA_API_TOKEN=your-token

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

### Search Issues
```json
{
  "jql": "project = PROJ AND status = Open",
  "maxResults": 10
}
```

Returns:
```json
{
  "success": true,
  "data": {
    "total": 5,
    "issues": [
      {
        "key": "PROJ-123",
        "summary": "Add authentication",
        "status": "Open",
        "url": "https://company.atlassian.net/browse/PROJ-123"
      }
    ]
  }
}
```

### Create Epic
```json
{
  "project": "PROJ",
  "title": "User Management",
  "description": "Implement user management features"
}
```

Returns:
```json
{
  "success": true,
  "data": {
    "key": "PROJ-456",
    "id": "10001",
    "url": "https://company.atlassian.net/browse/PROJ-456"
  }
}
```

### Transition Issue
```json
{
  "issueKey": "PROJ-123",
  "transitionName": "In Progress"
}
```

Returns:
```json
{
  "success": true,
  "data": {
    "issueKey": "PROJ-123",
    "newStatus": "In Progress"
  }
}
```

## Integration with Agent Framework

This MCP server is configured in `config/mcp-config.json` and automatically started by the Claude CLI when agents run with the `--mcp-config` flag.

See the main [MCP Setup Guide](../../docs/MCP_SETUP.md) for details.
