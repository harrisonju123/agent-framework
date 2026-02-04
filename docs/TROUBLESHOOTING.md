# Troubleshooting Guide

Common issues and solutions for the agent framework.

## Table of Contents

- [Setup Issues](#setup-issues)
- [JIRA Integration](#jira-integration)
- [GitHub Integration](#github-integration)
- [Agent Failures](#agent-failures)
- [Performance Issues](#performance-issues)

---

## Setup Issues

### Configuration Missing

**Symptom**: Error message "Configuration missing" or "Config files not found"

**Solution**:
1. Run the setup wizard: `agent dashboard` → Click "Setup" button
2. Or manually create config files: `agent init`
3. Ensure `.env` file exists with credentials

**Files Required**:
- `config/agent-framework.yaml`
- `config/agents.yaml`
- `config/jira.yaml`
- `config/github.yaml`
- `.env`

### Environment Variables Not Set

**Symptom**: "Missing environment variables" error

**Solution**:
1. Create `.env` file in workspace root
2. Add required variables:
   ```
   JIRA_SERVER=https://your-domain.atlassian.net
   JIRA_EMAIL=you@example.com
   JIRA_API_TOKEN=your-token
   GITHUB_TOKEN=ghp_your-token
   ```
3. Ensure `.env` is loaded (framework does this automatically)

---

## JIRA Integration

### Authentication Failed {#jira-auth}

**Symptom**: "JIRA authentication failed" or "401 Unauthorized"

**Cause**: Invalid API token or expired credentials

**Solution**:
1. Generate new API token:
   - Go to https://id.atlassian.com/manage-profile/security/api-tokens
   - Click "Create API token"
   - Copy the token (you won't be able to see it again!)
2. Update credentials:
   - Run setup wizard: `agent dashboard` → "Setup"
   - Or edit `.env` file directly
3. Verify email matches your Atlassian account
4. Test connection: `agent doctor`

**Common Mistakes**:
- Using password instead of API token
- Wrong email (must match Atlassian account)
- Expired token (regenerate if needed)
- Server URL missing `https://`

### Connection Timeout

**Symptom**: "Cannot reach JIRA server" or timeout errors

**Solution**:
1. Check internet connection
2. Verify JIRA server URL is correct
3. Check if JIRA is down: https://status.atlassian.com/
4. Try accessing JIRA in browser
5. Check corporate firewall/VPN settings

### Project Not Found

**Symptom**: "Project KEY does not exist"

**Solution**:
1. Verify project key is uppercase (e.g., "PROJ" not "proj")
2. Check you have access to the project in JIRA
3. Update project key in config:
   - `config/jira.yaml` → `default_project`
   - Or repository-specific in `config/agent-framework.yaml`

---

## GitHub Integration

### Authentication Failed {#github}

**Symptom**: "Bad credentials" or "401" from GitHub API

**Cause**: Invalid or expired personal access token

**Solution**:
1. Generate new token:
   - Go to https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Select scopes: `repo` (full control)
   - Copy token immediately
2. Update token:
   - Run setup wizard or edit `.env`
   - `GITHUB_TOKEN=ghp_your_new_token`
3. Test connection: `agent doctor`

**Required Permissions**:
- `repo` - Full control of private repositories
  - `repo:status` - Access commit status
  - `repo_deployment` - Access deployment status
  - `public_repo` - Access public repositories (if needed)

### Rate Limit Exceeded

**Symptom**: "API rate limit exceeded" or "429 Too Many Requests"

**Cause**: Too many API requests in short time

**Solution**:
1. Wait for rate limit to reset (check header: X-RateLimit-Reset)
2. Reduce polling frequency in `config/agent-framework.yaml`:
   ```yaml
   task_processing:
     poll_interval: 10  # Increase from 5 to 10 seconds
   ```
3. Check rate limit status:
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" https://api.github.com/rate_limit
   ```

**Rate Limits** (as of 2024):
- **Authenticated**: 5,000 requests/hour
- **Unauthenticated**: 60 requests/hour

---

## Agent Failures

### Circuit Breaker Activated

**Symptom**: "CircuitBreakerOpenError" or "System safeguards activated"

**Cause**: Too many consecutive failures triggered safety mechanism

**Solution**:
1. Check recent failures: `agent status`
2. View system health: `agent doctor`
3. Fix underlying issues (see errors in logs)
4. Reset circuit breaker:
   ```bash
   # Stop agents
   agent stop
   # Fix issues
   # Restart agents
   agent start
   ```

**Circuit Breaker Thresholds**:
- Opens after: 5 consecutive failures
- Reset timeout: 60 seconds
- Half-open state: Tests with 1 request

### MCP Server Not Available

**Symptom**: "mcp server not found" or "mcp not available"

**Cause**: MCP servers not configured (optional feature)

**Solution** (choose one):

**Option 1: Disable MCP** (recommended for MVP):
```yaml
# config/agent-framework.yaml
llm:
  use_mcp: false
```

**Option 2: Set up MCP servers**:
1. See `docs/MCP_SETUP.md` for instructions
2. Requires Node.js and additional configuration
3. Provides real-time JIRA/GitHub access

**Option 3: Ignore** (agents work without MCP):
- Agents will use CLI-based GitHub/JIRA access
- Slightly slower but fully functional

### Agent Stuck or Unresponsive

**Symptom**: Agent status shows "working" for extended period

**Solution**:
1. Check agent logs:
   ```bash
   tail -f workspace/logs/engineer.log
   ```
2. View task details in dashboard
3. If truly stuck, restart agent:
   ```bash
   agent stop
   agent start
   ```
4. Consider increasing timeout in config

---

## Performance Issues

### Slow Task Processing

**Possible Causes**:
- Large repository (clone/checkout takes time)
- Network latency (GitHub/JIRA API)
- LLM response time
- Resource constraints

**Solutions**:
1. **Repository size**: Use shallow clones
2. **Network**: Check connection, consider local GitHub Enterprise
3. **Resources**: Increase replicas
   ```bash
   agent start --replicas 2
   ```
4. **Timeouts**: Adjust in config if legitimate work takes longer

### High Memory Usage

**Cause**: Multiple repositories checked out simultaneously

**Solution**:
1. Limit concurrent agents:
   ```yaml
   # config/agent-framework.yaml
   task_processing:
     max_concurrent_tasks: 1  # Per agent type
   ```
2. Use worktrees (automatic) instead of full clones
3. Clean up old worktrees:
   ```bash
   git worktree prune
   ```

### Disk Space Issues

**Cause**: Accumulated worktrees and logs

**Solution**:
1. Clean old worktrees:
   ```bash
   find workspace/repos -type d -name '.git' -exec git -C {} worktree prune \;
   ```
2. Rotate logs (configure log rotation)
3. Remove completed task data (if archived)

---

## Getting Help

If you encounter issues not covered here:

1. **Run diagnostics**:
   ```bash
   agent doctor
   ```

2. **Check logs**:
   ```bash
   # View recent logs
   agent dashboard  # Web UI with live logs

   # Or via CLI
   tail -f workspace/logs/*.log
   ```

3. **Report issue**:
   - GitHub Issues: https://github.com/your-org/agent-framework/issues
   - Include:
     - Output of `agent doctor`
     - Relevant log excerpts
     - Steps to reproduce
     - Expected vs actual behavior

4. **Ask for help**:
   - Internal Slack: #agent-framework
   - Documentation: README.md, config/docs/*.md
