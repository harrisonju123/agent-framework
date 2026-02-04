# Getting Started with Agent Framework

Quick start guide for agent-framework setup.

## Prerequisites

- **Python 3.9+** installed
- **Git** installed
- **GitHub** personal access token
- **JIRA** API token
- **Claude Code CLI** installed (for LLM access)

---

## Quick Start (Web Setup)

Web-based setup wizard handles configuration automatically.

### 1. Install Agent Framework

```bash
# Clone repository
git clone https://github.com/your-org/agent-framework.git
cd agent-framework

# Install dependencies
pip install -e .
```

### 2. Launch Setup Wizard

```bash
# Start web dashboard
agent dashboard
```

This will:
- Open your browser to http://localhost:8080
- Show a setup prompt banner
- Guide you through configuration

### 3. Complete Setup Wizard

Steps:

**Step 1: JIRA Configuration**
- Server URL: `https://your-domain.atlassian.net`
- Email: Your Atlassian account email
- API Token: [Generate here](https://id.atlassian.com/manage-profile/security/api-tokens)
- Click "Test Connection" to verify

**Step 2: GitHub Configuration**
- Personal Access Token: [Generate here](https://github.com/settings/tokens)
  - Required scope: `repo` (full control)
- Click "Test Connection" to verify

**Step 3: Repository Configuration**
- Add repositories you want to work with
- Map each repo to a JIRA project
- Format: `owner/repo` → `PROJECT-KEY`

**Step 4: Review & Save**
- Review all settings
- Click "Save Configuration"
- Config files will be generated automatically

### 4. Start Agents

Once setup is complete:

```bash
# Start all agents
agent start

# Or start specific number of replicas
agent start --replicas 2
```

### 5. Create First Task

In the dashboard:
- Click "New Work"
- Enter goal: "Add a health check endpoint to the API"
- Select repository: `yourorg/yourrepo`
- Choose workflow: `simple` (engineer only)
- Click "Create"

Product-owner agent plans the work and queues it for engineer.

---

## Manual Setup (Alternative)

If you prefer CLI/manual configuration:

### 1. Initialize Workspace

```bash
agent init
```

This creates:
```
workspace/
  config/
    agents.yaml.example
    jira.yaml.example
    github.yaml.example
```

### 2. Create Configuration Files

```bash
cd config

# Copy examples and edit
cp agents.yaml.example agents.yaml
cp jira.yaml.example jira.yaml
cp github.yaml.example github.yaml
cp agent-framework.yaml.example agent-framework.yaml
```

### 3. Edit Configuration

**jira.yaml**:
```yaml
server: https://your-domain.atlassian.net
email: you@example.com
default_project: PROJ
```

**github.yaml**:
```yaml
default_branch: main
require_pr_approval: true
```

**agent-framework.yaml**:
```yaml
llm:
  mode: claude_cli
  use_mcp: false
  default_model: sonnet

repositories:
  - name: myapp
    github_repo: yourorg/myapp
    jira_project: PROJ
```

### 4. Create .env File

```bash
# In workspace root
cat > .env << 'EOF'
JIRA_SERVER=https://your-domain.atlassian.net
JIRA_EMAIL=you@example.com
JIRA_API_TOKEN=your-jira-token
GITHUB_TOKEN=ghp_your-github-token
EOF

# Set restrictive permissions
chmod 600 .env
```

### 5. Verify Setup

```bash
agent doctor
```

This runs health checks and reports any configuration issues.

---

## Next Steps

### Monitor Agent Activity

**Web Dashboard** (recommended):
```bash
agent dashboard
```

Features:
- Real-time agent status
- Live log streaming
- Task queue visualization
- Health monitoring
- Interactive controls

**CLI Status**:
```bash
# One-time status
agent status

# Watch mode (auto-refresh)
agent status --watch
```

### Create Work Tasks

**Via Dashboard**:
- Click "New Work" button
- Fill in goal and repository
- Select workflow (simple/standard/full)

**Via CLI**:
```bash
agent work --repo yourorg/yourrepo \
  --goal "Add user authentication" \
  --workflow standard
```

### Analyze Repository

Find and fix issues automatically:

```bash
agent analyze --repo yourorg/yourrepo
```

Creates JIRA epic with subtasks for found issues.

### Run Existing JIRA Tickets

**Via Dashboard**:
- Click "Run Ticket"
- Enter ticket ID: `PROJ-123`
- Select agent (or auto-assign)

**Via CLI**:
```bash
agent run --ticket PROJ-123
```

---

## Understanding Workflows

### Simple Workflow
- **Agents**: Engineer only
- **Best for**: Small features, bug fixes, quick changes
- **Flow**: Plan → Implement → Commit → PR

### Standard Workflow
- **Agents**: Engineer → QA
- **Best for**: Features needing testing
- **Flow**: Plan → Implement → Test → PR

### Full Workflow
- **Agents**: Architect → Engineer → QA
- **Best for**: Complex features, major changes
- **Flow**: Design → Implement → Test → Review → PR

---

## Configuration Reference

### Key Configuration Files

**config/agent-framework.yaml**:
- LLM settings (Claude CLI, MCP)
- Repository definitions
- Workflow definitions
- Task processing settings

**config/agents.yaml**:
- Agent definitions (engineer, qa, architect)
- Agent prompts and capabilities
- Queue assignments

**config/jira.yaml**:
- JIRA server URL
- Default project
- Integration settings

**config/github.yaml**:
- Default branch
- PR approval settings
- Integration settings

**.env**:
- Credentials (tokens, passwords)
- Never commit this file!

---

## Troubleshooting

### Setup Issues

**Problem**: "Configuration missing" error

**Solution**:
1. Run `agent doctor` to diagnose
2. Launch setup wizard: `agent dashboard` → "Setup"
3. Or manually check config files exist

**Problem**: "JIRA authentication failed"

**Solution**:
1. Verify JIRA API token: https://id.atlassian.com/manage-profile/security/api-tokens
2. Check email matches Atlassian account
3. Test connection in setup wizard

**Problem**: "GitHub authentication failed"

**Solution**:
1. Generate new token with `repo` scope: https://github.com/settings/tokens
2. Update `.env` file
3. Test connection: `agent doctor`

### Common Pitfalls

❌ **Don't**:
- Commit `.env` file to git
- Use password instead of API token for JIRA
- Forget to set file permissions on `.env`
- Use GitHub token without `repo` scope

✅ **Do**:
- Add `.env` to `.gitignore`
- Generate API tokens (not passwords)
- Run `agent doctor` after configuration changes
- Test connections before starting agents

---

## Learn More

- **Configuration Details**: See `config/docs/*.md`
- **Troubleshooting**: See `docs/TROUBLESHOOTING.md`
- **CLI Reference**: Run `agent --help`
- **Architecture**: See `docs/ARCHITECTURE.md`

---

## Getting Help

- **Health Check**: `agent doctor`
- **Dashboard**: `agent dashboard` (web UI with logs)
- **Documentation**: README.md
- **Issues**: GitHub Issues or internal Slack
