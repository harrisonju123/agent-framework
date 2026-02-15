# Getting Started with Agent Framework

## Prerequisites

- Python 3.9+, Git
- GitHub personal access token (`repo` scope)
- JIRA API token
- Claude Code CLI installed

## Quick Start (Web Setup)

```bash
git clone https://github.com/your-org/agent-framework.git
cd agent-framework
pip install -e .
agent dashboard
```

The dashboard opens at http://localhost:8080 with a setup wizard that walks you through JIRA, GitHub, and repository configuration.

## Manual Setup

```bash
agent init                    # Creates config/ directory with examples
cd config
cp agents.yaml.example agents.yaml
cp jira.yaml.example jira.yaml
cp github.yaml.example github.yaml
cp agent-framework.yaml.example agent-framework.yaml
```

Edit each file with your settings, then create `.env` in workspace root:

```
JIRA_SERVER=https://your-domain.atlassian.net
JIRA_EMAIL=you@example.com
JIRA_API_TOKEN=your-jira-token
GITHUB_TOKEN=ghp_your-github-token
```

```bash
chmod 600 .env
agent doctor          # Verify everything works
```

## Running Agents

```bash
agent start                       # Start all agents
agent start --replicas 2          # Multiple replicas
agent status --watch              # Monitor activity
agent dashboard                   # Web UI with live logs
```

## Creating Work

```bash
# Via CLI
agent work --repo yourorg/yourrepo --goal "Add user authentication" --workflow standard

# Via dashboard: Click "New Work", fill in goal and repository
```

## Workflows

| Workflow | Agents | Use for |
|----------|--------|---------|
| simple | Engineer only | Bug fixes, small features |
| standard | Engineer → QA | Features needing tests |
| full | Architect → Engineer → QA | Complex features |

## Learn More

- **Troubleshooting**: `docs/TROUBLESHOOTING.md`
- **Config reference**: `config/docs/*.md`
- **CLI help**: `agent --help`
- **Health check**: `agent doctor`
