# Agent Framework

A multi-agent system that turns JIRA tickets and GitHub issues into merged pull requests. Three AI agents — architect, engineer, and QA — collaborate through a DAG-based workflow to plan, implement, review, and ship code changes across multiple repositories.

## How it works

```
JIRA ticket / GitHub issue
        |
        v
   Architect ──> Engineer ──> Architect ──> QA ──> Architect
    (plan)     (implement)  (code review) (review) (create PR)
```

Each agent runs Claude Code as a subprocess with MCP tools for real-time JIRA and GitHub access. The framework handles task routing, retry with replanning, structured context passing between steps, and per-task budget management. Large tasks (>500 estimated lines) are automatically decomposed into subtasks.

## Quick start

```bash
pip install -e .

# Configure
cp config/agent-framework.yaml.example config/agent-framework.yaml
# Set GITHUB_TOKEN, optional JIRA credentials in .env

# Build MCP servers (JIRA + GitHub access during agent execution)
cd mcp-servers/jira && npm install && npm run build && cd ../..
cd mcp-servers/github && npm install && npm run build && cd ../..

# Validate setup
agent doctor

# Start working
agent up           # starts agents + dashboard
agent work         # describe what to build
agent down         # stop everything
```

## CLI

### Core commands

| Command | Description |
|---------|-------------|
| `agent work` | Describe a goal interactively, pick a repo, start the pipeline |
| `agent work --epic PROJ-100` | Process all tickets in a JIRA epic |
| `agent run PROJ-123` | Work a single JIRA ticket |
| `agent pull` | Pull unassigned tickets from JIRA backlog into the queue |

### Operations

| Command | Description |
|---------|-------------|
| `agent up` | Start agents + web dashboard, open browser |
| `agent down` | Stop everything gracefully |
| `agent start` | Start agent workers only (blocks terminal) |
| `agent start --replicas 4` | Run N replicas per agent for parallel processing |
| `agent stop` | Stop all services |
| `agent restart` | Restart agents and dashboard |
| `agent pause` / `agent resume` | Pause or resume task processing |
| `agent status --watch` | Live terminal view of queue state |
| `agent cancel <id>` | Cancel a queued task |
| `agent retry --all` | Retry all failed tasks |

### Teams

| Command | Description |
|---------|-------------|
| `agent team start -t full -r owner/repo` | Launch interactive team session |
| `agent team start -t full -r owner/repo -e PROJ-100` | Team session with epic context |
| `agent team escalate TASK-ID` | Debug a failed task with a team |

Templates: `full` (architect + engineer + QA), `review` (QA + security + performance), `debug` (2 investigators).

### Analysis and monitoring

| Command | Description |
|---------|-------------|
| `agent doctor` | Validate config, credentials, connectivity |
| `agent analyze --repo owner/repo` | Scan repo for tech debt, create JIRA epic |
| `agent analytics` | Cost, performance, and quality metrics |
| `agent dashboard` | Web dashboard |
| `agent summary --epic PROJ-100` | Epic progress with PR links |

## Architecture

### Agents

| Agent | Role | Permissions |
|-------|------|-------------|
| **Architect** | Plans work, breaks down epics, reviews code, creates PRs | JIRA create/update, git commit, PR create |
| **Engineer** | Implements features, writes tests, commits code | Git commit |
| **QA** | Linting, tests, security scan, code review, approval | PR create |

Each agent has configurable teammates (Claude Agent Teams) for specialized sub-tasks like security review, performance analysis, or pair programming.

### Workflow engine

Tasks flow through a configurable DAG. The default workflow:

**plan** -> **implement** -> **code_review** -> **qa_review** -> **create_pr**

Each transition evaluates structured verdicts (`approved`, `needs_fix`, `no_changes`). Review cycles bounce between engineer and reviewer with a cap of 2 cycles. Three depth ceilings prevent runaway loops:

1. `MAX_CHAIN_DEPTH=10` per workflow chain
2. `MAX_DAG_REVIEW_CYCLES=2` per QA/engineer bounce
3. `MAX_GLOBAL_CYCLES=15` absolute ceiling across escalations

### Task decomposition

Tasks estimated over 500 lines are split into subtasks by the architect. Subtasks run independently, then a fan-in task collects results and flows through the review chain as a single PR.

### Context passing

An append-only chain state file tracks each step's plan, files modified, verdict, and findings. Downstream agents receive structured context through a priority cascade:

1. Rejection feedback (direct fix instructions from reviewer)
2. Chain state (structured workflow history)
3. Structured findings (QA file-grouped checklist)
4. Upstream summary (inline text context)

### LLM backends

- **Claude Code CLI** (default) — subprocess with full MCP tool access
- **LiteLLM** — direct API calls for lighter workloads

Model routing is per-task: haiku for cheap eval/replan, sonnet for standard work, opus for complex planning. Intelligent routing scores tasks across complexity, history, specialization, budget, and retry signals.

### Budget management

Per-task USD ceilings by estimated effort (XS through XL). Context budget tracking alerts when token usage exceeds thresholds. Absolute ceiling prevents any single task tree from running away.

## Configuration

All config lives in `config/`:

| File | Purpose |
|------|---------|
| `agent-framework.yaml` | LLM settings, timeouts, budgets, workflows, repositories |
| `agents.yaml` | Agent definitions, prompts, teammates, JIRA permissions |
| `jira.yaml` | JIRA server, project, transitions |
| `github.yaml` | GitHub owner, repo, branch patterns |
| `mcp-config.json` | MCP server endpoints |
| `specializations.yaml` | Engineer specialization profiles (Go, Python, Ruby, etc.) |

### Adding a repository

```yaml
# config/agent-framework.yaml
repositories:
  - github_repo: your-org/your-app
    display_name: Your App
    auto_merge: true              # merge PRs after CI passes

  - github_repo: your-org/another-repo
    jira_project: PROJ            # optional JIRA integration
    display_name: Another Repo
```

Repos without `jira_project` use local-only task tracking through the same pipeline.

### Environment variables

```
GITHUB_TOKEN=ghp_...             # required for multi-repo
JIRA_SERVER=https://your-org.atlassian.net   # optional
JIRA_EMAIL=your@email.com                    # optional
JIRA_API_TOKEN=your-token                    # optional
```

## Observability

Every task produces a structured session log at `logs/sessions/{task_id}.jsonl` capturing prompts, tool calls, LLM responses, costs, and timing. Analytics collectors aggregate these into reports:

| Area | What's tracked |
|------|---------------|
| **Cost** | Per-task, per-model, per-step spend with trends |
| **Performance** | Success rates, retry patterns, completion times, handoff latency |
| **Quality** | Review cycle counts, verdict distribution, waste detection |
| **Git** | Commits per task, lines changed, push rates, edit-to-commit latency |
| **Workflow** | Step success rates, duration p50/p90, chain completion |

View via `agent analytics`, the web dashboard (`agent dashboard`), or raw JSONL files.

## MCP servers

Three Model Context Protocol servers give agents real-time access during execution:

| Server | Capabilities |
|--------|-------------|
| `mcp-servers/jira/` | Search, create, transition tickets, add comments |
| `mcp-servers/github/` | Read PRs/issues, post comments, check CI status |
| `mcp-servers/task-queue/` | Cross-agent task queue access |

## Project structure

```
src/agent_framework/
  analytics/       Metrics collectors (cost, performance, quality, git)
  cli/             Click CLI commands and TUI dashboard
  core/            Agent loop, task model, workflow routing, chain state
  indexing/        Codebase structural indexing for prompt context
  integrations/    JIRA and GitHub clients
  llm/             Claude CLI and LiteLLM backends, model routing
  memory/          Cross-task learning, tool pattern analysis
  queue/           File-based task queue with locking
  safeguards/      Circuit breaker, retry handler, escalation
  web/             FastAPI backend + Vue.js dashboard
  workflow/        DAG engine, step execution, task decomposition
  workspace/       Multi-repo clone management
```

## Safeguards

- Retry limits (max 5 per task) with exponential backoff
- Circuit breaker triggers on high failure rates or stuck tasks
- Per-task-type timeouts (15 min simple, 30 min bounded, 60 min large)
- Watchdog auto-restarts dead agents
- Stale lock recovery on poll cycle
- Queue size and task age limits
- Safety commits preserve uncommitted work before cleanup
- Budget ceilings per task and per effort estimate

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v              # 2,600+ tests
agent doctor                  # validate your setup
```

## License

MIT
