# Troubleshooting

Run `agent doctor` first — it catches most configuration issues automatically.

## JIRA

**"401 Unauthorized"** — Use an API token, not your password. Generate at https://id.atlassian.com/manage-profile/security/api-tokens. Email must match your Atlassian account. Server URL needs `https://`.

**"Project KEY does not exist"** — Project key must be uppercase (e.g., `PROJ` not `proj`). Verify you have access to the project in JIRA.

## GitHub

**"Bad credentials"** — Generate a new token at https://github.com/settings/tokens with `repo` scope. Update `GITHUB_TOKEN` in `.env`.

**"API rate limit exceeded"** — Authenticated limit is 5,000 req/hour. Increase `poll_interval` in `config/agent-framework.yaml` if hitting limits.

## Agents

**Circuit breaker activated** — 5 consecutive failures triggers the breaker. Check `agent status` for errors, fix the root cause, then restart with `agent stop && agent start`.

**MCP server not available** — MCP is optional. Set `use_mcp: false` in `config/agent-framework.yaml` to disable. Agents work without it using CLI-based access.

**Agent stuck** — Check logs: `tail -f workspace/logs/engineer.log`. If truly stuck, restart: `agent stop && agent start`.

## Budget

**Budget or quota exceeded** — The account has reached its usage limit. The system will NOT retry budget-exceeded errors to avoid wasting resources. Actions:
1. Review your account usage and spending in the API provider's dashboard
2. Check available credits or quota in your account settings
3. Upgrade your account plan or increase budget limits
4. Optimize resource usage (reduce model sizes, limit concurrent tasks) to reduce costs

Budget errors are detected by patterns like "budget exceeded", "max budget", "quota exceeded", "insufficient credits", or "usage limit exceeded". These errors skip retries and immediately escalate for human review.

## Performance

**Slow task processing** — Usually network latency or large repos. Try `agent start --replicas 2` for parallelism.

**High memory** — Limit concurrent tasks in `config/agent-framework.yaml` under `task_processing.max_concurrent_tasks`. Clean old worktrees with `git worktree prune`.

## Reporting Issues

Include output of `agent doctor` and relevant log excerpts.
