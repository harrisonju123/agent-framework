# Workflow

**Version:** 3.1
**Last Updated:** 2026-02-18
**Status:** Active

## Overview

Every task runs the same agent chain. No mode selection needed.

Agents do NOT manually queue tasks. The workflow DAG executor handles all routing.

## Agent Chain

```
Architect → Engineer → QA → Architect (creates PR)
```

1. **Architect** — Analyzes task, creates implementation plan. Framework routes to Engineer automatically
2. **Engineer** — Implements per plan, writes tests, commits & pushes. Does NOT create PR
3. **QA** — Reviews code, runs tests/linting/security. Framework routes pass → Architect, fail → Engineer
4. **Architect** — Final review against plan, creates PR

## Failure Loop

QA ↔ Engineer up to 3 cycles, then escalate to Architect for replanning.

## PR Creation

Architect always creates the PR after QA passes and architecture review.
Uses template from config/docs/pr_templates.md with metrics from config/docs/change_metrics.md.

## Task Splitting

If estimated >500 lines, the framework automatically decomposes into subtasks based on the plan's `files_to_modify` list.
See config/docs/change_metrics.md for size guidelines. Architects should ensure `files_to_modify` is accurate.
