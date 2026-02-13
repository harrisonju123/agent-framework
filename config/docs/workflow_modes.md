# Workflow

**Version:** 3.0
**Last Updated:** 2026-02-13
**Status:** Active

## Overview

Every task runs the same agent chain. No mode selection needed.

## Agent Chain

```
Architect → Engineer → QA → Architect (creates PR)
```

1. **Architect** — Analyzes task, creates implementation plan, queues to Engineer
2. **Engineer** — Implements per plan, writes tests, commits & pushes. Does NOT create PR
3. **QA** — Reviews code, runs tests/linting/security. Pass → queue to Architect. Fail → queue fix to Engineer
4. **Architect** — Final review against plan, creates PR

## Failure Loop

QA ↔ Engineer up to 3 cycles, then escalate to Architect for replanning.

## PR Creation

Architect always creates the PR after QA passes and architecture review.
Uses template from config/docs/pr_templates.md with metrics from config/docs/change_metrics.md.

## Task Splitting

If estimated >500 lines, Architect splits into multiple subtasks (<300 lines each).
See config/docs/change_metrics.md for splitting strategies.
