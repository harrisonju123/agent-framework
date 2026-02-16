#!/usr/bin/env python3
"""Queue Sprint 6 roadmap items as parallel tasks with dependency wiring.

Dependency graph:
  Wave 1 (parallel): #1 Test Coverage, #5 Debate→Memory, #6 Specialization→Routing
  Wave 2 (after #1): #2 Decompose agent.py
  Wave 3 (parallel, after #2): #3 Memory↔Replanning, #4 Memory↔Context Budget
"""

import time
from pathlib import Path

from agent_framework.core.task import Task, TaskStatus, TaskType, PlanDocument
from agent_framework.queue.file_queue import FileQueue
from datetime import datetime, timezone


GITHUB_REPO = "harrisonju123/agent-framework"
REPO_NAME = "Agent Framework"
JIRA_PROJECT = "AF"

SPRINT_ITEMS = [
    {
        "key": "s6-tests",
        "title": "Test coverage for agentic features (memory, reflection, replanning)",
        "goal": (
            "Write dedicated test suites for the untested agentic features: memory store, "
            "memory retriever, reflection/self-evaluation, and replanning. "
            "Target: 40+ new tests.\n\n"
            "Tests needed:\n"
            "- tests/unit/test_memory_store.py — persistence, deduplication, eviction (200 cap), "
            "category filtering, forget/cleanup\n"
            "- tests/unit/test_memory_retriever.py — relevance scoring, recency decay (7-day half-life), "
            "prompt char limit (3000), tag overlap\n"
            "- tests/unit/test_reflection.py — _self_evaluate() happy path, max retry enforcement (2), "
            "critique injection, skip when disabled\n"
            "- tests/unit/test_replanning.py — _request_replan() on retry 2+, _inject_replan_context(), "
            "history accumulation, skip on retry 1"
        ),
        "depends_on": [],
    },
    {
        "key": "s6-decompose",
        "title": "Decompose agent.py into focused modules",
        "goal": (
            "Extract cohesive responsibilities from agent.py (~4000 lines) into focused modules. "
            "The Agent class becomes a thin orchestrator that delegates.\n\n"
            "Extractions:\n"
            "- core/prompt_builder.py — _build_prompt(), memory injection, specialization, "
            "context budget, tool tips\n"
            "- core/post_completion.py — _run_post_completion_flow(), fan-in check, "
            "workflow chain, PR creation\n"
            "- core/qa_review.py — _parse_structured_findings(), _build_review_fix_task(), "
            "review verdict parsing\n"
            "- core/reflection.py — _self_evaluate(), _request_replan(), "
            "_inject_replan_context()\n\n"
            "Invariant: agent.py becomes ~1500 lines. Each extracted module has its own test file. "
            "Zero behavior change — pure refactor."
        ),
        "depends_on": ["s6-tests"],
    },
    {
        "key": "s6-memory-replan",
        "title": "Memory ↔ Replanning integration",
        "goal": (
            "When _request_replan() fires, inject relevant memories as context alongside "
            "the failure history. 'You've worked on this repo before. Here's what you know: ...'\n\n"
            "Changes:\n"
            "- core/reflection.py (after extraction) — replan pulls memories via MemoryRetriever\n"
            "- Memory categories conventions, test_commands, repo_structure prioritized for "
            "replan context"
        ),
        "depends_on": ["s6-decompose"],
    },
    {
        "key": "s6-memory-budget",
        "title": "Memory ↔ Context budget coordination",
        "goal": (
            "Memory injection is a fixed 3000 chars regardless of context budget remaining. "
            "Make it adaptive.\n\n"
            "ContextWindowManager provides remaining budget. Memory injection scales: "
            "full 3000 chars when budget is healthy, summarized to 1000 chars when budget "
            "is tight (>70% used), omitted when critical (>90%).\n\n"
            "Changes:\n"
            "- core/prompt_builder.py — query ContextWindowManager before injecting memories\n"
            "- memory/memory_retriever.py — accept max_chars parameter, already exists but "
            "caller needs to pass dynamic value"
        ),
        "depends_on": ["s6-decompose"],
    },
    {
        "key": "s6-debate-memory",
        "title": "Debate results → Memory persistence",
        "goal": (
            "Debates produce valuable architectural decisions (advocate/critic/arbiter synthesis "
            "with confidence levels). These are discarded after the session. Store them as "
            "persistent memories.\n\n"
            "After a debate concludes, automatically store the arbiter's synthesis as a memory "
            "entry with category architectural_decisions. Future tasks can recall these.\n\n"
            "Changes:\n"
            "- mcp-servers/task-queue/src/debate.ts — emit debate result to memory store after synthesis\n"
            "- memory/memory_store.py — new category architectural_decisions\n"
            "- Tag with repo + topic keywords for retrieval"
        ),
        "depends_on": [],
    },
    {
        "key": "s6-spec-routing",
        "title": "Specialization ↔ Model routing",
        "goal": (
            "A frontend task with 3 CSS files gets the same model as a complex backend distributed "
            "systems change. Specialization already knows the task domain — model routing should "
            "use this signal.\n\n"
            "Extend ModelSelector to accept specialization profile as input. Backend/infra tasks "
            "with high file count → premium model. Frontend/docs tasks → default model. "
            "Combine with existing retry escalation.\n\n"
            "Changes:\n"
            "- llm/model_selector.py — accept specialization_profile parameter\n"
            "- core/agent.py — pass profile to model selection"
        ),
        "depends_on": [],
    },
]


def main():
    workspace = Path("/Users/hju/PycharmProjects/agent-framework")
    queue = FileQueue(workspace)

    ts = int(time.time())
    task_ids = {}  # key -> full task ID

    # First pass: assign IDs
    for item in SPRINT_ITEMS:
        task_ids[item["key"]] = f"planning-{item['key']}-{ts}"

    # Second pass: create and queue tasks (skip if already queued)
    architect_dir = queue.queue_dir / "architect"
    created = []
    for item in SPRINT_ITEMS:
        key = item["key"]
        existing = list(architect_dir.glob(f"planning-{key}-*.json")) if architect_dir.exists() else []
        if existing:
            print(f"  ⏭️  Planning task for {key} already queued: {existing[0].stem}")
            continue

        task_id = task_ids[item["key"]]
        depends_on = [task_ids[dep] for dep in item["depends_on"]]

        instructions = f"""User Goal: {item['goal']}

Instructions for Architect Agent:
1. Explore the agent-framework codebase to understand current structure
2. Create a detailed implementation plan
3. Queue implementation task to engineer
4. Create JIRA ticket in project {JIRA_PROJECT}"""

        task = Task(
            id=task_id,
            type=TaskType.PLANNING,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="cli",
            assigned_to="architect",
            created_at=datetime.now(timezone.utc),
            title=f"Plan and delegate: {item['title']}",
            description=instructions,
            depends_on=depends_on,
            context={
                "mode": "planning",
                "workflow": "default_auto",
                "github_repo": GITHUB_REPO,
                "repository_name": REPO_NAME,
                "user_goal": item["goal"],
                "jira_available": True,
                "jira_project": JIRA_PROJECT,
            },
        )

        queue.push(task, "architect")
        created.append((item["key"], task_id, depends_on))
        print(f"  Queued: {task_id}")

    print(f"\n{len(created)} tasks queued to architect queue.")
    print("\nDependency graph:")
    for key, tid, deps in created:
        dep_str = " → depends on " + ", ".join(deps) if deps else " (no deps — starts immediately)"
        print(f"  {key}: {tid}{dep_str}")


if __name__ == "__main__":
    main()
