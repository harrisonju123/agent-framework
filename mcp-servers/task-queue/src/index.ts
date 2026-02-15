#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import { createLogger } from "./logger.js";
import {
  queueTaskForAgent,
  getQueueStatus,
  listPendingTasks,
  getTaskDetails,
  getEpicProgress,
  writeRoutingSignal,
} from "./queue-tools.js";
import { consultAgent, getRemainingConsultations, decrementConsultations } from "./consultation.js";
import { shareKnowledge, getKnowledge } from "./knowledge.js";
import { debateTopic } from "./debate.js";
import type { QueueTaskInput, AgentId } from "./types.js";

const logger = createLogger();

const workspace = process.env.WORKSPACE || ".";

const TOOLS: Tool[] = [
  {
    name: "queue_task_for_agent",
    description:
      "Queue a task for another agent (engineer, qa, architect, product-owner). Use this to delegate work as part of epic breakdown or workflow orchestration.",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: {
          type: "string",
          enum: ["engineer", "qa", "architect", "product-owner", "code-reviewer", "testing", "static-analysis", "repo-analyzer"],
          description: "Target agent to receive the task",
        },
        task_type: {
          type: "string",
          enum: [
            "implementation",
            "verification",
            "qa_verification",
            "architecture",
            "planning",
            "review",
            "fix",
            "bugfix",
            "enhancement",
            "testing",
            "documentation",
            "analysis",
          ],
          description: "Type of task determining model selection",
        },
        title: {
          type: "string",
          description: "Brief task title",
        },
        description: {
          type: "string",
          description: "Detailed task description with requirements",
        },
        context: {
          type: "object",
          properties: {
            jira_key: {
              type: "string",
              description: "JIRA ticket key (e.g., PROJ-123)",
            },
            jira_project: {
              type: "string",
              description: "JIRA project key",
            },
            github_repo: {
              type: "string",
              description: "GitHub repo in owner/repo format",
            },
            epic_key: {
              type: "string",
              description: "Parent epic key for progress tracking",
            },
            workflow: {
              type: "string",
              enum: ["simple", "standard", "full"],
              description: "Workflow mode for the task",
            },
            use_worktree: {
              type: "boolean",
              description: "Enable git worktree isolation for this task",
            },
          },
          description: "Task context with JIRA, GitHub, and workflow info",
        },
        depends_on: {
          type: "array",
          items: { type: "string" },
          description: "Task IDs that must complete before this task can start",
        },
        priority: {
          type: "number",
          description: "Task priority (1-100, lower = higher priority)",
          default: 50,
        },
        acceptance_criteria: {
          type: "array",
          items: { type: "string" },
          description: "Acceptance criteria for task completion",
        },
        plan: {
          type: "object",
          properties: {
            objectives: {
              type: "array",
              items: { type: "string" },
              description: "What we're trying to achieve",
            },
            approach: {
              type: "array",
              items: { type: "string" },
              description: "Step-by-step implementation approach",
            },
            risks: {
              type: "array",
              items: { type: "string" },
              description: "Potential issues and mitigations",
            },
            success_criteria: {
              type: "array",
              items: { type: "string" },
              description: "How to verify completion",
            },
            files_to_modify: {
              type: "array",
              items: { type: "string" },
              description: "Files that will be changed",
            },
            dependencies: {
              type: "array",
              items: { type: "string" },
              description: "External dependencies needed",
            },
          },
          description: "Structured architecture plan (for architect tasks)",
        },
      },
      required: ["agent_id", "task_type", "title", "description"],
    },
  },
  {
    name: "get_queue_status",
    description: "Get queue depth and status for all agent queues",
    inputSchema: {
      type: "object",
      properties: {},
      required: [],
    },
  },
  {
    name: "list_pending_tasks",
    description: "List all pending tasks for a specific agent queue",
    inputSchema: {
      type: "object",
      properties: {
        agent_id: {
          type: "string",
          enum: ["engineer", "qa", "architect", "product-owner", "code-reviewer", "testing", "static-analysis", "repo-analyzer"],
          description: "Agent queue to list tasks from",
        },
      },
      required: ["agent_id"],
    },
  },
  {
    name: "get_task_details",
    description: "Get full details of a task by its ID",
    inputSchema: {
      type: "object",
      properties: {
        task_id: {
          type: "string",
          description: "Task ID to retrieve",
        },
      },
      required: ["task_id"],
    },
  },
  {
    name: "get_epic_progress",
    description: "Get progress summary for all tasks associated with an epic",
    inputSchema: {
      type: "object",
      properties: {
        epic_key: {
          type: "string",
          description: "JIRA epic key (e.g., PROJ-100)",
        },
      },
      required: ["epic_key"],
    },
  },
  {
    name: "consult_agent",
    description:
      "Consult another agent for expert advice. Use this when you need architectural guidance, QA strategy, or implementation help from a specialist. Returns a concise response. Limited to 5 consultations per session.",
    inputSchema: {
      type: "object",
      properties: {
        target_agent: {
          type: "string",
          enum: ["engineer", "qa", "architect", "product-owner", "code-reviewer", "repo-analyzer"],
          description: "Agent to consult for expert advice",
        },
        question: {
          type: "string",
          description: "The question to ask the expert agent",
        },
        context: {
          type: "string",
          description: "Optional context about your current situation to help the expert give relevant advice",
        },
      },
      required: ["target_agent", "question"],
    },
  },
  {
    name: "debate_topic",
    description:
      "Spawn a multi-perspective debate on a complex decision. An Advocate argues in favor, a Critic argues against, and an Arbiter synthesizes both into a recommendation with trade-offs and confidence level. Uses 2 consultation slots. Best for architectural choices, approach decisions, or trade-off analysis.",
    inputSchema: {
      type: "object",
      properties: {
        topic: {
          type: "string",
          description: "The decision or approach to debate (e.g., 'Should we use Redis or in-memory caching?')",
        },
        context: {
          type: "string",
          description: "Optional context about the situation, requirements, or constraints",
        },
        custom_perspectives: {
          type: "object",
          properties: {
            advocate: {
              type: "string",
              description: "Custom perspective for the advocate (default: argue in favor)",
            },
            critic: {
              type: "string",
              description: "Custom perspective for the critic (default: argue against)",
            },
          },
          description: "Optional custom perspectives instead of default advocate/critic roles",
        },
      },
      required: ["topic"],
    },
  },
  {
    name: "share_knowledge",
    description:
      "Share a discovery or insight with other agents via the shared knowledge base. Use this to store reusable information like repo structure, test frameworks, or conventions discovered during your work.",
    inputSchema: {
      type: "object",
      properties: {
        topic: {
          type: "string",
          description: "Knowledge topic category (e.g., 'repo-structure', 'conventions', 'dependencies')",
        },
        key: {
          type: "string",
          description: "Specific key within the topic (e.g., 'test_framework', 'primary_language')",
        },
        value: {
          type: "string",
          description: "The knowledge value to store",
        },
      },
      required: ["topic", "key", "value"],
    },
  },
  {
    name: "get_knowledge",
    description:
      "Read from the shared knowledge base. Use this to check what other agents have discovered about the repo, conventions, or other shared context before starting work.",
    inputSchema: {
      type: "object",
      properties: {
        topic: {
          type: "string",
          description: "Knowledge topic to read (e.g., 'repo-structure', 'conventions')",
        },
        key: {
          type: "string",
          description: "Optional specific key to read. Omit to get all entries for the topic.",
        },
        max_age_hours: {
          type: "number",
          description: "Maximum age in hours for entries (default: 24). Older entries are marked stale.",
        },
      },
      required: ["topic"],
    },
  },
  {
    name: "transfer_to_engineer",
    description:
      "Signal that this task's work should continue with the engineer agent. Use when implementation or a code fix is needed next. The framework validates and routes accordingly.",
    inputSchema: {
      type: "object",
      properties: {
        reason: {
          type: "string",
          description: "Why the task should go to engineer (e.g., 'Architecture plan ready, needs implementation')",
        },
      },
      required: ["reason"],
    },
  },
  {
    name: "transfer_to_qa",
    description:
      "Signal that this task's work should continue with the QA agent. Use when the work needs review or verification next. The framework validates and routes accordingly.",
    inputSchema: {
      type: "object",
      properties: {
        reason: {
          type: "string",
          description: "Why the task should go to QA (e.g., 'PR created with passing tests, ready for code review')",
        },
      },
      required: ["reason"],
    },
  },
  {
    name: "transfer_to_architect",
    description:
      "Signal that this task's work should continue with the architect agent. Use when work needs re-planning, escalation, or architectural review. The framework validates and routes accordingly.",
    inputSchema: {
      type: "object",
      properties: {
        reason: {
          type: "string",
          description: "Why the task should go to architect (e.g., 'Requirements unclear, need architectural guidance')",
        },
      },
      required: ["reason"],
    },
  },
  {
    name: "mark_workflow_complete",
    description:
      "Signal that no further agent work is needed for this task. Use when the task is fully done and no handoff to another agent is required. The framework validates this (e.g., won't honor if a PR still needs review).",
    inputSchema: {
      type: "object",
      properties: {
        reason: {
          type: "string",
          description: "Why the workflow is complete (e.g., 'Documentation updated, no code changes needed')",
        },
      },
      required: ["reason"],
    },
  },
];

const server = new Server(
  {
    name: "task-queue-mcp-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: TOOLS };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    logger.info(`Tool called: ${name}`, { args });

    let result;

    switch (name) {
      case "queue_task_for_agent": {
        const input = args as unknown as QueueTaskInput;
        // Each agent subprocess sets AGENT_ID in its environment
        const createdBy = process.env.AGENT_ID || "mcp-client";
        result = await queueTaskForAgent(workspace, input, createdBy);
        break;
      }
      case "get_queue_status":
        result = getQueueStatus(workspace);
        break;
      case "list_pending_tasks": {
        const agentId = (args as { agent_id: AgentId }).agent_id;
        result = listPendingTasks(workspace, agentId);
        break;
      }
      case "get_task_details": {
        const taskId = (args as { task_id: string }).task_id;
        result = getTaskDetails(workspace, taskId);
        if (!result) {
          result = { error: `Task ${taskId} not found` };
        }
        break;
      }
      case "get_epic_progress": {
        const epicKey = (args as { epic_key: string }).epic_key;
        result = getEpicProgress(workspace, epicKey);
        break;
      }
      case "consult_agent": {
        const { target_agent, question, context } = args as Record<string, unknown>;
        if (typeof target_agent !== "string" || typeof question !== "string") {
          throw new Error("consult_agent requires string target_agent and question");
        }
        result = consultAgent(
          workspace,
          target_agent,
          question,
          typeof context === "string" ? context : undefined,
        );
        break;
      }
      case "share_knowledge": {
        const { topic, key, value } = args as Record<string, unknown>;
        if (typeof topic !== "string" || typeof key !== "string" || typeof value !== "string") {
          throw new Error("share_knowledge requires string topic, key, and value");
        }
        result = shareKnowledge(workspace, topic, key, value);
        break;
      }
      case "get_knowledge": {
        const { topic: getTopic, key: getKey, max_age_hours } = args as Record<string, unknown>;
        if (typeof getTopic !== "string") {
          throw new Error("get_knowledge requires string topic");
        }
        result = getKnowledge(
          workspace,
          getTopic,
          typeof getKey === "string" ? getKey : undefined,
          typeof max_age_hours === "number" ? max_age_hours : undefined,
        );
        break;
      }
      case "debate_topic": {
        const { topic, context, custom_perspectives } = args as Record<string, unknown>;
        if (typeof topic !== "string") {
          throw new Error("debate_topic requires string topic");
        }
        const perspectives = custom_perspectives as { advocate?: string; critic?: string } | undefined;
        result = await debateTopic(
          workspace,
          topic,
          typeof context === "string" ? context : undefined,
          perspectives && typeof perspectives.advocate === "string" && typeof perspectives.critic === "string"
            ? { advocate: perspectives.advocate, critic: perspectives.critic }
            : undefined,
          getRemainingConsultations,
          decrementConsultations,
        );
        break;
      }
      case "transfer_to_engineer":
      case "transfer_to_qa":
      case "transfer_to_architect":
      case "mark_workflow_complete": {
        const reason = (args as { reason: string }).reason;
        if (typeof reason !== "string" || !reason.trim()) {
          throw new Error(`${name} requires a non-empty reason string`);
        }
        // Must match WORKFLOW_COMPLETE in core/routing.py
        const targetMap: Record<string, string> = {
          transfer_to_engineer: "engineer",
          transfer_to_qa: "qa",
          transfer_to_architect: "architect",
          mark_workflow_complete: "__complete__",
        };
        const targetAgent = targetMap[name];
        const taskId = process.env.AGENT_TASK_ID;
        const sourceAgent = process.env.AGENT_ID || "unknown";
        if (!taskId) {
          throw new Error(`${name}: AGENT_TASK_ID not set in environment`);
        }
        result = writeRoutingSignal(workspace, taskId, targetAgent, reason, sourceAgent);
        break;
      }
      default:
        throw new Error(`Unknown tool: ${name}`);
    }

    logger.info(`Tool completed: ${name}`, { result });

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(result, null, 2),
        },
      ],
    };
  } catch (error: unknown) {
    const err = error as Error & { status?: string };
    logger.error(`Tool failed: ${name}`, { error: err.message, stack: err.stack });

    const errorResponse = {
      success: false,
      error: {
        code: err.status || "UNKNOWN_ERROR",
        message: err.message || "An unknown error occurred",
      },
    };

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(errorResponse, null, 2),
        },
      ],
      isError: true,
    };
  }
});

async function main() {
  logger.info("Starting Task Queue MCP server");

  const transport = new StdioServerTransport();
  await server.connect(transport);

  logger.info("Task Queue MCP server running", { workspace });
}

main().catch((error) => {
  logger.error("Fatal error", { error: error.message, stack: error.stack });
  process.exit(1);
});
