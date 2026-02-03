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
} from "./queue-tools.js";
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
          enum: ["engineer", "qa", "architect", "product-owner"],
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
          enum: ["engineer", "qa", "architect", "product-owner"],
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
        // Determine who is calling (from MCP context or default to "mcp-client")
        const createdBy = "product-owner"; // In practice, this would come from the calling agent's context
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
