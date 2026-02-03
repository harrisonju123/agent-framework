#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import JiraClient from "jira-client";
import { createLogger } from "./logger.js";
import {
  searchIssues,
  getIssue,
  createIssue,
  createEpic,
  createSubtask,
  transitionIssue,
  addComment,
  updateField,
  createEpicWithSubtasks,
} from "./jira-tools.js";

const logger = createLogger();

const jiraClient = new JiraClient({
  protocol: "https",
  host: process.env.JIRA_SERVER || "",
  username: process.env.JIRA_EMAIL || "",
  password: process.env.JIRA_API_TOKEN || "",
  apiVersion: "2",
  strictSSL: true,
});

const TOOLS: Tool[] = [
  {
    name: "jira_search_issues",
    description: "Search JIRA issues using JQL (JIRA Query Language)",
    inputSchema: {
      type: "object",
      properties: {
        jql: {
          type: "string",
          description: "JQL query string (e.g., 'project = PROJ AND status = Open')",
        },
        maxResults: {
          type: "number",
          description: "Maximum number of results to return (default: 50)",
          default: 50,
        },
        fields: {
          type: "array",
          items: { type: "string" },
          description: "Fields to include in response (default: key, summary, status)",
        },
      },
      required: ["jql"],
    },
  },
  {
    name: "jira_get_issue",
    description: "Get detailed information about a JIRA issue",
    inputSchema: {
      type: "object",
      properties: {
        issueKey: {
          type: "string",
          description: "JIRA issue key (e.g., 'PROJ-123')",
        },
      },
      required: ["issueKey"],
    },
  },
  {
    name: "jira_create_issue",
    description: "Create a new JIRA issue (Story, Bug, or Task)",
    inputSchema: {
      type: "object",
      properties: {
        project: {
          type: "string",
          description: "JIRA project key",
        },
        summary: {
          type: "string",
          description: "Issue summary/title",
        },
        description: {
          type: "string",
          description: "Issue description",
        },
        issueType: {
          type: "string",
          enum: ["Story", "Bug", "Task"],
          description: "Type of issue to create",
        },
        labels: {
          type: "array",
          items: { type: "string" },
          description: "Labels to add to the issue",
        },
      },
      required: ["project", "summary", "description", "issueType"],
    },
  },
  {
    name: "jira_create_epic",
    description: "Create a new JIRA epic",
    inputSchema: {
      type: "object",
      properties: {
        project: {
          type: "string",
          description: "JIRA project key",
        },
        title: {
          type: "string",
          description: "Epic title",
        },
        description: {
          type: "string",
          description: "Epic description",
        },
      },
      required: ["project", "title", "description"],
    },
  },
  {
    name: "jira_create_subtask",
    description: "Create a subtask under a parent issue",
    inputSchema: {
      type: "object",
      properties: {
        parentKey: {
          type: "string",
          description: "Parent issue key (e.g., 'PROJ-123')",
        },
        summary: {
          type: "string",
          description: "Subtask summary/title",
        },
        description: {
          type: "string",
          description: "Subtask description",
        },
      },
      required: ["parentKey", "summary", "description"],
    },
  },
  {
    name: "jira_transition_issue",
    description: "Change the status of a JIRA issue",
    inputSchema: {
      type: "object",
      properties: {
        issueKey: {
          type: "string",
          description: "JIRA issue key (e.g., 'PROJ-123')",
        },
        transitionName: {
          type: "string",
          description: "Name of the transition (e.g., 'In Progress', 'Done')",
        },
      },
      required: ["issueKey", "transitionName"],
    },
  },
  {
    name: "jira_add_comment",
    description: "Add a comment to a JIRA issue",
    inputSchema: {
      type: "object",
      properties: {
        issueKey: {
          type: "string",
          description: "JIRA issue key (e.g., 'PROJ-123')",
        },
        comment: {
          type: "string",
          description: "Comment text",
        },
      },
      required: ["issueKey", "comment"],
    },
  },
  {
    name: "jira_update_field",
    description: "Update a custom field on a JIRA issue",
    inputSchema: {
      type: "object",
      properties: {
        issueKey: {
          type: "string",
          description: "JIRA issue key (e.g., 'PROJ-123')",
        },
        fieldName: {
          type: "string",
          description: "Field name to update",
        },
        value: {
          description: "New value for the field",
        },
      },
      required: ["issueKey", "fieldName", "value"],
    },
  },
  {
    name: "jira_create_epic_with_subtasks",
    description: "Create an epic with multiple subtasks in one operation",
    inputSchema: {
      type: "object",
      properties: {
        project: {
          type: "string",
          description: "JIRA project key",
        },
        epicTitle: {
          type: "string",
          description: "Epic title",
        },
        epicDescription: {
          type: "string",
          description: "Epic description",
        },
        subtasks: {
          type: "array",
          items: {
            type: "object",
            properties: {
              summary: { type: "string" },
              description: { type: "string" },
            },
            required: ["summary", "description"],
          },
          description: "Array of subtasks to create",
        },
      },
      required: ["project", "epicTitle", "epicDescription", "subtasks"],
    },
  },
];

const server = new Server(
  {
    name: "jira-mcp-server",
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
      case "jira_search_issues":
        result = await searchIssues(jiraClient, args as any);
        break;
      case "jira_get_issue":
        result = await getIssue(jiraClient, args as any);
        break;
      case "jira_create_issue":
        result = await createIssue(jiraClient, args as any);
        break;
      case "jira_create_epic":
        result = await createEpic(jiraClient, args as any);
        break;
      case "jira_create_subtask":
        result = await createSubtask(jiraClient, args as any);
        break;
      case "jira_transition_issue":
        result = await transitionIssue(jiraClient, args as any);
        break;
      case "jira_add_comment":
        result = await addComment(jiraClient, args as any);
        break;
      case "jira_update_field":
        result = await updateField(jiraClient, args as any);
        break;
      case "jira_create_epic_with_subtasks":
        result = await createEpicWithSubtasks(jiraClient, args as any);
        break;
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
  } catch (error: any) {
    logger.error(`Tool failed: ${name}`, { error: error.message, stack: error.stack });

    const errorResponse = {
      success: false,
      error: {
        code: error.statusCode || "UNKNOWN_ERROR",
        message: error.message || "An unknown error occurred",
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
  logger.info("Starting JIRA MCP server");

  if (!process.env.JIRA_SERVER || !process.env.JIRA_EMAIL || !process.env.JIRA_API_TOKEN) {
    logger.error("Missing required environment variables: JIRA_SERVER, JIRA_EMAIL, JIRA_API_TOKEN");
    process.exit(1);
  }

  const transport = new StdioServerTransport();
  await server.connect(transport);

  logger.info("JIRA MCP server running");
}

main().catch((error) => {
  logger.error("Fatal error", { error: error.message, stack: error.stack });
  process.exit(1);
});
