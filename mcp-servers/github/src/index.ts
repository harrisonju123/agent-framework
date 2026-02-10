#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";
import { Octokit } from "@octokit/rest";
import { createLogger } from "./logger.js";
import {
  createBranch,
  createPR,
  addPRComment,
  getPRByBranch,
  getPR,
  getPRComments,
  getPRDiff,
  getCheckRuns,
  linkPRToJira,
  cloneRepo,
  commitChanges,
  pushBranch,
  updatePR,
} from "./github-tools.js";

const logger = createLogger();

const octokit = new Octokit({
  auth: process.env.GITHUB_TOKEN,
});

const TOOLS: Tool[] = [
  {
    name: "github_create_branch",
    description: "Create a new branch in a GitHub repository",
    inputSchema: {
      type: "object",
      properties: {
        owner: {
          type: "string",
          description: "Repository owner (organization or user)",
        },
        repo: {
          type: "string",
          description: "Repository name",
        },
        branchName: {
          type: "string",
          description: "Name of the new branch",
        },
        fromBranch: {
          type: "string",
          description: "Base branch to create from (default: main)",
          default: "main",
        },
      },
      required: ["owner", "repo", "branchName"],
    },
  },
  {
    name: "github_create_pr",
    description: "Create a new pull request",
    inputSchema: {
      type: "object",
      properties: {
        owner: {
          type: "string",
          description: "Repository owner",
        },
        repo: {
          type: "string",
          description: "Repository name",
        },
        title: {
          type: "string",
          description: "PR title",
        },
        body: {
          type: "string",
          description: "PR description",
        },
        head: {
          type: "string",
          description: "Branch name containing changes",
        },
        base: {
          type: "string",
          description: "Base branch to merge into (default: main)",
          default: "main",
        },
        draft: {
          type: "boolean",
          description: "Create as draft PR",
          default: false,
        },
        labels: {
          type: "array",
          items: { type: "string" },
          description: "Labels to add to PR",
        },
      },
      required: ["owner", "repo", "title", "body", "head"],
    },
  },
  {
    name: "github_add_pr_comment",
    description: "Add a comment to a pull request",
    inputSchema: {
      type: "object",
      properties: {
        owner: {
          type: "string",
          description: "Repository owner",
        },
        repo: {
          type: "string",
          description: "Repository name",
        },
        prNumber: {
          type: "number",
          description: "Pull request number",
        },
        body: {
          type: "string",
          description: "Comment text",
        },
      },
      required: ["owner", "repo", "prNumber", "body"],
    },
  },
  {
    name: "github_get_pr_by_branch",
    description: "Find a pull request by branch name",
    inputSchema: {
      type: "object",
      properties: {
        owner: {
          type: "string",
          description: "Repository owner",
        },
        repo: {
          type: "string",
          description: "Repository name",
        },
        branchName: {
          type: "string",
          description: "Branch name to search for",
        },
      },
      required: ["owner", "repo", "branchName"],
    },
  },
  {
    name: "github_link_pr_to_jira",
    description: "Update PR body to include JIRA ticket link",
    inputSchema: {
      type: "object",
      properties: {
        owner: {
          type: "string",
          description: "Repository owner",
        },
        repo: {
          type: "string",
          description: "Repository name",
        },
        prNumber: {
          type: "number",
          description: "Pull request number",
        },
        jiraKey: {
          type: "string",
          description: "JIRA ticket key (e.g., 'PROJ-123')",
        },
      },
      required: ["owner", "repo", "prNumber", "jiraKey"],
    },
  },
  {
    name: "github_clone_repo",
    description: "Clone a GitHub repository to a local path",
    inputSchema: {
      type: "object",
      properties: {
        owner: {
          type: "string",
          description: "Repository owner",
        },
        repo: {
          type: "string",
          description: "Repository name",
        },
        localPath: {
          type: "string",
          description: "Local filesystem path to clone to",
        },
      },
      required: ["owner", "repo", "localPath"],
    },
  },
  {
    name: "github_commit_changes",
    description: "Stage and commit all changes in a local repository",
    inputSchema: {
      type: "object",
      properties: {
        localPath: {
          type: "string",
          description: "Path to local repository",
        },
        message: {
          type: "string",
          description: "Commit message",
        },
      },
      required: ["localPath", "message"],
    },
  },
  {
    name: "github_push_branch",
    description: "Push a branch to the remote origin",
    inputSchema: {
      type: "object",
      properties: {
        localPath: {
          type: "string",
          description: "Path to local repository",
        },
        branchName: {
          type: "string",
          description: "Branch name to push",
        },
      },
      required: ["localPath", "branchName"],
    },
  },
  {
    name: "github_update_pr",
    description: "Update a pull request's title, description, or state",
    inputSchema: {
      type: "object",
      properties: {
        owner: {
          type: "string",
          description: "Repository owner",
        },
        repo: {
          type: "string",
          description: "Repository name",
        },
        prNumber: {
          type: "number",
          description: "Pull request number",
        },
        title: {
          type: "string",
          description: "New PR title (optional)",
        },
        body: {
          type: "string",
          description: "New PR description (optional)",
        },
        state: {
          type: "string",
          enum: ["open", "closed"],
          description: "PR state (optional)",
        },
      },
      required: ["owner", "repo", "prNumber"],
    },
  },
  {
    name: "github_get_pr",
    description: "Fetch pull request details including changed files with patches",
    inputSchema: {
      type: "object",
      properties: {
        owner: {
          type: "string",
          description: "Repository owner",
        },
        repo: {
          type: "string",
          description: "Repository name",
        },
        prNumber: {
          type: "number",
          description: "Pull request number",
        },
      },
      required: ["owner", "repo", "prNumber"],
    },
  },
  {
    name: "github_get_pr_comments",
    description: "Fetch all issue comments and inline review comments on a pull request",
    inputSchema: {
      type: "object",
      properties: {
        owner: {
          type: "string",
          description: "Repository owner",
        },
        repo: {
          type: "string",
          description: "Repository name",
        },
        prNumber: {
          type: "number",
          description: "Pull request number",
        },
      },
      required: ["owner", "repo", "prNumber"],
    },
  },
  {
    name: "github_get_pr_diff",
    description: "Fetch the raw diff for a pull request",
    inputSchema: {
      type: "object",
      properties: {
        owner: {
          type: "string",
          description: "Repository owner",
        },
        repo: {
          type: "string",
          description: "Repository name",
        },
        prNumber: {
          type: "number",
          description: "Pull request number",
        },
      },
      required: ["owner", "repo", "prNumber"],
    },
  },
  {
    name: "github_get_check_runs",
    description: "Fetch CI/CD check run status for a git ref (branch, tag, or SHA)",
    inputSchema: {
      type: "object",
      properties: {
        owner: {
          type: "string",
          description: "Repository owner",
        },
        repo: {
          type: "string",
          description: "Repository name",
        },
        ref: {
          type: "string",
          description: "Git ref to check (branch name, tag, or commit SHA)",
        },
      },
      required: ["owner", "repo", "ref"],
    },
  },
];

const server = new Server(
  {
    name: "github-mcp-server",
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
      case "github_create_branch":
        result = await createBranch(octokit, args as any);
        break;
      case "github_create_pr":
        result = await createPR(octokit, args as any);
        break;
      case "github_add_pr_comment":
        result = await addPRComment(octokit, args as any);
        break;
      case "github_get_pr_by_branch":
        result = await getPRByBranch(octokit, args as any);
        break;
      case "github_link_pr_to_jira":
        result = await linkPRToJira(octokit, args as any);
        break;
      case "github_clone_repo":
        result = await cloneRepo(args as any);
        break;
      case "github_commit_changes":
        result = await commitChanges(args as any);
        break;
      case "github_push_branch":
        result = await pushBranch(args as any);
        break;
      case "github_update_pr":
        result = await updatePR(octokit, args as any);
        break;
      case "github_get_pr":
        result = await getPR(octokit, args as any);
        break;
      case "github_get_pr_comments":
        result = await getPRComments(octokit, args as any);
        break;
      case "github_get_pr_diff":
        result = await getPRDiff(octokit, args as any);
        break;
      case "github_get_check_runs":
        result = await getCheckRuns(octokit, args as any);
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
        code: error.status || "UNKNOWN_ERROR",
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
  logger.info("Starting GitHub MCP server");

  if (!process.env.GITHUB_TOKEN) {
    logger.error("Missing required environment variable: GITHUB_TOKEN");
    process.exit(1);
  }

  const transport = new StdioServerTransport();
  await server.connect(transport);

  logger.info("GitHub MCP server running");
}

main().catch((error) => {
  logger.error("Fatal error", { error: error.message, stack: error.stack });
  process.exit(1);
});
