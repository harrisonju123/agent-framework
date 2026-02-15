/**
 * Inter-agent consultation via Claude CLI.
 *
 * Allows agents to consult other agent personas for expert advice
 * without spawning a full agent session. Uses haiku with --print
 * (no tool access) and --max-turns 1 to prevent consultation loops.
 */

import { execFileSync } from "child_process";
import { readFileSync, existsSync, writeFileSync, mkdirSync, statSync } from "fs";
import { join } from "path";
import yaml from "js-yaml";
import { createLogger } from "./logger.js";

const logger = createLogger();

// Rate limiting: track consultations per server lifetime
let consultationCount = 0;
const MAX_CONSULTATIONS = Math.min(
  parseInt(process.env.MAX_CONSULTATIONS_PER_SESSION || "5", 10),
  20, // hard upper bound
);

/**
 * Get remaining consultation slots.
 * Used by debate system to check availability before spawning perspectives.
 */
export function getRemainingConsultations(): number {
  return MAX_CONSULTATIONS - consultationCount;
}

/**
 * Decrement consultation counter by a given amount.
 * Used by debate system to reserve slots (debates cost 2 slots).
 * Returns false if insufficient slots available.
 */
export function decrementConsultations(count: number): boolean {
  if (consultationCount + count > MAX_CONSULTATIONS) {
    return false;
  }
  consultationCount += count;
  return true;
}

// Input length limits
const MAX_QUESTION_LENGTH = 2000;
const MAX_CONTEXT_LENGTH = 1000;
const MAX_EXPERTISE_SNIPPET = 2000;

interface AgentConfig {
  id: string;
  name: string;
  prompt: string;
}

interface AgentsYamlSchema {
  agents: Array<{
    id: string;
    name: string;
    prompt?: string;
    [key: string]: unknown;
  }>;
}

export interface ConsultationResult {
  success: boolean;
  agent: string;
  response: string;
  consultation_id: string;
  consultations_remaining: number;
}

// Cache parsed config to avoid re-reading on every consultation
let cachedAgents: AgentConfig[] | null = null;
let cachedMtime: number = 0;

function loadAgentsConfig(workspace: string): AgentConfig[] {
  const configPath = join(workspace, "config", "agents.yaml");
  if (!existsSync(configPath)) {
    throw new Error(`agents.yaml not found at ${configPath}`);
  }

  const { mtimeMs } = statSync(configPath);
  if (cachedAgents && mtimeMs === cachedMtime) {
    return cachedAgents;
  }

  const raw = readFileSync(configPath, "utf-8");
  const data = yaml.load(raw) as AgentsYamlSchema;

  if (!data?.agents || !Array.isArray(data.agents)) {
    throw new Error("Invalid agents.yaml: missing 'agents' array");
  }

  cachedAgents = data.agents
    .filter((a) => a.id && a.name)
    .map((a) => ({
      id: a.id,
      name: a.name,
      prompt: typeof a.prompt === "string" ? a.prompt : "",
    }));
  cachedMtime = mtimeMs;

  return cachedAgents;
}

function sanitizeLogId(id: string): string {
  return id.replace(/[^a-zA-Z0-9_-]/g, "_").substring(0, 100);
}

function logConsultation(
  workspace: string,
  id: string,
  data: Record<string, unknown>,
): void {
  const logDir = join(workspace, ".agent-communication", "consultations");
  if (!existsSync(logDir)) {
    mkdirSync(logDir, { recursive: true });
  }
  const safeId = sanitizeLogId(id);
  writeFileSync(join(logDir, `${safeId}.json`), JSON.stringify(data, null, 2));
}

/**
 * Build a minimal env for the consultation subprocess.
 * Only passes what Claude CLI needs — no leaked credentials.
 */
function buildConsultationEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  // Claude CLI needs these to function
  const allowedVars = [
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "TERM",
    "ANTHROPIC_API_KEY",
    // macOS-specific
    "TMPDIR",
  ];
  for (const key of allowedVars) {
    if (process.env[key]) {
      env[key] = process.env[key]!;
    }
  }
  return env;
}

export function consultAgent(
  workspace: string,
  targetAgent: string,
  question: string,
  callerContext?: string,
): ConsultationResult {
  const consultationId = `consult-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

  // Rate limit check
  if (consultationCount >= MAX_CONSULTATIONS) {
    return {
      success: false,
      agent: targetAgent,
      response:
        `Consultation limit reached (${MAX_CONSULTATIONS} per session). Proceed with your best judgment.`,
      consultation_id: consultationId,
      consultations_remaining: 0,
    };
  }

  // Input length validation
  if (question.length > MAX_QUESTION_LENGTH) {
    question = question.substring(0, MAX_QUESTION_LENGTH) + "... [truncated]";
  }
  if (callerContext && callerContext.length > MAX_CONTEXT_LENGTH) {
    callerContext = callerContext.substring(0, MAX_CONTEXT_LENGTH) + "... [truncated]";
  }

  const agents = loadAgentsConfig(workspace);
  const agent = agents.find((a) => a.id === targetAgent);
  if (!agent) {
    return {
      success: false,
      agent: targetAgent,
      response: `Agent '${targetAgent}' not found in config. Available: ${agents.map((a) => a.id).join(", ")}`,
      consultation_id: consultationId,
      consultations_remaining: MAX_CONSULTATIONS - consultationCount,
    };
  }

  const expertiseSnippet = agent.prompt.substring(0, MAX_EXPERTISE_SNIPPET);

  const prompt = [
    `You are ${agent.name} (${agent.id}). Another agent is consulting you for expert advice.`,
    `\nYour expertise:\n${expertiseSnippet}`,
    callerContext ? `\nCaller's context: ${callerContext}` : "",
    `\nQuestion: ${question}`,
    `\nProvide a concise, actionable answer (2-5 sentences). Focus on practical guidance.`,
  ].join("\n");

  try {
    // --print mode: text-only response, no tool use, no MCP access.
    // No --dangerously-skip-permissions — consultation is read-only advice.
    const result = execFileSync(
      "claude",
      [
        "--print",
        "--model",
        "haiku",
        "--max-turns",
        "1",
      ],
      {
        input: prompt,
        timeout: 60_000,
        encoding: "utf-8",
        cwd: workspace,
        env: buildConsultationEnv(),
      },
    );

    const response = result.trim();
    consultationCount++;

    logConsultation(workspace, consultationId, {
      target_agent: targetAgent,
      question,
      response,
      caller_context: callerContext,
      timestamp: new Date().toISOString(),
    });

    logger.info(`Consultation completed: ${targetAgent}`, {
      consultation_id: consultationId,
    });

    return {
      success: true,
      agent: targetAgent,
      response,
      consultation_id: consultationId,
      consultations_remaining: MAX_CONSULTATIONS - consultationCount,
    };
  } catch (error: unknown) {
    const err = error as Error;
    logger.error(`Consultation failed: ${targetAgent}`, {
      error: err.message,
    });

    return {
      success: false,
      agent: targetAgent,
      response: `Consultation failed: ${err.message}. Proceed with your best judgment.`,
      consultation_id: consultationId,
      consultations_remaining: MAX_CONSULTATIONS - consultationCount,
    };
  }
}
