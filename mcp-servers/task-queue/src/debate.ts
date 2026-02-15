/**
 * Multi-perspective debate system for adversarial reasoning.
 *
 * Spawns Advocate and Critic agents in parallel to argue different
 * perspectives on a complex decision, then an Arbiter synthesizes
 * a final recommendation with trade-offs and confidence level.
 *
 * Uses the same infrastructure as consultation.ts but orchestrates
 * multiple perspectives simultaneously.
 */

import { execFileSync } from "child_process";
import { writeFileSync, mkdirSync, existsSync } from "fs";
import { join } from "path";
import { createLogger } from "./logger.js";

const logger = createLogger();

// Debate counts against the consultation rate limit (2 slots per debate)
const DEBATE_COST = 2;

// Input validation limits
const MAX_TOPIC_LENGTH = 1500;
const MAX_DEBATE_CONTEXT_LENGTH = 3000;

export interface PerspectiveArgument {
  perspective: string;
  argument: string;
  success: boolean;
  error?: string;
}

export interface DebateResult {
  success: boolean;
  topic: string;
  advocate: PerspectiveArgument;
  critic: PerspectiveArgument;
  synthesis: {
    recommendation: string;
    confidence: "high" | "medium" | "low";
    trade_offs: string[];
    reasoning: string;
  };
  debate_id: string;
  consultations_used: number;
  consultations_remaining: number;
}

/**
 * Build a minimal env for debate subprocesses.
 * Reuses the same pattern as consultation.ts.
 */
function buildDebateEnv(): Record<string, string> {
  const env: Record<string, string> = {};
  const allowedVars = [
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "TERM",
    "ANTHROPIC_API_KEY",
    "TMPDIR",
  ];
  for (const key of allowedVars) {
    if (process.env[key]) {
      env[key] = process.env[key]!;
    }
  }
  return env;
}

function sanitizeDebateId(id: string): string {
  return id.replace(/[^a-zA-Z0-9_-]/g, "_").substring(0, 100);
}

function logDebate(
  workspace: string,
  id: string,
  data: Record<string, unknown>,
): void {
  const logDir = join(workspace, ".agent-communication", "debates");
  if (!existsSync(logDir)) {
    mkdirSync(logDir, { recursive: true });
  }
  const safeId = sanitizeDebateId(id);
  writeFileSync(join(logDir, `${safeId}.json`), JSON.stringify(data, null, 2));
}

/**
 * Spawn a single perspective agent with a specific role.
 */
function spawnPerspective(
  workspace: string,
  perspective: string,
  topic: string,
  context: string,
  timeout: number,
): PerspectiveArgument {
  const prompt = [
    `You are participating in a structured debate to help make a complex decision.`,
    `Your role: ${perspective}`,
    `Topic: ${topic}`,
    context ? `\nContext: ${context}` : "",
    `\nProvide your argument in 3-5 sentences. Be specific and focus on practical implications.`,
    `Your goal is to argue your assigned perspective, even if you see merit in the other side.`,
  ].join("\n");

  try {
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
        timeout,
        encoding: "utf-8",
        cwd: workspace,
        env: buildDebateEnv(),
      },
    );

    return {
      perspective,
      argument: result.trim(),
      success: true,
    };
  } catch (error: unknown) {
    const err = error as Error;
    logger.error(`Perspective ${perspective} failed`, { error: err.message });
    return {
      perspective,
      argument: "",
      success: false,
      error: err.message,
    };
  }
}

/**
 * Arbiter synthesizes both perspectives into a final recommendation.
 */
function synthesizeArguments(
  workspace: string,
  topic: string,
  advocate: PerspectiveArgument,
  critic: PerspectiveArgument,
  timeout: number,
): {
  recommendation: string;
  confidence: "high" | "medium" | "low";
  trade_offs: string[];
  reasoning: string;
} {
  const prompt = [
    `You are an Arbiter synthesizing two perspectives on a complex decision.`,
    `\nTopic: ${topic}`,
    `\nAdvocate's argument:\n${advocate.argument}`,
    `\nCritic's argument:\n${critic.argument}`,
    `\nYour task:`,
    `1. Provide a clear recommendation (2-3 sentences)`,
    `2. State your confidence level: high, medium, or low`,
    `3. List 2-4 key trade-offs to consider`,
    `4. Explain your reasoning (2-3 sentences)`,
    `\nFormat your response as JSON:`,
    `{`,
    `  "recommendation": "...",`,
    `  "confidence": "high|medium|low",`,
    `  "trade_offs": ["...", "..."],`,
    `  "reasoning": "..."`,
    `}`,
  ].join("\n");

  try {
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
        timeout,
        encoding: "utf-8",
        cwd: workspace,
        env: buildDebateEnv(),
      },
    );

    // Parse JSON from the response
    const text = result.trim();
    // Extract JSON from markdown code blocks if present
    const jsonMatch = text.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```/) ||
                      text.match(/(\{[\s\S]*\})/);

    if (!jsonMatch) {
      throw new Error("Could not extract JSON from arbiter response");
    }

    const synthesis = JSON.parse(jsonMatch[1]);

    // Validate structure
    if (!synthesis.recommendation || !synthesis.confidence || !Array.isArray(synthesis.trade_offs)) {
      throw new Error("Invalid synthesis structure");
    }

    return {
      recommendation: synthesis.recommendation,
      confidence: synthesis.confidence as "high" | "medium" | "low",
      trade_offs: synthesis.trade_offs,
      reasoning: synthesis.reasoning || "No reasoning provided",
    };
  } catch (error: unknown) {
    const err = error as Error;
    logger.error("Arbiter synthesis failed", { error: err.message });

    // Fallback synthesis
    return {
      recommendation: "Unable to synthesize perspectives due to parsing error. Review both arguments and decide based on your judgment.",
      confidence: "low",
      trade_offs: ["Synthesis failed - manual review needed"],
      reasoning: `Arbiter failed: ${err.message}`,
    };
  }
}

/**
 * Main debate function - coordinates advocate, critic, and arbiter.
 */
export function debateTopic(
  workspace: string,
  topic: string,
  context?: string,
  customPerspectives?: { advocate: string; critic: string },
  getRemainingConsultations?: () => number,
  decrementConsultations?: (count: number) => boolean,
): DebateResult {
  const debateId = `debate-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

  // Input validation
  if (topic.length > MAX_TOPIC_LENGTH) {
    topic = topic.substring(0, MAX_TOPIC_LENGTH) + "... [truncated]";
  }
  if (context && context.length > MAX_DEBATE_CONTEXT_LENGTH) {
    context = context.substring(0, MAX_DEBATE_CONTEXT_LENGTH) + "... [truncated]";
  }

  // Check if we have enough consultation slots (debates cost 2 slots)
  if (getRemainingConsultations && decrementConsultations) {
    const remaining = getRemainingConsultations();
    if (remaining < DEBATE_COST) {
      return {
        success: false,
        topic,
        advocate: { perspective: "advocate", argument: "", success: false, error: "Insufficient consultations" },
        critic: { perspective: "critic", argument: "", success: false, error: "Insufficient consultations" },
        synthesis: {
          recommendation: `Debate limit reached (need ${DEBATE_COST} consultations, ${remaining} remaining). Proceed with your best judgment.`,
          confidence: "low",
          trade_offs: [],
          reasoning: "Insufficient consultation slots for debate",
        },
        debate_id: debateId,
        consultations_used: 0,
        consultations_remaining: remaining,
      };
    }

    // Reserve consultation slots
    if (!decrementConsultations(DEBATE_COST)) {
      return {
        success: false,
        topic,
        advocate: { perspective: "advocate", argument: "", success: false, error: "Failed to reserve slots" },
        critic: { perspective: "critic", argument: "", success: false, error: "Failed to reserve slots" },
        synthesis: {
          recommendation: "Failed to reserve consultation slots. Proceed with your best judgment.",
          confidence: "low",
          trade_offs: [],
          reasoning: "Could not decrement consultation counter",
        },
        debate_id: debateId,
        consultations_used: 0,
        consultations_remaining: getRemainingConsultations(),
      };
    }
  }

  const advocatePerspective = customPerspectives?.advocate ||
    "Advocate - argue in favor of this approach, focusing on benefits and opportunities";
  const criticPerspective = customPerspectives?.critic ||
    "Critic - argue against this approach, focusing on risks and downsides";

  const timeout = 45_000; // 45 seconds per perspective
  const contextStr = context || "";

  logger.info(`Starting debate: ${topic}`, { debate_id: debateId });

  // Spawn advocate and critic in parallel (but we'll do them sequentially for error handling)
  // In a real async implementation, these could run in parallel
  const advocate = spawnPerspective(workspace, advocatePerspective, topic, contextStr, timeout);
  const critic = spawnPerspective(workspace, criticPerspective, topic, contextStr, timeout);

  // If both perspectives failed, return early
  if (!advocate.success && !critic.success) {
    logger.error("Both perspectives failed", { debate_id: debateId });

    const remaining = getRemainingConsultations ? getRemainingConsultations() : 0;

    logDebate(workspace, debateId, {
      topic,
      advocate,
      critic,
      synthesis: null,
      timestamp: new Date().toISOString(),
      success: false,
    });

    return {
      success: false,
      topic,
      advocate,
      critic,
      synthesis: {
        recommendation: "Both debate perspectives failed. Proceed with your best judgment based on available information.",
        confidence: "low",
        trade_offs: ["Unable to generate debate perspectives"],
        reasoning: "Debate execution failed",
      },
      debate_id: debateId,
      consultations_used: DEBATE_COST,
      consultations_remaining: remaining,
    };
  }

  // Arbiter synthesis - even if one perspective failed, try to synthesize
  const synthesis = synthesizeArguments(workspace, topic, advocate, critic, timeout);

  const remaining = getRemainingConsultations ? getRemainingConsultations() : 0;

  logDebate(workspace, debateId, {
    topic,
    context: contextStr,
    advocate,
    critic,
    synthesis,
    timestamp: new Date().toISOString(),
    success: true,
  });

  logger.info(`Debate completed: ${topic}`, {
    debate_id: debateId,
    confidence: synthesis.confidence,
  });

  return {
    success: true,
    topic,
    advocate,
    critic,
    synthesis,
    debate_id: debateId,
    consultations_used: DEBATE_COST,
    consultations_remaining: remaining,
  };
}
