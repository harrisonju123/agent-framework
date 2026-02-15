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

import { spawn } from "child_process";
import { writeFileSync, mkdirSync, existsSync } from "fs";
import { join } from "path";
import { createLogger } from "./logger.js";
import { buildConsultationEnv } from "./consultation.js";
import type { PerspectiveArgument, DebateResult } from "./types.js";

const logger = createLogger();

// Debate counts against the consultation rate limit (2 slots per debate)
const DEBATE_COST = 2;

// Input validation limits
const MAX_TOPIC_LENGTH = 1500;
const MAX_DEBATE_CONTEXT_LENGTH = 3000;

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
 * Spawn a single perspective agent with a specific role (async).
 */
function spawnPerspective(
  workspace: string,
  perspective: string,
  topic: string,
  context: string,
  timeout: number,
): Promise<PerspectiveArgument> {
  return new Promise((resolve) => {
    const prompt = [
      `You are participating in a structured debate to help make a complex decision.`,
      `Your role: ${perspective}`,
      `Topic: ${topic}`,
      context ? `\nContext: ${context}` : "",
      `\nProvide your argument in 3-5 sentences. Be specific and focus on practical implications.`,
      `Your goal is to argue your assigned perspective, even if you see merit in the other side.`,
    ].join("\n");

    let output = "";
    let errorOutput = "";

    const proc = spawn(
      "claude",
      [
        "--print",
        "--model",
        "haiku",
        "--max-turns",
        "1",
      ],
      {
        cwd: workspace,
        env: buildConsultationEnv(),
        timeout,
      },
    );

    proc.stdout.on("data", (data: Buffer) => {
      output += data.toString();
    });

    proc.stderr.on("data", (data: Buffer) => {
      errorOutput += data.toString();
    });

    proc.on("close", (code: number) => {
      if (code === 0) {
        resolve({
          perspective,
          argument: output.trim(),
          success: true,
        });
      } else {
        logger.error(`Perspective ${perspective} failed`, {
          error: errorOutput || `Process exited with code ${code}`
        });
        resolve({
          perspective,
          argument: "",
          success: false,
          error: errorOutput || `Process exited with code ${code}`,
        });
      }
    });

    proc.on("error", (err: Error) => {
      logger.error(`Perspective ${perspective} failed`, { error: err.message });
      resolve({
        perspective,
        argument: "",
        success: false,
        error: err.message,
      });
    });

    if (proc.stdin) {
      proc.stdin.write(prompt);
      proc.stdin.end();
    }
  });
}

/**
 * Arbiter synthesizes both perspectives into a final recommendation (async).
 */
function synthesizeArguments(
  workspace: string,
  topic: string,
  advocate: PerspectiveArgument,
  critic: PerspectiveArgument,
  timeout: number,
): Promise<{
  recommendation: string;
  confidence: "high" | "medium" | "low";
  trade_offs: string[];
  reasoning: string;
}> {
  return new Promise((resolve) => {
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

    let output = "";
    let errorOutput = "";

    const proc = spawn(
      "claude",
      [
        "--print",
        "--model",
        "haiku",
        "--max-turns",
        "1",
      ],
      {
        cwd: workspace,
        env: buildConsultationEnv(),
        timeout,
      },
    );

    proc.stdout.on("data", (data: Buffer) => {
      output += data.toString();
    });

    proc.stderr.on("data", (data: Buffer) => {
      errorOutput += data.toString();
    });

    proc.on("close", (code: number) => {
      if (code !== 0) {
        logger.error("Arbiter synthesis failed", {
          error: errorOutput || `Process exited with code ${code}`
        });
        resolve({
          recommendation: "Unable to synthesize perspectives due to execution error. Review both arguments and decide based on your judgment.",
          confidence: "low",
          trade_offs: ["Synthesis failed - manual review needed"],
          reasoning: `Arbiter failed: ${errorOutput || `exit code ${code}`}`,
        });
        return;
      }

      try {
        // Parse JSON from the response
        const text = output.trim();
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

        resolve({
          recommendation: synthesis.recommendation,
          confidence: synthesis.confidence as "high" | "medium" | "low",
          trade_offs: synthesis.trade_offs,
          reasoning: synthesis.reasoning || "No reasoning provided",
        });
      } catch (error: unknown) {
        const err = error as Error;
        logger.error("Arbiter synthesis failed", { error: err.message });

        // Fallback synthesis
        resolve({
          recommendation: "Unable to synthesize perspectives due to parsing error. Review both arguments and decide based on your judgment.",
          confidence: "low",
          trade_offs: ["Synthesis failed - manual review needed"],
          reasoning: `Arbiter failed: ${err.message}`,
        });
      }
    });

    proc.on("error", (err: Error) => {
      logger.error("Arbiter synthesis failed", { error: err.message });
      resolve({
        recommendation: "Unable to synthesize perspectives due to execution error. Review both arguments and decide based on your judgment.",
        confidence: "low",
        trade_offs: ["Synthesis failed - manual review needed"],
        reasoning: `Arbiter failed: ${err.message}`,
      });
    });

    if (proc.stdin) {
      proc.stdin.write(prompt);
      proc.stdin.end();
    }
  });
}

/**
 * Main debate function - coordinates advocate, critic, and arbiter.
 */
export async function debateTopic(
  workspace: string,
  topic: string,
  context?: string,
  customPerspectives?: { advocate: string; critic: string },
  getRemainingConsultations?: () => number,
  decrementConsultations?: (count: number) => boolean,
): Promise<DebateResult> {
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

  // Spawn advocate and critic in parallel using Promise.allSettled
  const [advocateResult, criticResult] = await Promise.allSettled([
    spawnPerspective(workspace, advocatePerspective, topic, contextStr, timeout),
    spawnPerspective(workspace, criticPerspective, topic, contextStr, timeout),
  ]);

  const advocate = advocateResult.status === "fulfilled" ? advocateResult.value : {
    perspective: advocatePerspective,
    argument: "",
    success: false,
    error: "Promise rejected",
  };

  const critic = criticResult.status === "fulfilled" ? criticResult.value : {
    perspective: criticPerspective,
    argument: "",
    success: false,
    error: "Promise rejected",
  };

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
  const synthesis = await synthesizeArguments(workspace, topic, advocate, critic, timeout);

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
