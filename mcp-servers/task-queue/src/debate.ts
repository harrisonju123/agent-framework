/**
 * Multi-perspective debate via Claude CLI.
 *
 * Enables structured debates with Advocate, Critic, and Arbiter perspectives
 * for complex architectural decisions. Uses parallel subprocess spawning for
 * advocate+critic, then sequential arbiter synthesis.
 */

import { spawn } from "child_process";
import { execFileSync } from "child_process";
import { existsSync, mkdirSync, writeFileSync } from "fs";
import { join } from "path";
import { createLogger } from "./logger.js";
import {
  buildConsultationEnv,
  getConsultationCount,
  incrementConsultationCount,
  getMaxConsultations,
} from "./consultation.js";
import type { DebateInput, DebateResult } from "./types.js";

const logger = createLogger();

// Input length limits
const MAX_TOPIC_LENGTH = 2000;
const MAX_CONTEXT_LENGTH = 1000;

function sanitizeLogId(id: string): string {
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
  const safeId = sanitizeLogId(id);
  writeFileSync(join(logDir, `${safeId}.json`), JSON.stringify(data, null, 2));
}

/**
 * Parse arbiter output into structured fields.
 * Expected format: RECOMMENDATION: ... | CONFIDENCE: ... | TRADE-OFFS: ... | REASONING: ...
 */
function parseArbiterOutput(output: string): {
  recommendation: string;
  confidence: "high" | "medium" | "low";
  trade_offs: string[];
  reasoning: string;
} {
  const defaultResult = {
    recommendation: "Unable to parse recommendation",
    confidence: "low" as const,
    trade_offs: [],
    reasoning: output,
  };

  try {
    // Extract RECOMMENDATION
    const recMatch = output.match(/RECOMMENDATION:\s*([^|]+)/i);
    const recommendation = recMatch
      ? recMatch[1].trim()
      : "See reasoning below";

    // Extract CONFIDENCE
    const confMatch = output.match(/CONFIDENCE:\s*(high|medium|low)/i);
    const confidence = (confMatch
      ? confMatch[1].toLowerCase()
      : "medium") as "high" | "medium" | "low";

    // Extract TRADE-OFFS (bullet list)
    const tradeOffsMatch = output.match(
      /TRADE-OFFS:\s*([^|]+?)(?=\s*\||$)/is,
    );
    let trade_offs: string[] = [];
    if (tradeOffsMatch) {
      const tradeOffsText = tradeOffsMatch[1].trim();
      trade_offs = tradeOffsText
        .split("\n")
        .map((line) => line.trim().replace(/^[-*•]\s*/, ""))
        .filter((line) => line.length > 0);
    }

    // Extract REASONING
    const reasoningMatch = output.match(/REASONING:\s*(.+?)$/is);
    const reasoning = reasoningMatch ? reasoningMatch[1].trim() : output;

    return { recommendation, confidence, trade_offs, reasoning };
  } catch (error) {
    logger.warn("Failed to parse arbiter output, using fallback", { error });
    return defaultResult;
  }
}

/**
 * Run a Claude CLI subprocess for a debate perspective.
 * Returns the output or an error message.
 */
async function runPerspective(
  workspace: string,
  prompt: string,
  timeoutMs: number,
): Promise<{ success: boolean; output: string }> {
  return new Promise((resolve) => {
    const child = spawn(
      "claude",
      ["--print", "--model", "haiku", "--max-turns", "1"],
      {
        cwd: workspace,
        env: buildConsultationEnv(),
        stdio: ["pipe", "pipe", "pipe"],
      },
    );

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    child.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    const timer = setTimeout(() => {
      child.kill();
      resolve({
        success: false,
        output: "Timeout exceeded",
      });
    }, timeoutMs);

    child.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) {
        resolve({ success: true, output: stdout.trim() });
      } else {
        logger.error("Perspective subprocess failed", {
          code,
          stderr: stderr.substring(0, 200),
        });
        resolve({
          success: false,
          output: stdout.trim() || `Failed with code ${code}`,
        });
      }
    });

    child.on("error", (error) => {
      clearTimeout(timer);
      logger.error("Perspective subprocess error", { error: error.message });
      resolve({
        success: false,
        output: `Failed: ${error.message}`,
      });
    });

    // Write prompt to stdin
    child.stdin.write(prompt);
    child.stdin.end();
  });
}

export async function debateTopic(
  workspace: string,
  input: DebateInput,
): Promise<DebateResult> {
  const debateId = `debate-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

  // Rate limit check — debate costs 2 consultation slots
  const currentCount = getConsultationCount();
  const maxConsultations = getMaxConsultations();
  if (currentCount + 2 > maxConsultations) {
    return {
      success: false,
      debate_id: debateId,
      advocate_argument: "",
      critic_argument: "",
      synthesis:
        `Debate limit reached (need 2 slots, have ${maxConsultations - currentCount} remaining). Proceed with your best judgment.`,
      confidence: "low",
      recommendation: "Insufficient consultation slots for debate",
      trade_offs: [],
      consultations_remaining: maxConsultations - currentCount,
    };
  }

  // Input validation
  let { topic, context, advocate_position, critic_position } = input;
  if (topic.length > MAX_TOPIC_LENGTH) {
    topic = topic.substring(0, MAX_TOPIC_LENGTH) + "... [truncated]";
  }
  if (context && context.length > MAX_CONTEXT_LENGTH) {
    context = context.substring(0, MAX_CONTEXT_LENGTH) + "... [truncated]";
  }

  // Default positions
  advocate_position =
    advocate_position || "in favor of the proposed approach";
  critic_position =
    critic_position ||
    "against the proposed approach, proposing alternatives";

  // Build prompts
  const advocatePrompt = [
    `You are the Advocate in a structured debate. Argue IN FAVOR of the following approach.`,
    `Present the strongest case: benefits, evidence, precedent. Be specific and practical.`,
    ``,
    `Topic: ${topic}`,
    context ? `Context: ${context}` : "",
    `Position: ${advocate_position}`,
    ``,
    `Provide a concise, persuasive argument (3-5 sentences).`,
  ]
    .filter((line) => line !== undefined)
    .join("\n");

  const criticPrompt = [
    `You are the Critic in a structured debate. Argue AGAINST the following approach.`,
    `Identify risks, weaknesses, alternatives, and hidden costs. Be specific and constructive — don't just naysay, propose better alternatives.`,
    ``,
    `Topic: ${topic}`,
    context ? `Context: ${context}` : "",
    `Position: ${critic_position}`,
    ``,
    `Provide a concise, constructive critique (3-5 sentences).`,
  ]
    .filter((line) => line !== undefined)
    .join("\n");

  logger.info(`Starting debate: ${debateId}`, { topic });

  // Run advocate and critic in parallel
  const [advocateResult, criticResult] = await Promise.all([
    runPerspective(workspace, advocatePrompt, 60_000),
    runPerspective(workspace, criticPrompt, 60_000),
  ]);

  const advocateArgument = advocateResult.success
    ? advocateResult.output
    : `[Advocate failed: ${advocateResult.output}]`;
  const criticArgument = criticResult.success
    ? criticResult.output
    : `[Critic failed: ${criticResult.output}]`;

  // If both failed, return early
  if (!advocateResult.success && !criticResult.success) {
    incrementConsultationCount(2);
    logDebate(workspace, debateId, {
      topic,
      context,
      advocate_argument: advocateArgument,
      critic_argument: criticArgument,
      success: false,
      timestamp: new Date().toISOString(),
    });

    return {
      success: false,
      debate_id: debateId,
      advocate_argument: advocateArgument,
      critic_argument: criticArgument,
      synthesis: "Both perspectives failed to execute.",
      confidence: "low",
      recommendation: "Unable to complete debate",
      trade_offs: [],
      consultations_remaining: maxConsultations - currentCount - 2,
    };
  }

  // Run arbiter to synthesize
  const arbiterPrompt = [
    `You are the Arbiter synthesizing a debate. Given the Advocate's and Critic's arguments, produce a final recommendation.`,
    `Be decisive — pick a clear direction.`,
    ``,
    `ADVOCATE ARGUMENT:`,
    advocateArgument,
    ``,
    `CRITIC ARGUMENT:`,
    criticArgument,
    ``,
    `Format your response as:`,
    `RECOMMENDATION: [one sentence verdict]`,
    `CONFIDENCE: [high/medium/low]`,
    `TRADE-OFFS:`,
    `- [key trade-off 1]`,
    `- [key trade-off 2]`,
    `REASONING: [2-3 sentences explaining your synthesis]`,
  ].join("\n");

  let arbiterOutput = "";
  let arbiterSuccess = false;
  try {
    const result = execFileSync(
      "claude",
      ["--print", "--model", "haiku", "--max-turns", "1"],
      {
        input: arbiterPrompt,
        timeout: 60_000,
        encoding: "utf-8",
        cwd: workspace,
        env: buildConsultationEnv(),
        maxBuffer: 1024 * 1024,
      },
    );
    arbiterOutput = result.trim();
    arbiterSuccess = true;
  } catch (error: unknown) {
    const err = error as Error;
    logger.error("Arbiter subprocess failed", { error: err.message });
    arbiterOutput = `Arbiter failed: ${err.message}`;
  }

  // Parse arbiter output
  const parsed = arbiterSuccess
    ? parseArbiterOutput(arbiterOutput)
    : {
        recommendation: "See advocate and critic arguments",
        confidence: "low" as const,
        trade_offs: [],
        reasoning: arbiterOutput,
      };

  incrementConsultationCount(2);

  const finalResult: DebateResult = {
    success: arbiterSuccess && (advocateResult.success || criticResult.success),
    debate_id: debateId,
    advocate_argument: advocateArgument,
    critic_argument: criticArgument,
    synthesis: parsed.reasoning,
    confidence: parsed.confidence,
    recommendation: parsed.recommendation,
    trade_offs: parsed.trade_offs,
    consultations_remaining: maxConsultations - currentCount - 2,
  };

  logDebate(workspace, debateId, {
    topic,
    context,
    advocate_position,
    critic_position,
    advocate_argument: advocateArgument,
    critic_argument: criticArgument,
    arbiter_output: arbiterOutput,
    recommendation: parsed.recommendation,
    confidence: parsed.confidence,
    trade_offs: parsed.trade_offs,
    synthesis: parsed.reasoning,
    timestamp: new Date().toISOString(),
  });

  logger.info(`Debate completed: ${debateId}`, {
    success: finalResult.success,
    confidence: parsed.confidence,
  });

  return finalResult;
}
