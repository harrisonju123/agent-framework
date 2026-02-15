/**
 * Multi-perspective debate module.
 *
 * Spawns parallel Advocate and Critic perspectives, then an Arbiter
 * synthesizes a final recommendation with confidence level.
 */

import { spawn } from "child_process";
import { existsSync, writeFileSync, mkdirSync } from "fs";
import { join } from "path";
import { createLogger } from "./logger.js";
import {
  buildConsultationEnv,
  getConsultationCount,
  incrementConsultationCount,
} from "./consultation.js";
import type { DebateInput, DebateResult } from "./types.js";

const logger = createLogger();

const MAX_CONSULTATIONS = Math.min(
  parseInt(process.env.MAX_CONSULTATIONS_PER_SESSION || "5", 10),
  20,
);

const MAX_TOPIC_LENGTH = 2000;
const MAX_CONTEXT_LENGTH = 1000;
const DEBATE_COST = 2; // advocate + critic count as 2 consultation slots

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
 * Parse arbiter output into structured components.
 * Expected format: RECOMMENDATION: ... | CONFIDENCE: ... | TRADE-OFFS: ... | REASONING: ...
 */
function parseArbiterOutput(output: string): {
  recommendation: string;
  confidence: "high" | "medium" | "low";
  trade_offs: string[];
  reasoning: string;
} {
  const lines = output.trim();

  // Extract sections using regex
  const recMatch = lines.match(/RECOMMENDATION:\s*([^|]+)/i);
  const confMatch = lines.match(/CONFIDENCE:\s*(\w+)/i);
  const tradeMatch = lines.match(/TRADE-OFFS:\s*([^|]+)/i);
  const reasonMatch = lines.match(/REASONING:\s*(.+)/is);

  const recommendation = recMatch?.[1]?.trim() || "Unable to reach a decision";
  const confidenceRaw = confMatch?.[1]?.trim().toLowerCase() || "medium";
  const confidence =
    confidenceRaw === "high" || confidenceRaw === "low"
      ? confidenceRaw
      : "medium";
  const tradeText = tradeMatch?.[1]?.trim() || "";
  const trade_offs = tradeText
    ? tradeText
        .split(/[-•]\s*/)
        .filter((s) => s.trim())
        .map((s) => s.trim())
    : ["No trade-offs identified"];
  const reasoning = reasonMatch?.[1]?.trim() || output;

  return { recommendation, confidence, trade_offs, reasoning };
}

/**
 * Run a Claude CLI subprocess for a debate perspective.
 * Uses spawn to support stdin piping (execFile doesn't support input in async mode).
 */
async function runPerspective(
  workspace: string,
  prompt: string,
  role: "advocate" | "critic" | "arbiter",
): Promise<string> {
  return new Promise<string>((resolve) => {
    const proc = spawn(
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

    proc.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    proc.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("error", (err) => {
      logger.warn(`${role} perspective spawn failed`, { error: err.message });
      resolve(`[${role} failed: ${err.message}]`);
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        logger.warn(`${role} perspective exited with code ${code}`, { stderr });
        resolve(`[${role} failed: exit code ${code}]`);
      } else {
        resolve(stdout.trim());
      }
    });

    // Timeout after 60 seconds
    const timeout = setTimeout(() => {
      proc.kill();
      resolve(`[${role} failed: timeout]`);
    }, 60_000);

    proc.on("close", () => {
      clearTimeout(timeout);
    });

    // Write prompt to stdin and close
    proc.stdin.write(prompt);
    proc.stdin.end();
  });
}

export async function debateTopic(
  workspace: string,
  input: DebateInput,
): Promise<DebateResult> {
  const debateId = `debate-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

  // Rate limit check — debate costs 2 slots
  const currentCount = getConsultationCount();
  if (currentCount + DEBATE_COST > MAX_CONSULTATIONS) {
    return {
      success: false,
      debate_id: debateId,
      advocate_argument: "",
      critic_argument: "",
      synthesis: `Debate limit reached (${MAX_CONSULTATIONS} consultations per session, debate costs ${DEBATE_COST}). Proceed with your best judgment.`,
      confidence: "low",
      recommendation: "Cannot proceed with debate",
      trade_offs: [],
      consultations_remaining: Math.max(0, MAX_CONSULTATIONS - currentCount),
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

  const advocatePos =
    advocate_position || "in favor of the proposed approach";
  const criticPos =
    critic_position ||
    "against the proposed approach, proposing alternatives";

  // Build prompts
  const advocatePrompt = [
    "You are the Advocate in a structured debate. Argue IN FAVOR of the following approach.",
    "Present the strongest case: benefits, evidence, precedent. Be specific and practical.",
    `\nTopic: ${topic}`,
    context ? `\nContext: ${context}` : "",
    `\nPosition: ${advocatePos}`,
    "\nProvide a concise argument (3-5 sentences).",
  ].join("\n");

  const criticPrompt = [
    "You are the Critic in a structured debate. Argue AGAINST the following approach.",
    "Identify risks, weaknesses, alternatives, and hidden costs. Be specific and constructive — don't just naysay, propose better alternatives.",
    `\nTopic: ${topic}`,
    context ? `\nContext: ${context}` : "",
    `\nPosition: ${criticPos}`,
    "\nProvide a concise argument (3-5 sentences).",
  ].join("\n");

  // Run advocate and critic in parallel
  const [advocateArgument, criticArgument] = await Promise.all([
    runPerspective(workspace, advocatePrompt, "advocate"),
    runPerspective(workspace, criticPrompt, "critic"),
  ]);

  // Check if both failed
  if (
    advocateArgument.startsWith("[advocate failed") &&
    criticArgument.startsWith("[critic failed")
  ) {
    return {
      success: false,
      debate_id: debateId,
      advocate_argument: advocateArgument,
      critic_argument: criticArgument,
      synthesis: "Both debate perspectives failed. Unable to proceed.",
      confidence: "low",
      recommendation: "Unable to reach a decision",
      trade_offs: [],
      consultations_remaining: Math.max(
        0,
        MAX_CONSULTATIONS - currentCount - DEBATE_COST,
      ),
    };
  }

  // Run arbiter synthesis (sequential, after advocate+critic)
  const arbiterPrompt = [
    "You are the Arbiter synthesizing a debate. Given the Advocate's and Critic's arguments, produce a final recommendation.",
    "Be decisive — pick a clear direction.",
    "\nFormat your response EXACTLY as:",
    "RECOMMENDATION: [one sentence verdict]",
    "CONFIDENCE: [high/medium/low]",
    "TRADE-OFFS: [bullet list of key trade-offs, one per line with - prefix]",
    "REASONING: [2-3 sentences explaining your synthesis]",
    `\n--- ADVOCATE ARGUMENT ---\n${advocateArgument}`,
    `\n--- CRITIC ARGUMENT ---\n${criticArgument}`,
  ].join("\n");

  const arbiterOutput = await runPerspective(workspace, arbiterPrompt, "arbiter");

  if (arbiterOutput.startsWith("[arbiter failed")) {
    logger.warn("Arbiter failed, returning raw arguments");
    // Graceful degradation: return arguments without synthesis
    const result: DebateResult = {
      success: true,
      debate_id: debateId,
      advocate_argument: advocateArgument,
      critic_argument: criticArgument,
      synthesis: arbiterOutput,
      confidence: "low",
      recommendation: "See advocate and critic arguments for manual synthesis",
      trade_offs: ["Arbiter synthesis unavailable"],
      consultations_remaining: Math.max(
        0,
        MAX_CONSULTATIONS - currentCount - DEBATE_COST,
      ),
    };
    logDebate(workspace, debateId, {
      ...result,
      timestamp: new Date().toISOString(),
    });
    incrementConsultationCount(DEBATE_COST);
    return result;
  }

  const parsed = parseArbiterOutput(arbiterOutput);

  incrementConsultationCount(DEBATE_COST);

  const result: DebateResult = {
    success: true,
    debate_id: debateId,
    advocate_argument: advocateArgument,
    critic_argument: criticArgument,
    synthesis: parsed.reasoning,
    confidence: parsed.confidence,
    recommendation: parsed.recommendation,
    trade_offs: parsed.trade_offs,
    consultations_remaining: Math.max(
      0,
      MAX_CONSULTATIONS - currentCount - DEBATE_COST,
    ),
  };

  logDebate(workspace, debateId, {
    ...result,
    topic,
    context,
    timestamp: new Date().toISOString(),
  });

  logger.info("Debate completed", { debate_id: debateId });

  return result;
}
