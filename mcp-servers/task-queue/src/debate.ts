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
import { agentRemember } from "./agent-memory.js";
import type { PerspectiveArgument, DebateResult } from "./types.js";

const logger = createLogger();

// Debate counts against the consultation rate limit (2 slots per debate)
const DEBATE_COST = 2;

// Input validation limits
const MAX_TOPIC_LENGTH = 1500;
const MAX_DEBATE_CONTEXT_LENGTH = 3000;

// Domain keyword map for specialization hint detection.
// Each key is a specialization domain; values are keywords that signal
// the debate topic belongs to that domain.
const DOMAIN_KEYWORDS: Record<string, string[]> = {
  database: ["database", "sql", "postgres", "mysql", "mongodb", "redis", "migration", "schema", "orm", "query", "index"],
  frontend: ["frontend", "react", "vue", "angular", "css", "ui", "ux", "component", "dom", "browser", "responsive"],
  backend: ["backend", "api", "rest", "graphql", "server", "endpoint", "middleware", "authentication", "authorization"],
  infrastructure: ["infrastructure", "docker", "kubernetes", "ci", "cd", "deploy", "aws", "gcp", "azure", "terraform", "helm"],
  security: ["security", "encryption", "vulnerability", "auth", "oauth", "jwt", "xss", "csrf", "injection", "credential"],
  testing: ["testing", "test", "coverage", "e2e", "integration", "unit test", "mock", "fixture", "assertion"],
  performance: ["performance", "optimization", "cache", "latency", "throughput", "scaling", "bottleneck", "profiling"],
};

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
 * Uses async spawn() for parallel execution support.
 */
async function spawnPerspective(
  workspace: string,
  perspective: string,
  topic: string,
  context: string,
  timeout: number,
): Promise<PerspectiveArgument> {
  const prompt = [
    `You are participating in a structured debate to help make a complex decision.`,
    `Your role: ${perspective}`,
    `Topic: ${topic}`,
    context ? `\nContext: ${context}` : "",
    `\nProvide your argument in 3-5 sentences. Be specific and focus on practical implications.`,
    `Your goal is to argue your assigned perspective, even if you see merit in the other side.`,
  ].join("\n");

  return new Promise<PerspectiveArgument>((resolve) => {
    let stdout = "";
    let stderr = "";

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
        stdio: ["pipe", "pipe", "pipe"],
      },
    );

    proc.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    proc.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("error", (err) => {
      logger.error(`Perspective ${perspective} spawn failed`, { error: err.message });
      resolve({
        perspective,
        argument: "",
        success: false,
        error: err.message,
      });
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        logger.error(`Perspective ${perspective} exited with code ${code}`, { stderr });
        resolve({
          perspective,
          argument: "",
          success: false,
          error: `Process exited with code ${code}`,
        });
      } else {
        resolve({
          perspective,
          argument: stdout.trim(),
          success: true,
        });
      }
    });

    // Timeout after specified duration
    const timeoutHandle = setTimeout(() => {
      proc.kill();
      resolve({
        perspective,
        argument: "",
        success: false,
        error: "Process timeout",
      });
    }, timeout);

    proc.on("close", () => {
      clearTimeout(timeoutHandle);
    });

    // Write prompt to stdin and close
    proc.stdin.write(prompt);
    proc.stdin.end();
  });
}

/**
 * Arbiter synthesizes both perspectives into a final recommendation.
 * Uses async spawn() for consistency with parallel perspective generation.
 */
async function synthesizeArguments(
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

  return new Promise<{
    recommendation: string;
    confidence: "high" | "medium" | "low";
    trade_offs: string[];
    reasoning: string;
  }>((resolve) => {
    let stdout = "";
    let stderr = "";

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
        stdio: ["pipe", "pipe", "pipe"],
      },
    );

    proc.stdout.on("data", (data) => {
      stdout += data.toString();
    });

    proc.stderr.on("data", (data) => {
      stderr += data.toString();
    });

    proc.on("error", (err) => {
      logger.error("Arbiter spawn failed", { error: err.message });
      resolve({
        recommendation: "Unable to synthesize perspectives due to error. Review both arguments and decide based on your judgment.",
        confidence: "low",
        trade_offs: ["Synthesis failed - error during execution"],
        reasoning: `Arbiter failed: ${err.message}`,
      });
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        logger.error(`Arbiter exited with code ${code}`, { stderr });
        resolve({
          recommendation: "Unable to synthesize perspectives due to error. Review both arguments and decide based on your judgment.",
          confidence: "low",
          trade_offs: ["Synthesis failed - process exited with error"],
          reasoning: `Arbiter failed: exit code ${code}`,
        });
        return;
      }

      try {
        // Parse JSON from the response
        const text = stdout.trim();
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
        logger.error("Arbiter parsing failed", { error: err.message });

        resolve({
          recommendation: "Unable to synthesize perspectives due to parsing error. Review both arguments and decide based on your judgment.",
          confidence: "low",
          trade_offs: ["Synthesis failed - manual review needed"],
          reasoning: `Arbiter failed: ${err.message}`,
        });
      }
    });

    // Timeout after specified duration
    const timeoutHandle = setTimeout(() => {
      proc.kill();
      resolve({
        recommendation: "Unable to synthesize perspectives due to timeout. Review both arguments and decide based on your judgment.",
        confidence: "low",
        trade_offs: ["Synthesis failed - timeout"],
        reasoning: "Arbiter failed: process timeout",
      });
    }, timeout);

    proc.on("close", () => {
      clearTimeout(timeoutHandle);
    });

    // Write prompt to stdin and close
    proc.stdin.write(prompt);
    proc.stdin.end();
  });
}

/**
 * Store debate synthesis as a persistent memory with category "architectural_decisions".
 * Memory failures are logged but don't break the debate flow.
 */
function storeDebateMemory(
  workspace: string,
  debateId: string,
  topic: string,
  synthesis: {
    recommendation: string;
    confidence: "high" | "medium" | "low";
    trade_offs: string[];
    reasoning: string;
  },
): void {
  try {
    const repoSlug = process.env.GITHUB_REPOSITORY || "unknown";
    // AGENT_ID is set by run_agent.py; AGENT_TYPE is an alternative fallback
    const originAgent = process.env.AGENT_ID || process.env.AGENT_TYPE || "unknown";

    // Extract topic keywords for tagging (first 3-4 meaningful words)
    // Include short technical terms like API, CI, DB, etc.
    const topicKeywords = topic
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter(w => w.length > 0)
      .slice(0, 4);

    // Format memory content with structured information
    const content = [
      `Topic: ${topic}`,
      `Recommendation: ${synthesis.recommendation}`,
      `Confidence: ${synthesis.confidence}`,
      `Trade-offs: ${synthesis.trade_offs.join("; ")}`,
      `Reasoning: ${synthesis.reasoning}`,
    ].join("\n");

    // Store confidence as a structured tag so retrieval code doesn't have to
    // parse it out of the prose content (avoids fragile regex coupling).
    const tags = ["debate", debateId, `origin:${originAgent}`, `confidence:${synthesis.confidence}`, ...topicKeywords];

    // Store under agent_type "shared" so any agent (engineer, qa, architect, etc.)
    // can recall these architectural decisions. The Python MemoryRetriever is
    // responsible for merging "shared" into each agent's recall results.
    const result = agentRemember(workspace, {
      repo_slug: repoSlug,
      agent_type: "shared",
      category: "architectural_decisions",
      content,
      tags,
    });

    if (result.success) {
      logger.info(`Debate memory stored: ${topic}`, { debate_id: debateId });
    } else {
      logger.warn(`Failed to store debate memory: ${result.message}`, { debate_id: debateId });
    }
  } catch (error: unknown) {
    const err = error as Error;
    logger.warn(`Error storing debate memory: ${err.message}`, { debate_id: debateId });
  }
}

/**
 * Detect which specialization domains a debate topic touches.
 * Returns matched domain names sorted by keyword hit count (highest first).
 */
function detectDomains(topic: string): string[] {
  const lower = topic.toLowerCase();
  const hits: { domain: string; count: number }[] = [];

  for (const [domain, keywords] of Object.entries(DOMAIN_KEYWORDS)) {
    const count = keywords.filter(kw => lower.includes(kw)).length;
    if (count > 0) {
      hits.push({ domain, count });
    }
  }

  hits.sort((a, b) => b.count - a.count);
  return hits.map(h => h.domain);
}

/**
 * Store debate-derived specialization hints when a high-confidence debate
 * reveals domain-specific recommendations. These hints feed into the
 * cross-feature learning loop — the Python-side specialization system
 * can query them to adjust agent profiles for future tasks.
 *
 * Only fires for high-confidence debates with detectable domain keywords,
 * to avoid polluting memory with low-signal hints.
 */
function storeDebateInsight(
  workspace: string,
  debateId: string,
  topic: string,
  synthesis: {
    recommendation: string;
    confidence: "high" | "medium" | "low";
    trade_offs: string[];
    reasoning: string;
  },
): void {
  if (synthesis.confidence !== "high") return;

  const domains = detectDomains(topic);
  if (domains.length === 0) return;

  try {
    const repoSlug = process.env.GITHUB_REPOSITORY || "unknown";
    const originAgent = process.env.AGENT_ID || process.env.AGENT_TYPE || "unknown";
    const primaryDomain = domains[0];

    const content = [
      `Domain: ${primaryDomain}`,
      `Topic: ${topic}`,
      `Recommendation: ${synthesis.recommendation}`,
      `Reasoning: ${synthesis.reasoning}`,
    ].join("\n");

    const tags = [
      "debate_insight",
      debateId,
      `origin:${originAgent}`,
      `domain:${primaryDomain}`,
      ...domains.slice(1).map(d => `domain:${d}`),
    ];

    const result = agentRemember(workspace, {
      repo_slug: repoSlug,
      agent_type: "shared",
      category: "debate_specialization_hints",
      content,
      tags,
    });

    if (result.success) {
      logger.info(`Debate specialization hint stored: ${primaryDomain}`, { debate_id: debateId });
    } else {
      logger.warn(`Failed to store debate insight: ${result.message}`, { debate_id: debateId });
    }
  } catch (error: unknown) {
    const err = error as Error;
    logger.warn(`Error storing debate insight: ${err.message}`, { debate_id: debateId });
  }
}

/**
 * Main debate function - coordinates advocate, critic, and arbiter.
 * Runs advocate and critic in parallel, then arbiter synthesis sequentially.
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

  // Run advocate and critic in parallel using Promise.allSettled
  const [advocateResult, criticResult] = await Promise.allSettled([
    spawnPerspective(workspace, advocatePerspective, topic, contextStr, timeout),
    spawnPerspective(workspace, criticPerspective, topic, contextStr, timeout),
  ]);

  // Extract results from allSettled promises
  const advocate = advocateResult.status === "fulfilled"
    ? advocateResult.value
    : { perspective: "advocate", argument: "", success: false, error: "Promise rejected" };

  const critic = criticResult.status === "fulfilled"
    ? criticResult.value
    : { perspective: "critic", argument: "", success: false, error: "Promise rejected" };

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

  // Store debate synthesis as a persistent memory
  storeDebateMemory(workspace, debateId, topic, synthesis);

  // Feed high-confidence domain-specific debates into specialization hints
  storeDebateInsight(workspace, debateId, topic, synthesis);

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
