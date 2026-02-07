/**
 * Shared knowledge base for inter-agent communication.
 *
 * File-based key-value store at .agent-communication/knowledge/.
 * Each topic gets its own JSON file. Entries include timestamps
 * for staleness detection. Uses atomic writes (temp + rename)
 * to prevent data loss under concurrent agent access.
 */

import {
  readFileSync,
  existsSync,
  writeFileSync,
  mkdirSync,
  renameSync,
  unlinkSync,
} from "fs";
import { join } from "path";

const DEFAULT_MAX_AGE_HOURS = 24;
const MAX_TOPIC_LENGTH = 100;
const MAX_KEY_LENGTH = 200;
const MAX_VALUE_LENGTH = 10_000;

// Keys that could cause prototype pollution in downstream consumers
const FORBIDDEN_KEYS = new Set(["__proto__", "constructor", "prototype"]);

interface KnowledgeEntry {
  value: string;
  updated_at: string;
}

interface KnowledgeStore {
  [key: string]: KnowledgeEntry;
}

export interface ShareKnowledgeResult {
  success: boolean;
  topic: string;
  key: string;
  message: string;
}

export interface GetKnowledgeResult {
  success: boolean;
  topic: string;
  entries: Record<string, KnowledgeEntry & { stale?: boolean }>;
  message: string;
}

function ensureDirectory(dirPath: string): void {
  if (!existsSync(dirPath)) {
    mkdirSync(dirPath, { recursive: true });
  }
}

function sanitizeName(name: string, maxLength: number): string {
  return name.replace(/[^a-zA-Z0-9_-]/g, "_").substring(0, maxLength);
}

export function shareKnowledge(
  workspace: string,
  topic: string,
  key: string,
  value: string,
): ShareKnowledgeResult {
  const safeTopic = sanitizeName(topic, MAX_TOPIC_LENGTH);

  // Validate key
  if (FORBIDDEN_KEYS.has(key)) {
    return {
      success: false,
      topic: safeTopic,
      key,
      message: `Forbidden key: '${key}'`,
    };
  }
  const safeKey = key.substring(0, MAX_KEY_LENGTH);
  const safeValue = value.substring(0, MAX_VALUE_LENGTH);

  const knowledgePath = join(workspace, ".agent-communication", "knowledge");
  ensureDirectory(knowledgePath);

  const topicFile = join(knowledgePath, `${safeTopic}.json`);
  const tmpFile = join(knowledgePath, `${safeTopic}.tmp.${process.pid}`);
  let existing: KnowledgeStore = {};

  if (existsSync(topicFile)) {
    try {
      existing = JSON.parse(readFileSync(topicFile, "utf-8"));
    } catch {
      existing = {};
    }
  }

  existing[safeKey] = {
    value: safeValue,
    updated_at: new Date().toISOString(),
  };

  // Atomic write: write to temp file then rename
  try {
    writeFileSync(tmpFile, JSON.stringify(existing, null, 2));
    renameSync(tmpFile, topicFile);
  } catch (err) {
    // Clean up temp file on failure
    try {
      if (existsSync(tmpFile)) {
        unlinkSync(tmpFile);
      }
    } catch {
      // best-effort cleanup
    }
    throw err;
  }

  return {
    success: true,
    topic: safeTopic,
    key: safeKey,
    message: `Knowledge stored: ${safeTopic}/${safeKey}`,
  };
}

export function getKnowledge(
  workspace: string,
  topic: string,
  key?: string,
  maxAgeHours?: number,
): GetKnowledgeResult {
  const safeTopic = sanitizeName(topic, MAX_TOPIC_LENGTH);
  const topicFile = join(
    workspace,
    ".agent-communication",
    "knowledge",
    `${safeTopic}.json`,
  );

  if (!existsSync(topicFile)) {
    return {
      success: true,
      topic: safeTopic,
      entries: {},
      message: `No knowledge found for topic '${safeTopic}'`,
    };
  }

  let data: KnowledgeStore;
  try {
    data = JSON.parse(readFileSync(topicFile, "utf-8"));
  } catch {
    return {
      success: false,
      topic: safeTopic,
      entries: {},
      message: `Failed to parse knowledge file for topic '${safeTopic}'`,
    };
  }

  const ageLimit = maxAgeHours ?? DEFAULT_MAX_AGE_HOURS;
  const now = Date.now();

  const entries: Record<string, KnowledgeEntry & { stale?: boolean }> = {};

  for (const [k, entry] of Object.entries(data)) {
    if (key && k !== key) continue;

    const updatedTime = new Date(entry.updated_at).getTime();
    const stale = isNaN(updatedTime) || (now - updatedTime) / 3_600_000 > ageLimit;

    entries[k] = {
      ...entry,
      ...(stale ? { stale: true } : {}),
    };
  }

  const entryCount = Object.keys(entries).length;
  const staleCount = Object.values(entries).filter((e) => e.stale).length;

  let message = `Found ${entryCount} entries for topic '${safeTopic}'`;
  if (staleCount > 0) {
    message += ` (${staleCount} stale, older than ${ageLimit}h)`;
  }

  return {
    success: true,
    topic: safeTopic,
    entries,
    message,
  };
}
