/**
 * Agent memory MCP tools — remember/recall for persistent cross-task learning.
 *
 * Wraps the file-based memory store at .agent-communication/memory/.
 * Each (repo, agent_type) pair gets its own JSON file containing an
 * array of memory entries with category, content, and access metadata.
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

const MAX_MEMORIES_PER_STORE = 200;
const MAX_CONTENT_LENGTH = 2000;
const MAX_CATEGORY_LENGTH = 50;

interface MemoryEntry {
  category: string;
  content: string;
  created_at: number;
  last_accessed: number;
  access_count: number;
  source_task_id: string | null;
  tags: string[];
}

export interface RememberInput {
  repo_slug: string;
  agent_type: string;
  category: string;
  content: string;
  tags?: string[];
}

export interface RecallInput {
  repo_slug: string;
  agent_type: string;
  category?: string;
  tags?: string[];
  limit?: number;
}

export interface MemoryResult {
  success: boolean;
  message: string;
  memories?: MemoryEntry[];
}

function safeSlug(name: string): string {
  return name.replace(/\//g, "__").replace(/[^a-zA-Z0-9_-]/g, "_");
}

function storePath(workspace: string, repoSlug: string, agentType: string): string {
  const safeRepo = safeSlug(repoSlug);
  return join(workspace, ".agent-communication", "memory", safeRepo, `${agentType}.json`);
}

function loadEntries(path: string): MemoryEntry[] {
  if (!existsSync(path)) return [];
  try {
    const data = JSON.parse(readFileSync(path, "utf-8"));
    return Array.isArray(data) ? data : [];
  } catch {
    return [];
  }
}

function saveEntries(path: string, entries: MemoryEntry[]): void {
  const dir = path.substring(0, path.lastIndexOf("/"));
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
  const tmpPath = `${path}.tmp.${process.pid}`;
  try {
    writeFileSync(tmpPath, JSON.stringify(entries, null, 2));
    renameSync(tmpPath, path);
  } catch (err) {
    try {
      if (existsSync(tmpPath)) unlinkSync(tmpPath);
    } catch { /* best-effort */ }
    throw err;
  }
}

export function agentRemember(workspace: string, input: RememberInput): MemoryResult {
  const { repo_slug, agent_type, tags } = input;
  const category = input.category.substring(0, MAX_CATEGORY_LENGTH);
  const content = input.content.substring(0, MAX_CONTENT_LENGTH);

  const path = storePath(workspace, repo_slug, agent_type);
  const entries = loadEntries(path);

  // Deduplicate
  const existing = entries.find(e => e.category === category && e.content === content);
  if (existing) {
    existing.last_accessed = Date.now() / 1000;
    existing.access_count += 1;
    saveEntries(path, entries);
    return { success: true, message: `Memory updated (access_count: ${existing.access_count})` };
  }

  const now = Date.now() / 1000;
  const entry: MemoryEntry = {
    category,
    content,
    created_at: now,
    last_accessed: now,
    access_count: 0,
    source_task_id: process.env.AGENT_TASK_ID || null,
    tags: tags || [],
  };
  entries.push(entry);

  // Evict oldest if over limit
  if (entries.length > MAX_MEMORIES_PER_STORE) {
    entries.sort((a, b) => a.last_accessed - b.last_accessed);
    entries.splice(0, entries.length - MAX_MEMORIES_PER_STORE);
  }

  saveEntries(path, entries);
  return { success: true, message: `Memory stored: [${category}] ${content.substring(0, 80)}...` };
}

export function agentRecall(workspace: string, input: RecallInput): MemoryResult {
  const { repo_slug, agent_type, category, tags, limit } = input;
  const maxResults = limit ?? 20;

  const path = storePath(workspace, repo_slug, agent_type);
  let entries = loadEntries(path);

  if (category) {
    entries = entries.filter(e => e.category === category);
  }
  if (tags && tags.length > 0) {
    const tagSet = new Set(tags);
    entries = entries.filter(e => e.tags.some(t => tagSet.has(t)));
  }

  // Sort by recency
  entries.sort((a, b) => b.last_accessed - a.last_accessed);
  entries = entries.slice(0, maxResults);

  // Touch accessed entries (update access metadata in the full store)
  if (entries.length > 0) {
    const fullEntries = loadEntries(path);
    const accessedContents = new Set(entries.map(e => `${e.category}:${e.content}`));
    const now = Date.now() / 1000;
    for (const e of fullEntries) {
      if (accessedContents.has(`${e.category}:${e.content}`)) {
        e.last_accessed = now;
        e.access_count += 1;
      }
    }
    saveEntries(path, fullEntries);
  }

  return {
    success: true,
    message: `Found ${entries.length} memories for ${repo_slug}/${agent_type}`,
    memories: entries,
  };
}
