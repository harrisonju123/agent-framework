/**
 * Read cache for cross-step file read deduplication.
 *
 * Stores file-path → summary mappings so downstream agents in a workflow
 * chain can skip re-reading files that earlier agents already analyzed.
 * Uses atomic writes (temp + rename) following the knowledge.ts pattern.
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

const MAX_SUMMARY_LENGTH = 2000;
const MAX_PATH_LENGTH = 500;
const MAX_ID_LENGTH = 200;

interface FileReadEntry {
  summary: string;
  read_by: string;
  read_at: string;
  workflow_step: string;
}

interface ReadCacheStore {
  root_task_id: string;
  entries: Record<string, FileReadEntry>;
}

export interface CacheFileReadResult {
  success: boolean;
  file_path: string;
  message: string;
}

export interface GetCachedReadsResult {
  success: boolean;
  entry_count: number;
  entries: Record<string, FileReadEntry>;
  message: string;
}

function ensureDirectory(dirPath: string): void {
  if (!existsSync(dirPath)) {
    mkdirSync(dirPath, { recursive: true });
  }
}

function sanitizeId(id: string): string {
  return id.replace(/[^a-zA-Z0-9_-]/g, "_").substring(0, MAX_ID_LENGTH);
}

/** Strip worktree prefix for cache portability across chain steps. */
export function toRelativePath(filePath: string, workingDir?: string): string {
  if (!workingDir || !filePath.startsWith("/")) {
    return filePath;
  }
  const prefix = workingDir.replace(/\/+$/, "") + "/";
  if (filePath.startsWith(prefix)) {
    return filePath.substring(prefix.length);
  }
  return filePath;
}

export function cacheFileRead(
  workspace: string,
  rootTaskId: string,
  filePath: string,
  summary: string,
  agentId: string,
  workflowStep: string,
): CacheFileReadResult {
  if (!filePath || filePath.length > MAX_PATH_LENGTH) {
    return {
      success: false,
      file_path: filePath || "",
      message: "Invalid or too-long file path",
    };
  }

  const safeSummary = summary.substring(0, MAX_SUMMARY_LENGTH);
  const safeRootId = sanitizeId(rootTaskId);

  // Normalize to repo-relative path for cross-worktree portability
  const cacheKey = toRelativePath(filePath, process.env.AGENT_WORKING_DIR);

  const cacheDir = join(workspace, ".agent-communication", "read-cache");
  ensureDirectory(cacheDir);

  const cacheFile = join(cacheDir, `${safeRootId}.json`);
  const tmpFile = join(cacheDir, `${safeRootId}.tmp.${process.pid}`);

  let existing: ReadCacheStore = {
    root_task_id: rootTaskId,
    entries: {},
  };

  if (existsSync(cacheFile)) {
    try {
      existing = JSON.parse(readFileSync(cacheFile, "utf-8"));
    } catch {
      // Corrupted file — start fresh
      existing = { root_task_id: rootTaskId, entries: {} };
    }
  }

  existing.entries[cacheKey] = {
    summary: safeSummary,
    read_by: agentId,
    read_at: new Date().toISOString(),
    workflow_step: workflowStep,
  };

  try {
    writeFileSync(tmpFile, JSON.stringify(existing, null, 2));
    renameSync(tmpFile, cacheFile);
  } catch (err) {
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
    file_path: cacheKey,
    message: `Cached read: ${cacheKey}`,
  };
}

export function getCachedReads(
  workspace: string,
  rootTaskId: string,
): GetCachedReadsResult {
  const safeRootId = sanitizeId(rootTaskId);
  const cacheFile = join(
    workspace,
    ".agent-communication",
    "read-cache",
    `${safeRootId}.json`,
  );

  if (!existsSync(cacheFile)) {
    return {
      success: true,
      entry_count: 0,
      entries: {},
      message: "No cached reads found for this workflow chain",
    };
  }

  let data: ReadCacheStore;
  try {
    data = JSON.parse(readFileSync(cacheFile, "utf-8"));
  } catch {
    return {
      success: false,
      entry_count: 0,
      entries: {},
      message: "Failed to parse read cache file",
    };
  }

  const entries = data.entries || {};
  const count = Object.keys(entries).length;

  return {
    success: true,
    entry_count: count,
    entries,
    message: `Found ${count} cached file reads from previous agents`,
  };
}
