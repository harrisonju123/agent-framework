/**
 * Task queue tool implementations.
 * Interacts with the file-based queue system in .agent-communication/queues/
 */

import { existsSync, mkdirSync, writeFileSync, readFileSync, readdirSync } from "fs";
import { join } from "path";
import type {
  Task,
  QueueTaskInput,
  QueueTaskResult,
  QueueStatus,
  TaskSummary,
  EpicProgress,
  AgentId,
} from "./types.js";

function generateTaskId(agentId: string, taskType: string): string {
  const timestamp = Date.now();
  const random = Math.random().toString(36).substring(2, 8);
  return `${taskType}-${agentId}-${timestamp}-${random}`;
}

function getQueuePath(workspace: string): string {
  return join(workspace, ".agent-communication", "queues");
}

function getCompletedPath(workspace: string): string {
  return join(workspace, ".agent-communication", "completed");
}

function ensureDirectory(path: string): void {
  if (!existsSync(path)) {
    mkdirSync(path, { recursive: true });
  }
}

function loadTask(filePath: string): Task | null {
  try {
    const content = readFileSync(filePath, "utf-8");
    return JSON.parse(content) as Task;
  } catch {
    return null;
  }
}

function loadAllTasksFromDir(dirPath: string): Task[] {
  const tasks: Task[] = [];
  if (!existsSync(dirPath)) return tasks;

  const files = readdirSync(dirPath).filter((f) => f.endsWith(".json") && !f.endsWith(".tmp"));
  for (const file of files) {
    const task = loadTask(join(dirPath, file));
    if (task) tasks.push(task);
  }
  return tasks;
}

export async function queueTaskForAgent(
  workspace: string,
  input: QueueTaskInput,
  createdBy: string
): Promise<QueueTaskResult> {
  const taskId = generateTaskId(input.agent_id, input.task_type);
  const queuePath = getQueuePath(workspace);
  const agentQueuePath = join(queuePath, input.agent_id);

  ensureDirectory(agentQueuePath);

  const task: Task = {
    id: taskId,
    type: input.task_type,
    status: "pending",
    priority: input.priority ?? 50,
    created_by: createdBy,
    assigned_to: input.agent_id,
    created_at: new Date().toISOString(),
    title: input.title,
    description: input.description,
    depends_on: input.depends_on ?? [],
    blocks: [],
    acceptance_criteria: input.acceptance_criteria ?? [],
    deliverables: [],
    notes: [],
    context: input.context ?? {},
    retry_count: 0,
    plan: input.plan,
  };

  const taskFile = join(agentQueuePath, `${taskId}.json`);
  const tmpFile = join(agentQueuePath, `${taskId}.json.tmp`);

  // Atomic write: write to tmp then rename
  writeFileSync(tmpFile, JSON.stringify(task, null, 2));
  const fs = await import("fs/promises");
  await fs.rename(tmpFile, taskFile);

  return {
    success: true,
    task_id: taskId,
    queue: input.agent_id,
    message: `Task ${taskId} queued for ${input.agent_id}`,
  };
}

export function getQueueStatus(workspace: string): QueueStatus[] {
  const queuePath = getQueuePath(workspace);
  const results: QueueStatus[] = [];

  const agentQueues: AgentId[] = ["engineer", "qa", "architect", "product-owner", "code-reviewer", "testing", "static-analysis"];

  for (const agentId of agentQueues) {
    const agentQueuePath = join(queuePath, agentId);
    const tasks = loadAllTasksFromDir(agentQueuePath);

    const pending = tasks.filter((t) => t.status === "pending").length;
    const inProgress = tasks.filter((t) => t.status === "in_progress").length;

    results.push({
      queue: agentId,
      pending,
      in_progress: inProgress,
      total: tasks.length,
    });
  }

  return results;
}

export function listPendingTasks(workspace: string, agentId: AgentId): TaskSummary[] {
  const queuePath = getQueuePath(workspace);
  const agentQueuePath = join(queuePath, agentId);
  const tasks = loadAllTasksFromDir(agentQueuePath);

  return tasks
    .filter((t) => t.status === "pending")
    .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
    .map((t) => ({
      id: t.id,
      title: t.title,
      type: t.type,
      status: t.status,
      jira_key: t.context?.jira_key,
      epic_key: t.context?.epic_key,
      depends_on: t.depends_on,
      created_at: t.created_at,
    }));
}

export function getTaskDetails(workspace: string, taskId: string): Task | null {
  const queuePath = getQueuePath(workspace);
  const completedPath = getCompletedPath(workspace);

  // Search in all agent queues
  const agentQueues: AgentId[] = ["engineer", "qa", "architect", "product-owner", "code-reviewer", "testing", "static-analysis"];
  for (const agentId of agentQueues) {
    const taskFile = join(queuePath, agentId, `${taskId}.json`);
    if (existsSync(taskFile)) {
      return loadTask(taskFile);
    }
  }

  // Search in completed directory
  const completedFile = join(completedPath, `${taskId}.json`);
  if (existsSync(completedFile)) {
    return loadTask(completedFile);
  }

  return null;
}

export function getEpicProgress(workspace: string, epicKey: string): EpicProgress {
  const queuePath = getQueuePath(workspace);
  const completedPath = getCompletedPath(workspace);
  const allTasks: Task[] = [];

  // Collect tasks from all queues
  const agentQueues: AgentId[] = ["engineer", "qa", "architect", "product-owner", "code-reviewer", "testing", "static-analysis"];
  for (const agentId of agentQueues) {
    const agentQueuePath = join(queuePath, agentId);
    const tasks = loadAllTasksFromDir(agentQueuePath);
    allTasks.push(...tasks);
  }

  // Collect completed tasks
  const completedTasks = loadAllTasksFromDir(completedPath);
  allTasks.push(...completedTasks);

  // Filter by epic
  const epicTasks = allTasks.filter((t) => t.context?.epic_key === epicKey);

  const completed = epicTasks.filter((t) => t.status === "completed").length;
  const inProgress = epicTasks.filter((t) => t.status === "in_progress").length;
  const pending = epicTasks.filter((t) => t.status === "pending").length;
  const failed = epicTasks.filter((t) => t.status === "failed").length;
  const total = epicTasks.length;

  const percentComplete = total > 0 ? Math.round((completed / total) * 100) : 0;

  const taskSummaries: TaskSummary[] = epicTasks.map((t) => ({
    id: t.id,
    title: t.title,
    type: t.type,
    status: t.status,
    jira_key: t.context?.jira_key,
    epic_key: t.context?.epic_key,
    depends_on: t.depends_on,
    created_at: t.created_at,
  }));

  return {
    epic_key: epicKey,
    total_tasks: total,
    completed,
    in_progress: inProgress,
    pending,
    failed,
    percent_complete: percentComplete,
    tasks: taskSummaries,
  };
}
