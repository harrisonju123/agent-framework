/**
 * Task types and interfaces matching the Python agent-framework Task model.
 * See: src/agent_framework/core/task.py
 */

export type TaskStatus = "pending" | "in_progress" | "testing" | "awaiting_review" | "completed" | "failed";

export type TaskType =
  | "testing"
  | "verification"
  | "qa_verification"
  | "fix"
  | "bugfix"
  | "bug-fix"
  | "coordination"
  | "status_report"
  | "documentation"
  | "implementation"
  | "architecture"
  | "planning"
  | "review"
  | "enhancement"
  | "pr_request"
  | "escalation"
  | "analysis";

export type AgentId = "engineer" | "qa" | "architect" | "product-owner" | "code-reviewer" | "testing" | "static-analysis" | "repo-analyzer";

export interface PlanDocument {
  objectives: string[];
  approach: string[];
  risks?: string[];
  success_criteria: string[];
  files_to_modify?: string[];
  dependencies?: string[];
}

export interface TaskContext {
  jira_key?: string;
  jira_project?: string;
  github_repo?: string;
  epic_key?: string;
  workflow?: "simple" | "standard" | "full";
  use_worktree?: boolean;
  [key: string]: unknown;
}

export interface Task {
  id: string;
  type: TaskType;
  status: TaskStatus;
  priority: number;
  created_by: string;
  assigned_to: AgentId;
  created_at: string;
  title: string;
  description: string;

  depends_on?: string[];
  blocks?: string[];
  acceptance_criteria?: string[];
  deliverables?: string[];
  notes?: string[];
  context?: TaskContext;

  retry_count?: number;
  last_failed_at?: string;

  started_at?: string;
  started_by?: string;
  completed_at?: string;
  completed_by?: string;
  failed_at?: string;
  failed_by?: string;

  failed_task_id?: string;
  needs_human_review?: boolean;

  estimated_effort?: string;

  result_summary?: string;
  last_error?: string;

  optimization_override?: boolean;
  optimization_override_reason?: string;

  plan?: PlanDocument;
}

export interface QueueTaskInput {
  agent_id: AgentId;
  task_type: TaskType;
  title: string;
  description: string;
  context?: TaskContext;
  depends_on?: string[];
  priority?: number;
  acceptance_criteria?: string[];
  plan?: PlanDocument;
}

export interface QueueTaskResult {
  success: boolean;
  task_id: string;
  queue: string;
  message: string;
}

export interface QueueStatus {
  queue: string;
  pending: number;
  in_progress: number;
  total: number;
}

export interface TaskSummary {
  id: string;
  title: string;
  type: TaskType;
  status: TaskStatus;
  jira_key?: string;
  epic_key?: string;
  depends_on?: string[];
  created_at: string;
}

export interface EpicProgress {
  epic_key: string;
  total_tasks: number;
  completed: number;
  in_progress: number;
  pending: number;
  failed: number;
  percent_complete: number;
  tasks: TaskSummary[];
}

// Inter-agent consultation types

export interface ConsultationInput {
  target_agent: AgentId;
  question: string;
  context?: string;
}

export interface ShareKnowledgeInput {
  topic: string;
  key: string;
  value: string;
}

export interface GetKnowledgeInput {
  topic: string;
  key?: string;
  max_age_hours?: number;
}

// Multi-perspective debate types

export interface DebateInput {
  topic: string;
  context?: string;
  advocate_position?: string;
  critic_position?: string;
}

export interface DebateResult {
  success: boolean;
  debate_id: string;
  advocate_argument: string;
  critic_argument: string;
  synthesis: string;
  confidence: "high" | "medium" | "low";
  recommendation: string;
  trade_offs: string[];
  consultations_remaining: number;
}
