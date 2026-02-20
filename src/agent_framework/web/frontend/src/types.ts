// Dashboard data types matching Python Pydantic models

export type AgentStatus = 'idle' | 'working' | 'completing' | 'dead'

export interface CurrentTask {
  id: string
  title: string
  type: string
  started_at: string
}

export interface ToolActivity {
  tool_name: string
  tool_input_summary: string | null
  started_at: string
  tool_call_count: number
}

export interface Agent {
  id: string
  name: string
  queue: string
  status: AgentStatus
  current_task: CurrentTask | null
  current_phase: string | null
  phases_completed: number
  elapsed_seconds: number | null
  last_updated: string | null
  tool_activity: ToolActivity | null
}

export interface QueueStats {
  queue_id: string
  agent_name: string
  pending_count: number
  oldest_task_age: number | null
}

export interface ActivityEvent {
  type: 'start' | 'complete' | 'fail' | 'phase'
  agent: string
  task_id: string
  title: string
  timestamp: string
  duration_ms: number | null
  retry_count: number | null
  phase: string | null
  error_message: string | null
  pr_url: string | null
}

export interface FailedTask {
  id: string
  title: string
  jira_key: string | null
  assigned_to: string
  retry_count: number
  last_error: string | null
  failed_at: string | null
}

export interface ActiveTask {
  id: string
  title: string
  status: 'pending' | 'in_progress'
  jira_key: string | null
  assigned_to: string
  created_at: string
  started_at: string | null
  task_type: string
  parent_task_id: string | null
}

export interface HealthCheck {
  name: string
  passed: boolean
  message: string | null
}

export interface HealthReport {
  passed: boolean
  checks: HealthCheck[]
  warnings: string[]
}

export interface DashboardState {
  agents: Agent[]
  queues: QueueStats[]
  events: ActivityEvent[]
  failed_tasks: FailedTask[]
  health: HealthReport
  is_paused: boolean
  uptime_seconds: number
}

// API Response types
export interface SuccessResponse {
  success: boolean
  message: string
}

export interface AgentActionResponse {
  success: boolean
  agent_id: string
  action: string
  message: string
}

export interface TaskActionResponse {
  success: boolean
  task_id: string
  action: string
  message: string
}

// Operation request types
export interface WorkRequest {
  goal: string
  repository: string
  workflow?: string
}

export interface AnalyzeRequest {
  repository: string
  severity?: 'all' | 'critical' | 'high' | 'medium'
  max_issues?: number
  dry_run?: boolean
  focus?: string
}

export interface RunTicketRequest {
  ticket_id: string
  agent?: string
}

export interface CreateTaskRequest {
  title: string
  description: string
  task_type: string
  assigned_to: string
  repository?: string
  priority?: number
}

export interface OperationResponse {
  success: boolean
  task_id: string | null
  message: string
}

// Log streaming types
export interface LogEntry {
  id?: number  // Client-side ID for Vue key binding
  agent: string
  task_id?: string  // For claude-cli logs, links to specific task
  source?: 'agent' | 'claude-cli'  // Log source type
  line: string
  timestamp: string
  level: string | null
}

// Agentic metrics types (mirrors analytics.agentic_metrics Pydantic models)

export interface MemoryMetrics {
  total_recalls: number
  tasks_with_recall: number
  avg_chars_injected: number
  recall_rate: number
  completion_rate_with_recall: number
  completion_rate_without_recall: number
  recall_usefulness_delta: number
}

export interface SelfEvalMetrics {
  total_evals: number
  pass_count: number
  fail_count: number
  auto_pass_count: number
  catch_rate: number
}

export interface ReplanMetrics {
  total_replans: number
  tasks_with_replan: number
  tasks_completed_after_replan: number
  trigger_rate: number
  success_rate_after_replan: number
}

export interface SpecializationMetrics {
  distribution: Record<string, number>
  total_active_agents: number
}

export interface DebateMetrics {
  available: boolean
  total_debates: number
  successful_debates: number
  confidence_distribution: Record<string, number>
  success_rate: number
  avg_trade_offs_count: number
}

export interface ContextBudgetMetrics {
  sample_count: number
  avg_prompt_length: number
  max_prompt_length: number
  min_prompt_length: number
  p50_prompt_length: number
  p90_prompt_length: number
}

export interface ToolUsageMetrics {
  total_tasks_analyzed: number
  avg_tool_calls_per_task: number
  max_tool_calls: number
  tool_distribution: Record<string, number>
  duplicate_read_rate: number
  avg_duplicate_reads_per_task: number
  avg_read_before_write_ratio: number
  avg_edit_density: number
  top_tasks_by_calls: Record<string, number>
  p90_tool_calls: number
  exploration_alert_threshold: number
  sessions_exceeding_threshold: number
  by_agent: Record<string, number>
}

export interface TrendBucket {
  timestamp: string
  memory_recall_rate: number
  self_eval_catch_rate: number
  replan_trigger_rate: number
  avg_prompt_length: number
  task_count: number
  avg_tool_calls: number
  avg_edit_density: number
  sessions_exceeding_threshold: number
}

export interface AgenticMetricsReport {
  generated_at: string
  time_range_hours: number
  total_observed_tasks: number
  memory: MemoryMetrics
  self_eval: SelfEvalMetrics
  replan: ReplanMetrics
  specialization: SpecializationMetrics
  debate: DebateMetrics
  context_budget: ContextBudgetMetrics
  tool_usage: ToolUsageMetrics
  trends: TrendBucket[]
}

