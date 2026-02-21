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
  by_step: Record<string, number>
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

export interface LanguageMismatchMetrics {
  total_tasks_with_mismatches: number
  total_mismatch_events: number
  by_searched_language: Record<string, number>
  by_tool: Record<string, number>
  mismatch_rate: number
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
  language_mismatch: LanguageMismatchMetrics
  trends: TrendBucket[]
}

// ============== LLM Metrics ==============

export interface TaskCostSummary {
  task_id: string
  total_cost: number
  total_tokens_in: number
  total_tokens_out: number
  total_duration_ms: number
  llm_call_count: number
  token_efficiency: number
  primary_model: string
}

export interface ModelTierMetrics {
  tier: string
  call_count: number
  total_cost: number
  total_tokens_in: number
  total_tokens_out: number
  avg_cost_per_call: number
  avg_duration_ms: number
  cost_share_pct: number
}

export interface CostTrendBucket {
  timestamp: string
  total_cost: number
  total_tokens_in: number
  total_tokens_out: number
  call_count: number
  avg_duration_ms: number
}

export interface LatencyMetrics {
  sample_count: number
  avg_ms: number
  p50_ms: number
  p90_ms: number
  p99_ms: number
  max_ms: number
}

export interface LlmMetricsReport {
  generated_at: string
  time_range_hours: number
  tasks_with_llm_calls: number
  total_llm_calls: number
  total_cost: number
  total_tokens_in: number
  total_tokens_out: number
  overall_token_efficiency: number
  model_tiers: ModelTierMetrics[]
  latency: LatencyMetrics
  top_cost_tasks: TaskCostSummary[]
  trends: CostTrendBucket[]
}

// ============== Chain Metrics ==============

export interface StepTypeMetrics {
  step_id: string
  total_count: number
  success_count: number
  failure_count: number
  success_rate: number
  avg_duration_seconds: number
  p50_duration_seconds: number
  p90_duration_seconds: number
}

export interface ChainSummary {
  root_task_id: string
  workflow: string
  step_count: number
  attempt: number
  completed: boolean
  files_modified_count: number
  total_duration_seconds: number
}

export interface ChainMetricsReport {
  generated_at: string
  time_range_hours: number
  total_chains: number
  completed_chains: number
  chain_completion_rate: number
  avg_chain_depth: number
  avg_files_modified: number
  avg_attempts: number
  step_type_metrics: StepTypeMetrics[]
  top_failing_steps: StepTypeMetrics[]
  recent_chains: ChainSummary[]
}

// ============== Decomposition Metrics ==============

export interface DecompositionRateMetrics {
  tasks_evaluated: number
  tasks_decomposed: number
  decomposition_rate: number
}

export interface SubtaskDistribution {
  distribution: Record<string, number>  // JSON serializes int keys as strings
  avg_subtask_count: number
  min_subtask_count: number
  max_subtask_count: number
}

export interface EstimationSample {
  task_id: string
  estimated_lines: number
  actual_lines: number
  ratio: number
}

export interface EstimationAccuracy {
  sample_count: number
  avg_estimated: number
  avg_actual: number
  avg_ratio: number
  samples: EstimationSample[]
}

export interface FanInMetrics {
  decomposed_tasks: number
  fan_ins_created: number
  fan_in_success_rate: number
}

export interface DecompositionReport {
  generated_at: string
  time_range_hours: number
  rate: DecompositionRateMetrics
  distribution: SubtaskDistribution
  estimation: EstimationAccuracy
  fan_in: FanInMetrics
}

// ============== Git Metrics ==============

export interface TaskGitMetrics {
  root_task_id: string
  total_commits: number
  total_insertions: number
  total_deletions: number
  push_attempts: number
  push_successes: number
  first_edit_to_commit_secs: number | null
}

export interface GitMetricsSummary {
  avg_commits_per_task: number
  avg_insertions_per_commit: number
  avg_deletions_per_commit: number
  push_success_rate: number
  p50_edit_to_commit_secs: number | null
  p90_edit_to_commit_secs: number | null
}

export interface GitMetricsReport {
  generated_at: string
  time_range_hours: number
  total_tasks: number
  per_task: TaskGitMetrics[]
  summary: GitMetricsSummary
}

// ============== Performance Metrics ==============

export interface AgentPerformance {
  agent_id: string
  total_tasks: number
  completed_tasks: number
  failed_tasks: number
  success_rate: number
  avg_duration_seconds: number
  avg_tokens_per_task: number
  avg_cost_per_task: number
  total_cost: number
  retry_rate: number
}

export interface TaskTypePerformanceMetrics {
  task_type: string
  total_tasks: number
  success_rate: number
  avg_duration_seconds: number
  p50_tokens: number
  p90_tokens: number
  p99_tokens: number
  avg_tokens: number
  avg_cost: number
}

export interface HandoffSummary {
  transition: string
  count: number
  avg_total_ms: number
  p50_total_ms: number
  p90_total_ms: number
  avg_queue_wait_ms: number
  failed_count: number
  delayed_count: number
}

export interface PerformanceReport {
  generated_at: string
  time_range_hours: number
  overall_success_rate: number
  total_tasks: number
  total_cost: number
  agent_performance: AgentPerformance[]
  task_type_metrics: TaskTypePerformanceMetrics[]
  top_failures: Record<string, unknown>[]
  handoff_summaries: HandoffSummary[]
}

// ============== Waste Metrics ==============

export interface RootTaskWaste {
  root_task_id: string
  total_cost: number
  wasted_cost: number
  waste_ratio: number
  productive_cost: number
  total_tasks: number
  failed_tasks: number
  completed_tasks: number
  has_pr: boolean
  title: string
}

export interface WasteMetricsReport {
  generated_at: string
  time_range_hours: number
  roots_analyzed: number
  total_cost: number
  total_wasted_cost: number
  aggregate_waste_ratio: number
  avg_waste_ratio: number
  max_waste_ratio: number
  roots_with_zero_delivery: number
  top_waste_roots: RootTaskWaste[]
}

// ============== Review Cycle Metrics ==============

export interface StepBreakdown {
  workflow_step: string
  checks: number
  enforcements: number
  phase_resets: number
}

export interface ReviewCycleMetricsData {
  total_checks: number
  total_enforcements: number
  total_phase_resets: number
  total_halts: number
  enforcement_rate: number
  cap_violations: number
  violation_task_ids: string[]
  enforcement_count_distribution: Record<string, number>  // JSON serializes int keys as strings
  by_step: StepBreakdown[]
}

export interface ReviewCycleMetricsReport {
  generated_at: string
  time_range_hours: number
  metrics: ReviewCycleMetricsData
  raw_events: Record<string, unknown>[]
}

// ============== Verdict Metrics ==============

export interface VerdictMethodDistribution {
  total_verdicts: number
  by_method: Record<string, number>
  by_value: Record<string, number>
  fallback_rate: number
  override_rate: number
  ambiguous_rate: number
}

export interface VerdictPatternFrequency {
  category: string
  pattern: string
  match_count: number
  suppression_count: number
  false_positive_risk: number
}

export interface VerdictMetricsReport {
  generated_at: string
  time_range_hours: number
  total_tasks_with_verdicts: number
  distribution: VerdictMethodDistribution
  pattern_frequencies: VerdictPatternFrequency[]
  recent_audits: Record<string, unknown>[]
}

// ============== Failure Analysis ==============

export interface FailureCategory {
  category: string
  pattern: string
  count: number
  percentage: number
  affected_agents: string[]
  sample_errors: string[]
  recommendation: string | null
}

export interface FailureTrend {
  category: string
  weekly_count: number
  weekly_change_pct: number
  is_increasing: boolean
}

export interface FailureAnalysisReport {
  generated_at: string
  time_range_hours: number
  total_failures: number
  failure_rate: number
  categories: FailureCategory[]
  trends: FailureTrend[]
  top_recommendations: string[]
}

