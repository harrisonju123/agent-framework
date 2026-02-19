import { ref } from 'vue'
import type {
  ActiveTask,
  AgentActionResponse,
  TaskActionResponse,
  SuccessResponse,
  WorkRequest,
  AnalyzeRequest,
  RunTicketRequest,
  CreateTaskRequest,
  OperationResponse,
  AgenticMetrics,
} from '../types'

const baseUrl = '/api'

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${url}`, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.detail || `HTTP ${response.status}`)
  }

  return response.json()
}

export function useApi() {
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function withLoading<T>(fn: () => Promise<T>): Promise<T | null> {
    loading.value = true
    error.value = null
    try {
      return await fn()
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Unknown error'
      return null
    } finally {
      loading.value = false
    }
  }

  // Agent actions
  async function startAgent(agentId: string): Promise<AgentActionResponse | null> {
    return withLoading(() =>
      fetchJson<AgentActionResponse>(`/agents/${agentId}/start`, { method: 'POST' })
    )
  }

  async function stopAgent(agentId: string): Promise<AgentActionResponse | null> {
    return withLoading(() =>
      fetchJson<AgentActionResponse>(`/agents/${agentId}/stop`, { method: 'POST' })
    )
  }

  async function restartAgent(agentId: string): Promise<AgentActionResponse | null> {
    return withLoading(() =>
      fetchJson<AgentActionResponse>(`/agents/${agentId}/restart`, { method: 'POST' })
    )
  }

  // Task actions
  async function retryTask(taskId: string): Promise<TaskActionResponse | null> {
    return withLoading(() =>
      fetchJson<TaskActionResponse>(`/tasks/${taskId}/retry`, { method: 'POST' })
    )
  }

  async function cancelTask(taskId: string, reason?: string): Promise<TaskActionResponse | null> {
    return withLoading(() =>
      fetchJson<TaskActionResponse>(`/tasks/${taskId}/cancel`, {
        method: 'POST',
        ...(reason ? { body: JSON.stringify({ reason }) } : {}),
      })
    )
  }

  async function deleteTask(taskId: string): Promise<TaskActionResponse | null> {
    return withLoading(() =>
      fetchJson<TaskActionResponse>(`/tasks/${taskId}`, { method: 'DELETE' })
    )
  }

  async function createTask(request: CreateTaskRequest): Promise<OperationResponse | null> {
    return withLoading(() =>
      fetchJson<OperationResponse>('/tasks', {
        method: 'POST',
        body: JSON.stringify(request),
      })
    )
  }

  // Polling fetches — no loading state to avoid UI flicker
  async function getActiveTasks(): Promise<ActiveTask[]> {
    try {
      return await fetchJson<ActiveTask[]>('/tasks/active')
    } catch {
      return []
    }
  }

  // System actions
  async function pauseSystem(): Promise<SuccessResponse | null> {
    return withLoading(() =>
      fetchJson<SuccessResponse>('/system/pause', { method: 'POST' })
    )
  }

  async function resumeSystem(): Promise<SuccessResponse | null> {
    return withLoading(() =>
      fetchJson<SuccessResponse>('/system/resume', { method: 'POST' })
    )
  }

  // Bulk agent actions
  async function startAllAgents(): Promise<SuccessResponse | null> {
    return withLoading(() =>
      fetchJson<SuccessResponse>('/agents/start-all', { method: 'POST' })
    )
  }

  async function stopAllAgents(): Promise<SuccessResponse | null> {
    return withLoading(() =>
      fetchJson<SuccessResponse>('/agents/stop-all', { method: 'POST' })
    )
  }

  // Operations
  async function createWork(request: WorkRequest): Promise<OperationResponse | null> {
    return withLoading(() =>
      fetchJson<OperationResponse>('/operations/work', {
        method: 'POST',
        body: JSON.stringify(request),
      })
    )
  }

  async function analyzeRepo(request: AnalyzeRequest): Promise<OperationResponse | null> {
    return withLoading(() =>
      fetchJson<OperationResponse>('/operations/analyze', {
        method: 'POST',
        body: JSON.stringify(request),
      })
    )
  }

  async function runTicket(request: RunTicketRequest): Promise<OperationResponse | null> {
    return withLoading(() =>
      fetchJson<OperationResponse>('/operations/run-ticket', {
        method: 'POST',
        body: JSON.stringify(request),
      })
    )
  }

  // Observability metrics — no loading state to avoid flicker on background polls
  async function getAgenticMetrics(hours: number = 24): Promise<AgenticMetrics | null> {
    try {
      return await fetchJson<AgenticMetrics>(`/observability/metrics?hours=${hours}`)
    } catch {
      return null
    }
  }

  return {
    loading,
    error,
    startAgent,
    stopAgent,
    restartAgent,
    retryTask,
    cancelTask,
    deleteTask,
    createTask,
    getActiveTasks,
    pauseSystem,
    resumeSystem,
    startAllAgents,
    stopAllAgents,
    createWork,
    analyzeRepo,
    runTicket,
    getAgenticMetrics,
  }
}
