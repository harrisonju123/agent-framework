import { ref, computed, provide, inject, type InjectionKey, type Ref, type ComputedRef } from 'vue'
import { useWebSocket } from './useWebSocket'
import { useLogStream } from './useLogStream'
import { useApi } from './useApi'
import { useToast } from 'primevue/usetoast'
import { useConfirm } from 'primevue/useconfirm'
import type { CheckpointData, LogEntry } from '../types'

export interface AppState {
  // WebSocket state
  state: ReturnType<typeof useWebSocket>['state']
  connected: Ref<boolean>
  wsError: Ref<string | null>
  reconnect: () => void
  reconnecting: Ref<boolean>
  reconnectAttempt: Ref<number>

  // Log stream
  logs: Ref<LogEntry[]>
  logsConnected: Ref<boolean>
  logsClear: () => void
  logsReconnect: () => void

  // API methods
  restartAgent: ReturnType<typeof useApi>['restartAgent']
  retryTask: ReturnType<typeof useApi>['retryTask']
  approveCheckpoint: ReturnType<typeof useApi>['approveCheckpoint']
  rejectCheckpoint: ReturnType<typeof useApi>['rejectCheckpoint']
  pauseSystem: ReturnType<typeof useApi>['pauseSystem']
  resumeSystem: ReturnType<typeof useApi>['resumeSystem']
  startAllAgents: ReturnType<typeof useApi>['startAllAgents']
  stopAllAgents: ReturnType<typeof useApi>['stopAllAgents']
  createWork: ReturnType<typeof useApi>['createWork']
  analyzeRepo: ReturnType<typeof useApi>['analyzeRepo']
  runTicket: ReturnType<typeof useApi>['runTicket']
  loading: Ref<boolean>
  apiError: Ref<string | null>

  // Computed dashboard state
  isPaused: ComputedRef<boolean>
  uptime: ComputedRef<number>
  agents: ComputedRef<ReturnType<typeof useWebSocket>['state']['value'] extends null ? never[] : NonNullable<ReturnType<typeof useWebSocket>['state']['value']>['agents']>
  queues: ComputedRef<any[]>
  events: ComputedRef<any[]>
  failedTasks: ComputedRef<any[]>
  pendingCheckpoints: ComputedRef<CheckpointData[]>
  health: ComputedRef<any>
  agentIds: ComputedRef<string[]>
  uptimeDisplay: ComputedRef<string>
  queueSummary: ComputedRef<string>

  // Setup
  setupComplete: Ref<boolean>
  showSetupPrompt: Ref<boolean>
  checkSetupStatus: () => Promise<void>
  handleSetupComplete: () => void
  dismissSetupPrompt: () => void

  // Action handlers
  handleRestart: (agentId: string) => Promise<void>
  handleStart: () => Promise<void>
  handleStop: () => Promise<void>
  handlePause: () => Promise<void>
  handleRetryAll: () => Promise<void>
  handleApproveAll: () => Promise<void>
  handleRetryTask: (taskId: string) => Promise<void>

  // Dialog visibility (shared so keyboard shortcuts + TopBar can both toggle)
  showWorkDialog: Ref<boolean>
  showAnalyzeDialog: Ref<boolean>
  showTicketDialog: Ref<boolean>

  // Dialog helpers
  showToast: (message: string, type: 'success' | 'error' | 'info') => void
  showConfirm: (title: string, message: string, action: () => Promise<void>, destructive?: boolean) => void
}

const APP_STATE_KEY: InjectionKey<AppState> = Symbol('appState')

export function provideAppState(): AppState {
  const { state, connected, error: wsError, reconnect, reconnecting, reconnectAttempt } = useWebSocket()
  const { logs, connected: logsConnected, clear: logsClear, reconnect: logsReconnect } = useLogStream()
  const {
    restartAgent, retryTask, approveCheckpoint, rejectCheckpoint, pauseSystem, resumeSystem,
    startAllAgents, stopAllAgents, createWork, analyzeRepo, runTicket,
    loading, error: apiError,
  } = useApi()

  const toast = useToast()
  const confirm = useConfirm()

  // Setup state
  const setupComplete = ref(true)
  const showSetupPrompt = ref(false)

  // Dialog visibility refs (shared between keyboard shortcuts and TopBar)
  const showWorkDialog = ref(false)
  const showAnalyzeDialog = ref(false)
  const showTicketDialog = ref(false)

  // Computed dashboard values
  const isPaused = computed(() => state.value?.is_paused ?? false)
  const uptime = computed(() => state.value?.uptime_seconds ?? 0)
  const agents = computed(() => state.value?.agents ?? [])
  const queues = computed(() => state.value?.queues ?? [])
  const events = computed(() => state.value?.events ?? [])
  const failedTasks = computed(() => state.value?.failed_tasks ?? [])
  const pendingCheckpoints = computed(() => state.value?.pending_checkpoints ?? [])
  const health = computed(() => state.value?.health ?? { passed: true, checks: [], warnings: [] })
  const agentIds = computed(() => agents.value.map((a: any) => a.id))

  const uptimeDisplay = computed(() => {
    const minutes = Math.floor(uptime.value / 60)
    const seconds = uptime.value % 60
    return `${minutes}m ${seconds}s`
  })

  const queueSummary = computed(() => {
    return queues.value.map((q: any) => `${q.queue_id}(${q.pending_count})`).join(' ')
  })

  // Toast wrapper
  function showToast(message: string, type: 'success' | 'error' | 'info') {
    const severityMap: Record<string, string> = { success: 'success', error: 'error', info: 'info' }
    const summaryMap: Record<string, string> = { success: 'Success', error: 'Error', info: 'Info' }
    toast.add({
      severity: severityMap[type] as any,
      summary: summaryMap[type],
      detail: message,
      life: 3000,
    })
  }

  // Confirm wrapper
  function showConfirm(title: string, message: string, action: () => Promise<void>, destructive = false) {
    confirm.require({
      header: title,
      message,
      acceptClass: destructive ? '!bg-red-600 !border-red-600' : undefined,
      accept: action,
    })
  }

  // Setup handlers
  async function checkSetupStatus() {
    try {
      const response = await fetch('/api/setup/status')
      const status = await response.json()
      setupComplete.value = status.ready_to_start
      if (!setupComplete.value && !localStorage.getItem('dismissed_setup')) {
        showSetupPrompt.value = true
      }
    } catch (e) {
      console.error('Failed to check setup status:', e)
    }
  }

  function handleSetupComplete() {
    setupComplete.value = true
    showSetupPrompt.value = false
    showToast('Configuration saved. Click "Start All" to begin.', 'success')
  }

  function dismissSetupPrompt() {
    showSetupPrompt.value = false
    localStorage.setItem('dismissed_setup', 'true')
  }

  // Action handlers
  async function handleRestart(agentId: string) {
    const result = await restartAgent(agentId)
    if (result?.success) {
      showToast(`Agent ${agentId} restarted`, 'success')
    } else if (apiError.value) {
      showToast(apiError.value, 'error')
    }
  }

  async function handleStart() {
    const result = await startAllAgents()
    if (result?.success) {
      showToast(result.message, 'success')
    } else if (apiError.value) {
      showToast(apiError.value, 'error')
    }
  }

  async function handleStop() {
    const result = await stopAllAgents()
    if (result?.success) {
      showToast(result.message, 'success')
    } else if (apiError.value) {
      showToast(apiError.value, 'error')
    }
  }

  async function handlePause() {
    if (isPaused.value) {
      const result = await resumeSystem()
      if (result?.success) {
        showToast('System resumed', 'success')
      } else if (apiError.value) {
        showToast(apiError.value, 'error')
      }
    } else {
      const result = await pauseSystem()
      if (result?.success) {
        showToast('System paused', 'info')
      } else if (apiError.value) {
        showToast(apiError.value, 'error')
      }
    }
  }

  async function handleRetryAll() {
    const tasks = failedTasks.value
    if (tasks.length === 0) return
    let succeeded = 0
    let failed = 0
    for (const task of tasks) {
      const result = await retryTask(task.id)
      if (result?.success) { succeeded++ } else { failed++ }
    }
    if (failed === 0) {
      showToast(`Retrying ${succeeded} failed task(s)`, 'info')
    } else {
      showToast(`Retried ${succeeded}, failed to retry ${failed} task(s)`, 'error')
    }
  }

  async function handleApproveAll() {
    const checkpoints = pendingCheckpoints.value
    if (checkpoints.length === 0) return
    let succeeded = 0
    let failed = 0
    for (const cp of checkpoints) {
      const result = await approveCheckpoint(cp.id)
      if (result?.success) { succeeded++ } else { failed++ }
    }
    if (failed === 0) {
      showToast(`Approved ${succeeded} checkpoint(s)`, 'success')
    } else {
      showToast(`Approved ${succeeded}, failed to approve ${failed} checkpoint(s)`, 'error')
    }
  }

  async function handleRetryTask(taskId: string) {
    const result = await retryTask(taskId)
    if (result?.success) {
      showToast(`Task ${taskId} queued for retry`, 'success')
    } else if (apiError.value) {
      showToast(apiError.value, 'error')
    }
  }

  const appState: AppState = {
    state, connected, wsError, reconnect, reconnecting, reconnectAttempt,
    logs, logsConnected, logsClear, logsReconnect,
    restartAgent, retryTask, approveCheckpoint, rejectCheckpoint, pauseSystem, resumeSystem,
    startAllAgents, stopAllAgents, createWork, analyzeRepo, runTicket,
    loading, apiError,
    isPaused, uptime, agents, queues, events, failedTasks, pendingCheckpoints,
    health, agentIds, uptimeDisplay, queueSummary,
    setupComplete, showSetupPrompt, checkSetupStatus, handleSetupComplete, dismissSetupPrompt,
    handleRestart, handleStart, handleStop, handlePause, handleRetryAll, handleApproveAll, handleRetryTask,
    showWorkDialog, showAnalyzeDialog, showTicketDialog,
    showToast, showConfirm,
  }

  provide(APP_STATE_KEY, appState)
  return appState
}

export function useAppState(): AppState {
  const state = inject(APP_STATE_KEY)
  if (!state) {
    throw new Error('useAppState() called without provideAppState() in ancestor component')
  }
  return state
}
