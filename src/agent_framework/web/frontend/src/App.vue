<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useWebSocket } from './composables/useWebSocket'
import { useLogStream } from './composables/useLogStream'
import { useApi } from './composables/useApi'
import { useKeyboard } from './composables/useKeyboard'
import AgentCard from './components/AgentCard.vue'
import QueuePanel from './components/QueuePanel.vue'
import ActivityFeed from './components/ActivityFeed.vue'
import FailedTasks from './components/FailedTasks.vue'
import HealthStatus from './components/HealthStatus.vue'
import LogViewer from './components/LogViewer.vue'
import Modal from './components/Modal.vue'
import Toast from './components/Toast.vue'
import ConfirmDialog from './components/ConfirmDialog.vue'
import SetupWizard from './components/SetupWizard.vue'
import type { ModalType } from './types'

const { state, connected, error: wsError, reconnect, reconnecting, reconnectAttempt } = useWebSocket()
const { logs, connected: logsConnected } = useLogStream()
const {
  restartAgent,
  retryTask,
  pauseSystem,
  resumeSystem,
  startAllAgents,
  stopAllAgents,
  createWork,
  analyzeRepo,
  runTicket,
  loading,
  error: apiError,
} = useApi()

// Modal state
const activeModal = ref<ModalType | 'setup'>(null)

// Setup status
const setupComplete = ref(true)
const showSetupPrompt = ref(false)

// Confirm dialog state
const confirmDialog = ref<{
  open: boolean
  title: string
  message: string
  action: () => Promise<void>
  destructive: boolean
}>({
  open: false,
  title: '',
  message: '',
  action: async () => {},
  destructive: false,
})

// Toast notification state
const toast = ref<{
  show: boolean
  message: string
  type: 'success' | 'error' | 'info'
}>({
  show: false,
  message: '',
  type: 'info',
})

// Log panel collapsed state
const logsExpanded = ref(false)

// Form state
const workForm = ref({
  goal: '',
  repository: '',
  workflow: 'default' as string,
})

const analyzeForm = ref({
  repository: '',
  severity: 'high' as 'all' | 'critical' | 'high' | 'medium',
  max_issues: 50,
  dry_run: false,
  focus: '',
})

const ticketForm = ref({
  ticket_id: '',
  agent: '' as string,
})

// Computed values
const isPaused = computed(() => state.value?.is_paused ?? false)
const uptime = computed(() => state.value?.uptime_seconds ?? 0)
const agents = computed(() => state.value?.agents ?? [])
const queues = computed(() => state.value?.queues ?? [])
const events = computed(() => state.value?.events ?? [])
const failedTasks = computed(() => state.value?.failed_tasks ?? [])
const health = computed(() => state.value?.health ?? { passed: true, checks: [], warnings: [] })
const agentIds = computed(() => agents.value.map(a => a.id))

const uptimeDisplay = computed(() => {
  const minutes = Math.floor(uptime.value / 60)
  const seconds = uptime.value % 60
  return `${minutes}m ${seconds}s`
})

const queueSummary = computed(() => {
  return queues.value
    .map(q => `${q.queue_id}(${q.pending_count})`)
    .join(' ')
})

// Validation helpers
const repoPattern = /^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_.-]+$/

const workFormValid = computed(() => {
  return workForm.value.goal.length >= 10 &&
    repoPattern.test(workForm.value.repository)
})

const workFormErrors = computed(() => {
  const errors: string[] = []
  if (workForm.value.goal && workForm.value.goal.length < 10) {
    errors.push('Goal must be at least 10 characters')
  }
  if (workForm.value.repository && !repoPattern.test(workForm.value.repository)) {
    errors.push('Repository must be in owner/repo format')
  }
  return errors
})

const analyzeFormValid = computed(() => {
  return repoPattern.test(analyzeForm.value.repository)
})

const analyzeFormErrors = computed(() => {
  const errors: string[] = []
  if (analyzeForm.value.repository && !repoPattern.test(analyzeForm.value.repository)) {
    errors.push('Repository must be in owner/repo format')
  }
  return errors
})

// Ticket form validation
const ticketPattern = /^[A-Z]+-\d+$/
const agentPattern = /^[a-z0-9_-]*$/

const ticketFormValid = computed(() => {
  return ticketPattern.test(ticketForm.value.ticket_id) &&
    agentPattern.test(ticketForm.value.agent)
})

const ticketFormErrors = computed(() => {
  const errors: string[] = []
  if (ticketForm.value.ticket_id && !ticketPattern.test(ticketForm.value.ticket_id)) {
    errors.push('Ticket must be in PROJ-123 format')
  }
  if (ticketForm.value.agent && !agentPattern.test(ticketForm.value.agent)) {
    errors.push('Agent must be lowercase alphanumeric with dashes/underscores')
  }
  return errors
})

// Keyboard shortcuts
useKeyboard({
  onStart: handleStart,
  onStop: () => showConfirmDialog('Stop All Agents', 'Are you sure you want to stop all agents?', handleStop, true),
  onPause: handlePause,
  onWork: () => { activeModal.value = 'work' },
  onAnalyze: () => { activeModal.value = 'analyze' },
  onTicket: () => { activeModal.value = 'ticket' },
  onRetry: handleRetryAll,
  onEscape: () => {
    activeModal.value = null
    confirmDialog.value.open = false
  },
})

// Helper to show toast
function showToast(message: string, type: 'success' | 'error' | 'info') {
  toast.value = { show: true, message, type }
}

// Helper to show confirm dialog
function showConfirmDialog(title: string, message: string, action: () => Promise<void>, destructive = false) {
  confirmDialog.value = { open: true, title, message, action, destructive }
}

// Handlers
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
    }
  } else {
    const result = await pauseSystem()
    if (result?.success) {
      showToast('System paused', 'info')
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
    if (result?.success) {
      succeeded++
    } else {
      failed++
    }
  }

  if (failed === 0) {
    showToast(`Retrying ${succeeded} failed task(s)`, 'info')
  } else {
    showToast(`Retried ${succeeded}, failed to retry ${failed} task(s)`, 'error')
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

async function submitWork() {
  if (!workFormValid.value) return

  const result = await createWork({
    goal: workForm.value.goal,
    repository: workForm.value.repository,
    workflow: workForm.value.workflow,
  })

  if (result?.success) {
    activeModal.value = null
    workForm.value = { goal: '', repository: '', workflow: 'default' }
    showToast(result.message, 'success')
  } else if (apiError.value) {
    showToast(apiError.value, 'error')
  }
}

async function submitAnalyze() {
  if (!analyzeFormValid.value) return

  const result = await analyzeRepo({
    repository: analyzeForm.value.repository,
    severity: analyzeForm.value.severity,
    max_issues: analyzeForm.value.max_issues,
    dry_run: analyzeForm.value.dry_run,
    focus: analyzeForm.value.focus || undefined,
  })

  if (result?.success) {
    activeModal.value = null
    analyzeForm.value = { repository: '', severity: 'high', max_issues: 50, dry_run: false, focus: '' }
    showToast(result.message, 'success')
  } else if (apiError.value) {
    showToast(apiError.value, 'error')
  }
}

async function submitTicket() {
  if (!ticketFormValid.value) return

  const result = await runTicket({
    ticket_id: ticketForm.value.ticket_id,
    agent: ticketForm.value.agent || undefined,
  })

  if (result?.success) {
    activeModal.value = null
    ticketForm.value = { ticket_id: '', agent: '' }
    showToast(result.message, 'success')
  } else if (apiError.value) {
    showToast(apiError.value, 'error')
  }
}

async function handleConfirmAction() {
  confirmDialog.value.open = false
  await confirmDialog.value.action()
}

// Setup-related handlers
async function checkSetupStatus() {
  try {
    const response = await fetch('/api/setup/status')
    const status = await response.json()

    setupComplete.value = status.ready_to_start

    // Show setup prompt if not configured and not dismissed
    if (!setupComplete.value && !localStorage.getItem('dismissed_setup')) {
      showSetupPrompt.value = true
    }
  } catch (e) {
    console.error('Failed to check setup status:', e)
  }
}

function handleSetupComplete() {
  activeModal.value = null
  setupComplete.value = true
  showSetupPrompt.value = false
  showToast('Configuration saved. Click "Start All" to begin.', 'success')
}

function dismissSetupPrompt() {
  showSetupPrompt.value = false
  localStorage.setItem('dismissed_setup', 'true')
}

// Check setup status on mount
onMounted(() => {
  checkSetupStatus()
})
</script>

<template>
  <div class="h-screen flex flex-col bg-black text-gray-300 overflow-hidden">
    <!-- Setup prompt banner -->
    <div v-if="showSetupPrompt" class="bg-yellow-500/20 border-b border-yellow-500 px-4 py-3 shrink-0">
      <div class="flex items-center justify-between gap-3 flex-wrap">
        <div class="flex items-center gap-3">
          <span class="text-2xl">⚠️</span>
          <div>
            <p class="text-yellow-200 font-medium text-sm">System not configured</p>
            <p class="text-yellow-300/70 text-xs">Complete setup to start using agent framework</p>
          </div>
        </div>
        <div class="flex gap-2">
          <button
            @click="activeModal = 'setup'; showSetupPrompt = false"
            class="px-4 py-2 bg-yellow-600 hover:bg-yellow-500 text-white font-medium text-sm rounded"
          >
            Start Setup
          </button>
          <button
            @click="dismissSetupPrompt"
            class="px-3 py-2 text-yellow-300 hover:text-yellow-100 text-sm"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>

    <!-- Header -->
    <header class="flex flex-wrap items-center justify-between px-4 py-2 bg-gray-900 border-b border-gray-800 shrink-0 gap-2">
      <div class="flex items-center gap-3 font-mono text-sm">
        <span class="text-gray-200 font-medium">Agent Dashboard</span>
        <span v-if="isPaused" class="px-2 py-0.5 bg-yellow-500/20 text-yellow-400 text-xs rounded">PAUSED</span>
      </div>

      <!-- Action Toolbar -->
      <div class="flex flex-wrap items-center gap-2">
        <button
          @click="handlePause"
          class="px-3 py-1.5 text-sm font-mono rounded transition-colors"
          :class="isPaused
            ? 'bg-green-600 hover:bg-green-500 text-white'
            : 'bg-yellow-600 hover:bg-yellow-500 text-white'"
          :aria-label="isPaused ? 'Resume system' : 'Pause system'"
        >
          {{ isPaused ? 'Resume' : 'Pause' }}
        </button>
        <button
          @click="handleStart"
          :disabled="loading"
          class="px-3 py-1.5 text-sm font-mono bg-cyan-600 hover:bg-cyan-500 text-white rounded transition-colors disabled:opacity-50"
          aria-label="Start all agents"
        >
          Start All
        </button>
        <button
          @click="showConfirmDialog('Stop All Agents', 'Are you sure you want to stop all agents? This will terminate all running tasks.', handleStop, true)"
          :disabled="loading"
          class="px-3 py-1.5 text-sm font-mono bg-red-600 hover:bg-red-500 text-white rounded transition-colors disabled:opacity-50"
          aria-label="Stop all agents"
        >
          Stop All
        </button>
        <div class="w-px h-6 bg-gray-700 mx-1 hidden sm:block"></div>
        <button
          @click="activeModal = 'work'"
          class="px-3 py-1.5 text-sm font-mono bg-gray-700 hover:bg-gray-600 text-gray-200 rounded transition-colors"
          aria-label="Create new work"
        >
          + New Work
        </button>
        <button
          @click="activeModal = 'analyze'"
          class="px-3 py-1.5 text-sm font-mono bg-gray-700 hover:bg-gray-600 text-gray-200 rounded transition-colors"
          aria-label="Analyze repository"
        >
          Analyze Repo
        </button>
        <button
          @click="activeModal = 'ticket'"
          class="px-3 py-1.5 text-sm font-mono bg-gray-700 hover:bg-gray-600 text-gray-200 rounded transition-colors"
          aria-label="Run JIRA ticket"
        >
          Run Ticket
        </button>
        <div class="w-px h-6 bg-gray-700 mx-1 hidden sm:block"></div>
        <button
          v-if="!setupComplete"
          @click="activeModal = 'setup'"
          class="px-3 py-1.5 text-sm font-mono bg-green-600 hover:bg-green-500 text-white rounded transition-colors"
          aria-label="Setup wizard"
        >
          ⚙️ Setup
        </button>
      </div>
    </header>

    <!-- Connection error -->
    <div v-if="wsError && !reconnecting" class="flex flex-col items-center justify-center flex-1 gap-4">
      <div class="text-red-400 font-mono text-sm">{{ wsError }}</div>
      <button
        @click="reconnect"
        class="px-4 py-2 bg-gray-800 hover:bg-gray-700 font-mono text-sm rounded transition-colors"
      >
        Reconnect
      </button>
    </div>

    <!-- Reconnecting state -->
    <div v-else-if="reconnecting" class="flex items-center justify-center flex-1">
      <div class="text-yellow-400 font-mono text-sm animate-pulse">
        Reconnecting ({{ reconnectAttempt }}/10)...
      </div>
    </div>

    <!-- Loading state -->
    <div v-else-if="!state" class="flex items-center justify-center flex-1">
      <div class="text-gray-500 font-mono text-sm">Connecting...</div>
    </div>

    <!-- Main content -->
    <template v-else>
      <div class="flex-1 min-h-0 overflow-y-auto">
        <!-- Agent Cards -->
        <div class="p-4">
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            <AgentCard
              v-for="agent in agents"
              :key="agent.id"
              :agent="agent"
              :on-restart="handleRestart"
            />
          </div>

          <!-- Empty state -->
          <div v-if="agents.length === 0" class="text-center py-8 text-gray-500 font-mono">
            No agents configured
          </div>
        </div>

        <!-- Status Bar -->
        <div class="px-4 py-2 bg-gray-900/50 border-y border-gray-800 font-mono text-xs flex flex-wrap items-center justify-between gap-2">
          <div class="flex items-center gap-4 text-gray-500">
            <span>Queues: <span class="text-gray-400">{{ queueSummary || 'none' }}</span></span>
          </div>
          <div class="flex items-center gap-4 text-gray-500">
            <span>
              Health:
              <span :class="health.passed ? 'text-green-400' : 'text-yellow-400'">
                {{ health.passed ? 'OK' : 'WARN' }} ({{ health.checks.filter(c => c.passed).length }}/{{ health.checks.length }})
              </span>
            </span>
            <span>Uptime: <span class="text-gray-400">{{ uptimeDisplay }}</span></span>
            <span class="flex items-center gap-1.5" :class="connected ? 'text-green-400' : 'text-red-400'">
              <span class="w-1.5 h-1.5 rounded-full" :class="connected ? 'bg-green-500' : 'bg-red-500'"></span>
              {{ connected ? 'Connected' : 'Disconnected' }}
            </span>
          </div>
        </div>

        <!-- Activity and Failed Tasks -->
        <div class="p-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
          <!-- Recent Activity -->
          <div class="bg-gray-900/50 border border-gray-800 rounded-lg overflow-hidden">
            <div class="px-4 py-2 border-b border-gray-800 text-sm font-medium text-gray-400">
              Recent Activity
            </div>
            <div class="max-h-48 overflow-y-auto">
              <ActivityFeed :events="events" />
            </div>
          </div>

          <!-- Failed Tasks -->
          <div class="bg-gray-900/50 border border-gray-800 rounded-lg overflow-hidden">
            <div class="px-4 py-2 border-b border-gray-800 text-sm font-medium text-gray-400 flex items-center justify-between">
              <span>Failed Tasks</span>
              <span v-if="failedTasks.length > 0" class="text-red-400">({{ failedTasks.length }})</span>
            </div>
            <div class="max-h-48 overflow-y-auto">
              <FailedTasks :tasks="failedTasks" :on-retry="handleRetryTask" />
            </div>
          </div>
        </div>
      </div>

      <!-- Collapsible Log Panel -->
      <div class="border-t border-gray-800 shrink-0">
        <button
          @click="logsExpanded = !logsExpanded"
          class="w-full px-4 py-2 flex items-center justify-between text-sm font-mono text-gray-400 hover:bg-gray-900/30 transition-colors"
          :aria-expanded="logsExpanded"
          aria-controls="log-panel"
        >
          <span class="flex items-center gap-2">
            <span :class="logsExpanded ? 'rotate-90' : ''" class="transition-transform">></span>
            Live Logs
            <span v-if="!logsConnected" class="text-yellow-500 text-xs">(disconnected)</span>
          </span>
          <span class="text-xs text-gray-600">{{ logs.length }} entries</span>
        </button>
        <div
          v-show="logsExpanded"
          id="log-panel"
          class="h-96 border-t border-gray-800"
        >
          <LogViewer :logs="logs" :agents="agentIds" />
        </div>
      </div>

      <!-- Footer with keyboard shortcuts -->
      <footer class="bg-gray-900 border-t border-gray-800 px-4 py-1.5 font-mono text-xs text-gray-600 flex items-center gap-4 shrink-0">
        <span>[s] start</span>
        <span>[x] stop</span>
        <span>[p] pause</span>
        <span>[w] work</span>
        <span>[a] analyze</span>
        <span>[t] ticket</span>
        <span>[r] retry</span>
        <span class="flex-1"></span>
        <span v-if="loading" class="text-cyan-400">Loading...</span>
      </footer>
    </template>

    <!-- Work Modal -->
    <Modal :open="activeModal === 'work'" title="New Work" @close="activeModal = null">
      <form @submit.prevent="submitWork" class="space-y-4">
        <div>
          <label class="block text-xs text-gray-500 mb-1" for="work-goal">Goal (min 10 characters)</label>
          <textarea
            id="work-goal"
            v-model="workForm.goal"
            rows="3"
            class="w-full bg-black border px-2 py-1 text-sm focus:outline-none rounded"
            :class="workForm.goal && workForm.goal.length < 10 ? 'border-red-500' : 'border-gray-700 focus:border-cyan-500'"
            placeholder="Describe what you want to build..."
          ></textarea>
          <span v-if="workForm.goal && workForm.goal.length < 10" class="text-xs text-red-400">
            {{ workForm.goal.length }}/10 characters minimum
          </span>
        </div>
        <div>
          <label class="block text-xs text-gray-500 mb-1" for="work-repo">Repository (owner/repo)</label>
          <input
            id="work-repo"
            v-model="workForm.repository"
            type="text"
            class="w-full bg-black border px-2 py-1 text-sm focus:outline-none rounded"
            :class="workForm.repository && !repoPattern.test(workForm.repository) ? 'border-red-500' : 'border-gray-700 focus:border-cyan-500'"
            placeholder="justworkshr/pto"
          />
          <span v-if="workForm.repository && !repoPattern.test(workForm.repository)" class="text-xs text-red-400">
            Invalid format. Use owner/repo (e.g., justworkshr/pto)
          </span>
        </div>
        <div class="flex justify-end gap-2 pt-2">
          <button
            type="button"
            @click="activeModal = null"
            class="px-3 py-1.5 text-sm text-gray-500 hover:text-gray-300 rounded"
          >
            Cancel
          </button>
          <button
            type="submit"
            :disabled="!workFormValid || loading"
            class="px-4 py-1.5 text-sm bg-cyan-600 hover:bg-cyan-500 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Create
          </button>
        </div>
      </form>
    </Modal>

    <!-- Analyze Modal -->
    <Modal :open="activeModal === 'analyze'" title="Analyze Repository" @close="activeModal = null">
      <form @submit.prevent="submitAnalyze" class="space-y-4">
        <div>
          <label class="block text-xs text-gray-500 mb-1" for="analyze-repo">Repository (owner/repo)</label>
          <input
            id="analyze-repo"
            v-model="analyzeForm.repository"
            type="text"
            class="w-full bg-black border px-2 py-1 text-sm focus:outline-none rounded"
            :class="analyzeForm.repository && !repoPattern.test(analyzeForm.repository) ? 'border-red-500' : 'border-gray-700 focus:border-cyan-500'"
            placeholder="justworkshr/pto"
          />
          <span v-if="analyzeForm.repository && !repoPattern.test(analyzeForm.repository)" class="text-xs text-red-400">
            Invalid format. Use owner/repo (e.g., justworkshr/pto)
          </span>
        </div>
        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="block text-xs text-gray-500 mb-1" for="analyze-severity">Severity</label>
            <select
              id="analyze-severity"
              v-model="analyzeForm.severity"
              class="w-full bg-black border border-gray-700 px-2 py-1 text-sm focus:outline-none focus:border-cyan-500 rounded"
            >
              <option value="all">all</option>
              <option value="critical">critical</option>
              <option value="high">high</option>
              <option value="medium">medium</option>
            </select>
          </div>
          <div>
            <label class="block text-xs text-gray-500 mb-1" for="analyze-max">Max Issues</label>
            <input
              id="analyze-max"
              v-model.number="analyzeForm.max_issues"
              type="number"
              min="1"
              max="500"
              class="w-full bg-black border border-gray-700 px-2 py-1 text-sm focus:outline-none focus:border-cyan-500 rounded"
            />
          </div>
        </div>
        <div>
          <label class="block text-xs text-gray-500 mb-1" for="analyze-focus">Focus (optional)</label>
          <textarea
            id="analyze-focus"
            v-model="analyzeForm.focus"
            rows="2"
            class="w-full bg-black border border-gray-700 px-2 py-1 text-sm focus:outline-none focus:border-cyan-500 rounded"
            placeholder="e.g., review PTO accrual flow for tech debt"
          ></textarea>
        </div>
        <div class="flex items-center gap-2">
          <input
            v-model="analyzeForm.dry_run"
            type="checkbox"
            id="dry-run"
            class="bg-black border border-gray-700 rounded"
          />
          <label for="dry-run" class="text-xs text-gray-500">Dry run (no JIRA tickets)</label>
        </div>
        <div class="flex justify-end gap-2 pt-2">
          <button
            type="button"
            @click="activeModal = null"
            class="px-3 py-1.5 text-sm text-gray-500 hover:text-gray-300 rounded"
          >
            Cancel
          </button>
          <button
            type="submit"
            :disabled="!analyzeFormValid || loading"
            class="px-4 py-1.5 text-sm bg-cyan-600 hover:bg-cyan-500 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Analyze
          </button>
        </div>
      </form>
    </Modal>

    <!-- Setup Wizard Modal -->
    <Modal :open="activeModal === 'setup'" title="Setup Wizard" @close="activeModal = null">
      <SetupWizard @complete="handleSetupComplete" />
    </Modal>

    <!-- Run Ticket Modal -->
    <Modal :open="activeModal === 'ticket'" title="Run JIRA Ticket" @close="activeModal = null">
      <form @submit.prevent="submitTicket" class="space-y-4">
        <div>
          <label class="block text-xs text-gray-500 mb-1" for="ticket-id">Ticket ID (e.g., PROJ-123)</label>
          <input
            id="ticket-id"
            v-model="ticketForm.ticket_id"
            type="text"
            class="w-full bg-black border px-2 py-1 text-sm focus:outline-none rounded uppercase"
            :class="ticketForm.ticket_id && !ticketPattern.test(ticketForm.ticket_id) ? 'border-red-500' : 'border-gray-700 focus:border-cyan-500'"
            placeholder="PROJ-123"
          />
          <span v-if="ticketForm.ticket_id && !ticketPattern.test(ticketForm.ticket_id)" class="text-xs text-red-400">
            Invalid format. Use PROJ-123 format
          </span>
        </div>
        <div>
          <label class="block text-xs text-gray-500 mb-1" for="ticket-agent">Agent (optional)</label>
          <select
            id="ticket-agent"
            v-model="ticketForm.agent"
            class="w-full bg-black border border-gray-700 px-2 py-1 text-sm focus:outline-none focus:border-cyan-500 rounded"
          >
            <option value="">Auto-assign based on ticket type</option>
            <option value="architect">architect</option>
            <option value="engineer">engineer</option>
            <option value="qa">qa</option>
          </select>
          <span class="text-xs text-gray-600">Leave empty to auto-assign based on JIRA ticket type</span>
        </div>
        <div class="flex justify-end gap-2 pt-2">
          <button
            type="button"
            @click="activeModal = null"
            class="px-3 py-1.5 text-sm text-gray-500 hover:text-gray-300 rounded"
          >
            Cancel
          </button>
          <button
            type="submit"
            :disabled="!ticketFormValid || loading"
            class="px-4 py-1.5 text-sm bg-cyan-600 hover:bg-cyan-500 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Run
          </button>
        </div>
      </form>
    </Modal>

    <!-- Confirm Dialog -->
    <ConfirmDialog
      :open="confirmDialog.open"
      :title="confirmDialog.title"
      :message="confirmDialog.message"
      :destructive="confirmDialog.destructive"
      confirm-label="Confirm"
      cancel-label="Cancel"
      @confirm="handleConfirmAction"
      @cancel="confirmDialog.open = false"
    />

    <!-- Toast Notifications -->
    <Toast
      v-if="toast.show"
      :message="toast.message"
      :type="toast.type"
      @dismiss="toast.show = false"
    />
  </div>
</template>

<style>
/* Global styles */
* {
  box-sizing: border-box;
}

html, body, #app {
  margin: 0;
  padding: 0;
  height: 100%;
  overflow: hidden;
}

body {
  font-family: ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Monaco, 'Liberation Mono', 'Courier New', monospace;
  background: #000;
}

/* Scrollbar styling */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: #000;
}

::-webkit-scrollbar-thumb {
  background: #374151;
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: #4b5563;
}

/* Form elements */
input, textarea, select {
  font-family: inherit;
}

input:focus, textarea:focus, select:focus {
  outline: none;
}

/* Screen reader only */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
</style>
