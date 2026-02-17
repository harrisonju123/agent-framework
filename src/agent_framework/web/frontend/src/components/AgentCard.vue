<script setup lang="ts">
import { computed, ref, onMounted, onUnmounted, watch } from 'vue'
import type { Agent } from '../types'
import Button from 'primevue/button'

const props = defineProps<{
  agent: Agent
  onRestart: (agentId: string) => void
}>()

// Client-side elapsed time calculation for smooth updates
const clientElapsed = ref<number | null>(null)
let intervalId: ReturnType<typeof setInterval> | null = null

function updateElapsed() {
  if ((props.agent.status === 'working' || props.agent.status === 'completing') && props.agent.current_task?.started_at) {
    const started = new Date(props.agent.current_task.started_at).getTime()
    const now = Date.now()
    clientElapsed.value = Math.floor((now - started) / 1000)
  } else {
    clientElapsed.value = null
  }
}

function startTimer() {
  stopTimer()
  updateElapsed()
  intervalId = setInterval(updateElapsed, 1000)
}

function stopTimer() {
  if (intervalId) {
    clearInterval(intervalId)
    intervalId = null
  }
}

watch(() => props.agent.status, (status) => {
  if (status === 'working' || status === 'completing') {
    startTimer()
  } else {
    stopTimer()
    clientElapsed.value = null
  }
}, { immediate: true })

onMounted(() => {
  if (props.agent.status === 'working' || props.agent.status === 'completing') {
    startTimer()
  }
})

onUnmounted(() => {
  stopTimer()
})

const borderClass = computed(() => {
  switch (props.agent.status) {
    case 'working': return 'border-l-emerald-500'
    case 'completing': return 'border-l-blue-500'
    case 'idle': return 'border-l-amber-400'
    case 'dead': return 'border-l-red-500'
    default: return 'border-l-slate-300'
  }
})

const statusDotClass = computed(() => {
  switch (props.agent.status) {
    case 'working': return 'bg-emerald-500'
    case 'completing': return 'bg-blue-500'
    case 'idle': return 'bg-amber-400'
    case 'dead': return 'bg-red-500'
    default: return 'bg-slate-400'
  }
})

const statusLabel = computed(() => {
  switch (props.agent.status) {
    case 'working': return 'Working'
    case 'completing': return 'Completing'
    case 'idle': return 'Idle'
    case 'dead': return 'Dead'
    default: return 'Unknown'
  }
})

const statusBadgeClass = computed(() => {
  switch (props.agent.status) {
    case 'working': return 'bg-emerald-50 text-emerald-700'
    case 'completing': return 'bg-blue-50 text-blue-700'
    case 'idle': return 'bg-amber-50 text-amber-700'
    case 'dead': return 'bg-red-50 text-red-700'
    default: return 'bg-slate-100 text-slate-600'
  }
})

const phaseDisplay = computed(() => {
  if (!props.agent.current_phase) return null
  return props.agent.current_phase.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
})

const elapsedDisplay = computed(() => {
  const elapsed = clientElapsed.value ?? props.agent.elapsed_seconds
  if (!elapsed) return null
  const minutes = Math.floor(elapsed / 60)
  const seconds = elapsed % 60
  return `${minutes}m ${seconds}s`
})

const TOOL_VERBS: Record<string, string> = {
  Read: 'Reading',
  Edit: 'Editing',
  Write: 'Writing',
  Bash: 'Running',
  Grep: 'Searching',
  Glob: 'Finding',
  Task: 'Delegating',
}

const toolActivityDisplay = computed(() => {
  const ta = props.agent.tool_activity
  if (!ta) return null
  const verb = TOOL_VERBS[ta.tool_name] || ta.tool_name
  const summary = ta.tool_input_summary ? `: ${ta.tool_input_summary}` : ''
  return { text: `${verb}${summary}`, count: ta.tool_call_count }
})

const progressDots = computed(() => {
  const total = 5
  const completed = Math.min(props.agent.phases_completed, total)
  return {
    filled: completed,
    empty: total - completed,
  }
})

function handleRestart() {
  props.onRestart(props.agent.id)
}
</script>

<template>
  <div
    class="bg-white shadow-sm rounded-xl border border-slate-200 border-l-4 p-4 transition-all hover:shadow-md"
    :class="borderClass"
    role="article"
    :aria-label="`Agent ${agent.name}, status: ${statusLabel}`"
  >
    <div class="flex items-center justify-between">
      <div class="flex items-center gap-3">
        <span
          class="w-3 h-3 rounded-full"
          :class="[statusDotClass, agent.status === 'working' ? 'animate-pulse' : '']"
          aria-hidden="true"
        ></span>
        <span class="font-semibold text-lg text-slate-800">{{ agent.name }}</span>
      </div>
      <span
        class="px-2 py-1 text-xs font-medium rounded-md"
        :class="statusBadgeClass"
        role="status"
      >
        {{ statusLabel }}
        <span class="sr-only">status</span>
      </span>
    </div>

    <div v-if="agent.status === 'working' && agent.current_task" class="mt-3">
      <p class="text-sm text-slate-700 truncate" :title="agent.current_task.title">
        {{ agent.current_task.title }}
      </p>
      <div class="flex items-center justify-between mt-2">
        <span v-if="phaseDisplay" class="text-xs text-blue-600 font-medium">
          {{ phaseDisplay }}
        </span>
        <span v-if="elapsedDisplay" class="text-xs text-slate-400">
          {{ elapsedDisplay }}
        </span>
      </div>
      <div v-if="toolActivityDisplay" class="flex items-center gap-2 mt-1">
        <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" aria-hidden="true"></span>
        <span class="text-xs text-emerald-600 truncate" :title="toolActivityDisplay.text">
          {{ toolActivityDisplay.text }}
        </span>
        <span class="text-xs text-slate-400 ml-auto whitespace-nowrap">
          {{ toolActivityDisplay.count }} tools
        </span>
      </div>
      <div class="flex gap-1 mt-2" role="progressbar" :aria-valuenow="progressDots.filled" :aria-valuemin="0" :aria-valuemax="5" :aria-label="`Progress: ${progressDots.filled} of 5 phases completed`">
        <span
          v-for="i in progressDots.filled"
          :key="'filled-' + i"
          class="w-2 h-2 rounded-full bg-blue-500"
          aria-hidden="true"
        ></span>
        <span
          v-for="i in progressDots.empty"
          :key="'empty-' + i"
          class="w-2 h-2 rounded-full bg-slate-200"
          aria-hidden="true"
        ></span>
      </div>
    </div>

    <div v-else-if="agent.status === 'completing' && agent.current_task" class="mt-3">
      <p class="text-sm text-slate-700 truncate" :title="agent.current_task.title">
        {{ agent.current_task.title }}
      </p>
      <div class="flex items-center justify-between mt-2">
        <span class="text-xs text-blue-600 font-medium">
          Task completed
        </span>
        <span v-if="elapsedDisplay" class="text-xs text-slate-400">
          {{ elapsedDisplay }}
        </span>
      </div>
      <div class="flex gap-1 mt-2" role="progressbar" aria-valuenow="5" aria-valuemin="0" aria-valuemax="5" aria-label="All phases completed">
        <span
          v-for="i in 5"
          :key="'complete-' + i"
          class="w-2 h-2 rounded-full bg-blue-500"
          aria-hidden="true"
        ></span>
      </div>
    </div>

    <div v-else-if="agent.status === 'idle'" class="mt-3">
      <p class="text-sm text-slate-400">Waiting for tasks...</p>
    </div>

    <div v-else-if="agent.status === 'dead'" class="mt-3">
      <p class="text-sm text-red-600 mb-2">No heartbeat detected</p>
      <Button
        label="Restart"
        severity="danger"
        size="small"
        @click="handleRestart"
        :aria-label="`Restart ${agent.name} agent`"
      />
    </div>
  </div>
</template>
