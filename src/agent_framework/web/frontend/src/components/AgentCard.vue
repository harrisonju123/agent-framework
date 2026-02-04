<script setup lang="ts">
import { computed, ref, onMounted, onUnmounted, watch } from 'vue'
import type { Agent } from '../types'

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

const statusClass = computed(() => {
  switch (props.agent.status) {
    case 'working':
      return 'border-green-500 bg-green-500/10'
    case 'completing':
      return 'border-blue-500 bg-blue-500/10'
    case 'idle':
      return 'border-yellow-500 bg-yellow-500/10'
    case 'dead':
      return 'border-red-500 bg-red-500/10'
    default:
      return 'border-gray-500'
  }
})

const statusDotClass = computed(() => {
  switch (props.agent.status) {
    case 'working':
      return 'bg-green-500'
    case 'completing':
      return 'bg-blue-500'
    case 'idle':
      return 'bg-yellow-500'
    case 'dead':
      return 'bg-red-500'
    default:
      return 'bg-gray-500'
  }
})

const statusLabel = computed(() => {
  switch (props.agent.status) {
    case 'working':
      return 'Working'
    case 'completing':
      return 'Completing'
    case 'idle':
      return 'Idle'
    case 'dead':
      return 'Dead'
    default:
      return 'Unknown'
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
    class="p-4 rounded-lg border-2 transition-all hover:shadow-lg"
    :class="statusClass"
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
        <span class="font-semibold text-lg">{{ agent.name }}</span>
      </div>
      <span
        class="px-2 py-1 text-xs font-medium rounded"
        :class="{
          'bg-green-500/20 text-green-400': agent.status === 'working',
          'bg-blue-500/20 text-blue-400': agent.status === 'completing',
          'bg-yellow-500/20 text-yellow-400': agent.status === 'idle',
          'bg-red-500/20 text-red-400': agent.status === 'dead',
        }"
        role="status"
      >
        {{ statusLabel }}
        <span class="sr-only">status</span>
      </span>
    </div>

    <div v-if="agent.status === 'working' && agent.current_task" class="mt-3">
      <p class="text-sm text-gray-300 truncate" :title="agent.current_task.title">
        {{ agent.current_task.title }}
      </p>
      <div class="flex items-center justify-between mt-2">
        <span v-if="phaseDisplay" class="text-xs text-cyan-400">
          {{ phaseDisplay }}
        </span>
        <span v-if="elapsedDisplay" class="text-xs text-gray-500">
          {{ elapsedDisplay }}
        </span>
      </div>
      <div class="flex gap-1 mt-2" role="progressbar" :aria-valuenow="progressDots.filled" :aria-valuemin="0" :aria-valuemax="5" :aria-label="`Progress: ${progressDots.filled} of 5 phases completed`">
        <span
          v-for="i in progressDots.filled"
          :key="'filled-' + i"
          class="w-2 h-2 rounded-full bg-cyan-500"
          aria-hidden="true"
        ></span>
        <span
          v-for="i in progressDots.empty"
          :key="'empty-' + i"
          class="w-2 h-2 rounded-full bg-gray-600"
          aria-hidden="true"
        ></span>
      </div>
    </div>

    <div v-else-if="agent.status === 'completing' && agent.current_task" class="mt-3">
      <p class="text-sm text-gray-300 truncate" :title="agent.current_task.title">
        ✓ {{ agent.current_task.title }}
      </p>
      <div class="flex items-center justify-between mt-2">
        <span class="text-xs text-blue-400">
          Task completed
        </span>
        <span v-if="elapsedDisplay" class="text-xs text-gray-500">
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
      <p class="text-sm text-gray-500">Waiting for tasks...</p>
    </div>

    <div v-else-if="agent.status === 'dead'" class="mt-3">
      <p class="text-sm text-red-400 mb-2">No heartbeat detected</p>
      <button
        @click="handleRestart"
        class="px-3 py-1 text-sm bg-red-600 hover:bg-red-700 rounded transition-colors"
        :aria-label="`Restart ${agent.name} agent`"
      >
        Restart
      </button>
    </div>
  </div>
</template>
