<script setup lang="ts">
import { computed } from 'vue'
import type { QueueStats, HealthReport } from '../types'

const props = defineProps<{
  isPaused: boolean
  connected: boolean
  queues: QueueStats[]
  health: HealthReport
  uptime: number
}>()

const emit = defineEmits<{
  start: []
  stop: []
  pause: []
  work: []
  analyze: []
  retry: []
}>()

// Format uptime as "5m 32s"
const uptimeDisplay = computed(() => {
  const minutes = Math.floor(props.uptime / 60)
  const seconds = props.uptime % 60
  return `${minutes}m ${seconds}s`
})

// Queue summary "engineer(3) qa(1) architect(0)"
const queueSummary = computed(() => {
  return props.queues
    .map(q => `${q.queue_id}(${q.pending_count})`)
    .join(' ')
})

// Health summary "OK (8/8)" or "WARN (6/8)"
const healthSummary = computed(() => {
  const total = props.health.checks.length
  const passed = props.health.checks.filter(c => c.passed).length
  const status = props.health.passed ? 'OK' : 'WARN'
  return `${status} (${passed}/${total})`
})

const healthClass = computed(() => {
  return props.health.passed ? 'text-green-400' : 'text-yellow-400'
})

const shortcuts = [
  { key: 's', label: 'start', action: () => emit('start') },
  { key: 'x', label: 'stop', action: () => emit('stop') },
  { key: 'p', label: 'pause', action: () => emit('pause') },
  { key: 'w', label: 'work', action: () => emit('work') },
  { key: 'a', label: 'analyze', action: () => emit('analyze') },
  { key: 'r', label: 'retry', action: () => emit('retry') },
]
</script>

<template>
  <div class="bg-gray-900 border-t border-gray-800 font-mono text-xs">
    <!-- Status bar -->
    <div class="flex items-center justify-between px-2 py-1 border-b border-gray-800/50 text-gray-500">
      <div class="flex items-center gap-4">
        <span>queues: {{ queueSummary }}</span>
      </div>
      <div class="flex items-center gap-4">
        <span>health: <span :class="healthClass">{{ healthSummary }}</span></span>
        <span>uptime: {{ uptimeDisplay }}</span>
        <span
          class="flex items-center gap-1"
          :class="connected ? 'text-green-400' : 'text-red-400'"
        >
          <span class="w-1.5 h-1.5 rounded-full" :class="connected ? 'bg-green-500' : 'bg-red-500'"></span>
          {{ connected ? 'connected' : 'disconnected' }}
        </span>
        <span v-if="isPaused" class="text-yellow-400 font-bold">PAUSED</span>
      </div>
    </div>

    <!-- Keyboard shortcuts -->
    <div class="flex items-center gap-1 px-2 py-1">
      <button
        v-for="shortcut in shortcuts"
        :key="shortcut.key"
        @click="shortcut.action"
        class="px-2 py-0.5 hover:bg-gray-800 rounded transition-colors"
      >
        <span class="text-gray-500">[</span>
        <span class="text-cyan-400">{{ shortcut.key }}</span>
        <span class="text-gray-500">]</span>
        <span class="text-gray-400 ml-1">{{ shortcut.label }}</span>
      </button>
    </div>
  </div>
</template>
