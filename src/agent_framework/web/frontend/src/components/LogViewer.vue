<script setup lang="ts">
import { ref, computed, watch, onUnmounted } from 'vue'
import type { LogEntry } from '../types'

const props = defineProps<{
  logs: LogEntry[]
  agents: string[]
}>()

const filter = ref<string>('all')
const sourceFilter = ref<string>('all')
const logEl = ref<HTMLElement | null>(null)
const autoScroll = ref(true)

// Track how many logs existed when user paused scrolling
const pausedAtCount = ref(0)
let scrollRAFId = 0

const filteredLogs = computed(() => {
  let logs = props.logs

  if (sourceFilter.value !== 'all') {
    logs = logs.filter(log => (log.source || 'agent') === sourceFilter.value)
  }

  if (filter.value !== 'all') {
    logs = logs.filter(log => log.agent === filter.value)
  }

  return logs
})

// Only render the tail — keeps DOM node count bounded
const visibleLogs = computed(() => filteredLogs.value.slice(-100))

const newLogsSincePause = computed(() => {
  if (autoScroll.value) return 0
  return Math.max(0, filteredLogs.value.length - pausedAtCount.value)
})

const agentColors: Record<string, string> = {
  'architect': 'text-purple-400',
  'engineer': 'text-sky-400',
  'qa': 'text-green-400',
  'watchdog': 'text-gray-400',
}

function getAgentColor(agent: string): string {
  return agentColors[agent] || 'text-blue-400'
}

function formatTime(timestamp: string): string {
  try {
    const date = new Date(timestamp)
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  } catch {
    return '--:--:--'
  }
}

function getLevelClass(level: string | null): string {
  switch (level?.toUpperCase()) {
    case 'ERROR':
    case 'CRITICAL':
      return 'text-red-400'
    case 'WARNING':
      return 'text-yellow-400'
    case 'DEBUG':
      return 'text-gray-500'
    default:
      return 'text-gray-300'
  }
}

// Coalesce scroll updates to one per frame instead of one per log message
function scheduleScroll() {
  if (scrollRAFId) return
  scrollRAFId = requestAnimationFrame(() => {
    scrollRAFId = 0
    if (autoScroll.value && logEl.value) {
      logEl.value.scrollTop = logEl.value.scrollHeight
    }
  })
}

watch(
  () => filteredLogs.value.length,
  () => {
    if (autoScroll.value) {
      scheduleScroll()
    }
  }
)

function handleScroll() {
  if (!logEl.value) return
  const { scrollTop, scrollHeight, clientHeight } = logEl.value
  const wasAutoScroll = autoScroll.value
  autoScroll.value = scrollHeight - scrollTop - clientHeight < 50

  // Capture count when user first scrolls away from bottom
  if (wasAutoScroll && !autoScroll.value) {
    pausedAtCount.value = filteredLogs.value.length
  }
}

function scrollToBottom() {
  autoScroll.value = true
  pausedAtCount.value = 0
  if (logEl.value) {
    logEl.value.scrollTop = logEl.value.scrollHeight
  }
}

const uniqueAgents = computed(() => {
  const agents = new Set(props.logs.map(log => log.agent))
  return Array.from(agents).sort()
})

onUnmounted(() => {
  if (scrollRAFId) cancelAnimationFrame(scrollRAFId)
})
</script>

<template>
  <div class="bg-slate-900 font-mono text-sm flex flex-col h-full">
    <!-- Filter bar (light themed) -->
    <div class="flex gap-4 px-3 py-2 text-xs bg-slate-100 border-b border-slate-200 shrink-0">
      <!-- Source filter -->
      <span
        @click="sourceFilter = 'all'"
        class="cursor-pointer transition-colors"
        :class="sourceFilter === 'all' ? 'text-blue-600 font-medium' : 'text-slate-400 hover:text-slate-700'"
      >
        all
      </span>
      <span
        @click="sourceFilter = 'agent'"
        class="cursor-pointer transition-colors"
        :class="sourceFilter === 'agent' ? 'text-blue-600 font-medium' : 'text-slate-400 hover:text-slate-700'"
      >
        agent
      </span>
      <span
        @click="sourceFilter = 'claude-cli'"
        class="cursor-pointer transition-colors"
        :class="sourceFilter === 'claude-cli' ? 'text-blue-600 font-medium' : 'text-slate-400 hover:text-slate-700'"
      >
        cli
      </span>
      <span class="text-slate-300">|</span>
      <!-- Agent filter -->
      <span
        @click="filter = 'all'"
        class="cursor-pointer transition-colors"
        :class="filter === 'all' ? 'text-blue-600 font-medium' : 'text-slate-400 hover:text-slate-700'"
      >
        all agents
      </span>
      <span
        v-for="agent in uniqueAgents"
        :key="agent"
        @click="filter = agent"
        class="cursor-pointer transition-colors"
        :class="filter === agent ? 'text-blue-600 font-medium' : 'text-slate-400 hover:text-slate-700'"
      >
        {{ agent }}
      </span>
    </div>

    <!-- Log output (dark) -->
    <div class="flex-1 overflow-y-auto px-2 py-1 relative" ref="logEl" @scroll="handleScroll">
      <div
        v-for="log in visibleLogs"
        :key="log.id ?? log.timestamp + log.agent"
        class="leading-tight whitespace-pre-wrap break-all"
        :class="{ 'italic': log.source === 'claude-cli' }"
      >
        <span class="text-gray-600">{{ formatTime(log.timestamp) }}</span>
        <span class="mx-1" :class="getAgentColor(log.agent)">[{{ log.agent.slice(0, 12) }}]</span>
        <span v-if="log.source === 'claude-cli'" class="text-indigo-400 mr-1">[CLI]</span>
        <span :class="log.source === 'claude-cli' ? 'text-indigo-300' : getLevelClass(log.level)">{{ log.line }}</span>
      </div>

      <!-- Cursor / empty state -->
      <div v-if="filteredLogs.length === 0" class="text-gray-600 py-2">
        Waiting for logs...
      </div>
      <div
        v-else
        @click="scrollToBottom"
        class="text-blue-400 cursor-pointer"
        :class="{ 'animate-pulse': autoScroll }"
      >
        _
      </div>

      <!-- Floating jump-to-latest badge -->
      <div v-if="!autoScroll" class="jump-badge-anchor">
        <div @click="scrollToBottom" class="jump-badge">
          Jump to latest<span v-if="newLogsSincePause > 0"> ({{ newLogsSincePause }} new)</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.overflow-y-auto {
  scrollbar-width: thin;
  scrollbar-color: #374151 #0f172a;
}

.overflow-y-auto::-webkit-scrollbar {
  width: 6px;
}

.overflow-y-auto::-webkit-scrollbar-track {
  background: #0f172a;
}

.overflow-y-auto::-webkit-scrollbar-thumb {
  background: #374151;
  border-radius: 3px;
}

.jump-badge-anchor {
  position: sticky;
  bottom: 8px;
  display: flex;
  justify-content: flex-end;
  pointer-events: none;
  z-index: 10;
}

.jump-badge {
  pointer-events: auto;
  padding: 4px 12px;
  font-size: 0.75rem;
  line-height: 1rem;
  color: #fff;
  background: #3b82f6;
  border-radius: 9999px;
  cursor: pointer;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
  transition: background 0.15s;
}

.jump-badge:hover {
  background: #2563eb;
}
</style>
