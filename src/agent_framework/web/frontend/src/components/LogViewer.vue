<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted } from 'vue'
import type { LogEntry } from '../types'

const props = defineProps<{
  logs: LogEntry[]
  agents: string[]
}>()

const filter = ref<string>('all')
const sourceFilter = ref<string>('all')  // 'all', 'agent', 'claude-cli'
const logEl = ref<HTMLElement | null>(null)
const autoScroll = ref(true)

const filteredLogs = computed(() => {
  let logs = props.logs

  // Filter by source type
  if (sourceFilter.value !== 'all') {
    logs = logs.filter(log => (log.source || 'agent') === sourceFilter.value)
  }

  // Filter by agent
  if (filter.value !== 'all') {
    logs = logs.filter(log => log.agent === filter.value)
  }

  return logs
})

// Agent colors for visual distinction
const agentColors: Record<string, string> = {
  'engineer': 'text-sky-400',
  'qa': 'text-green-400',
  'architect': 'text-purple-400',
  'product-owner': 'text-yellow-400',
  'repo-analyzer': 'text-orange-400',
  'code-reviewer': 'text-pink-400',
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

// Auto-scroll to bottom when new logs arrive
watch(
  () => props.logs.length,
  async () => {
    if (autoScroll.value && logEl.value) {
      await nextTick()
      logEl.value.scrollTop = logEl.value.scrollHeight
    }
  }
)

// Detect manual scroll to disable auto-scroll
function handleScroll() {
  if (!logEl.value) return
  const { scrollTop, scrollHeight, clientHeight } = logEl.value
  // If user scrolled up more than 50px from bottom, disable auto-scroll
  autoScroll.value = scrollHeight - scrollTop - clientHeight < 50
}

// Re-enable auto-scroll when clicking the cursor
function scrollToBottom() {
  autoScroll.value = true
  if (logEl.value) {
    logEl.value.scrollTop = logEl.value.scrollHeight
  }
}

// Get unique agents from logs
const uniqueAgents = computed(() => {
  const agents = new Set(props.logs.map(log => log.agent))
  return Array.from(agents).sort()
})
</script>

<template>
  <div class="bg-black font-mono text-sm flex flex-col h-full">
    <!-- Filter bar -->
    <div class="flex gap-4 px-2 py-1 text-xs border-b border-gray-800 shrink-0">
      <!-- Source filter -->
      <span
        @click="sourceFilter = 'all'"
        :class="sourceFilter === 'all' ? 'text-green-500' : 'text-gray-500 cursor-pointer hover:text-gray-300'"
      >
        all
      </span>
      <span
        @click="sourceFilter = 'agent'"
        :class="sourceFilter === 'agent' ? 'text-cyan-500' : 'text-gray-500 cursor-pointer hover:text-gray-300'"
      >
        agent
      </span>
      <span
        @click="sourceFilter = 'claude-cli'"
        :class="sourceFilter === 'claude-cli' ? 'text-indigo-400' : 'text-gray-500 cursor-pointer hover:text-gray-300'"
      >
        cli
      </span>
      <span class="text-gray-700">|</span>
      <!-- Agent filter -->
      <span
        @click="filter = 'all'"
        :class="filter === 'all' ? 'text-green-500' : 'text-gray-500 cursor-pointer hover:text-gray-300'"
      >
        all agents
      </span>
      <span
        v-for="agent in uniqueAgents"
        :key="agent"
        @click="filter = agent"
        :class="filter === agent ? 'text-green-500' : 'text-gray-500 cursor-pointer hover:text-gray-300'"
      >
        {{ agent }}
      </span>
      <span class="flex-1"></span>
      <span
        v-if="!autoScroll"
        @click="scrollToBottom"
        class="text-yellow-500 cursor-pointer hover:text-yellow-400"
      >
        scroll locked - click to unlock
      </span>
    </div>

    <!-- Log output -->
    <div
      ref="logEl"
      @scroll="handleScroll"
      class="flex-1 overflow-y-auto px-2 py-1"
    >
      <div
        v-for="log in filteredLogs"
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
        class="text-green-500 cursor-pointer"
        :class="{ 'animate-pulse': autoScroll }"
      >
        _
      </div>
    </div>
  </div>
</template>

<style scoped>
/* Ensure proper scrolling behavior */
.overflow-y-auto {
  scrollbar-width: thin;
  scrollbar-color: #374151 #000;
}

.overflow-y-auto::-webkit-scrollbar {
  width: 6px;
}

.overflow-y-auto::-webkit-scrollbar-track {
  background: #000;
}

.overflow-y-auto::-webkit-scrollbar-thumb {
  background: #374151;
  border-radius: 3px;
}
</style>
