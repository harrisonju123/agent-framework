<script setup lang="ts">
import { computed } from 'vue'
import type { Agent, QueueStats, FailedTask } from '../types'

const props = defineProps<{
  agents: Agent[]
  queues: QueueStats[]
  failedTasks: FailedTask[]
  uptimeDisplay: string
}>()

const activeAgents = computed(() => props.agents.filter(a => a.status === 'working').length)
const totalPending = computed(() => props.queues.reduce((sum, q) => sum + q.pending_count, 0))

const cards = computed(() => [
  {
    label: 'Active Agents',
    value: activeAgents.value,
    subtitle: `${props.agents.length} total`,
    icon: 'pi-server',
    color: 'emerald',
  },
  {
    label: 'Pending Tasks',
    value: totalPending.value,
    subtitle: `${props.queues.length} queues`,
    icon: 'pi-list',
    color: 'blue',
  },
  {
    label: 'Failed Tasks',
    value: props.failedTasks.length,
    subtitle: '',
    icon: 'pi-exclamation-triangle',
    color: 'red',
  },
  {
    label: 'System Uptime',
    value: props.uptimeDisplay,
    subtitle: '',
    icon: 'pi-clock',
    color: 'slate',
  },
])

function bgClass(color: string) {
  const map: Record<string, string> = {
    emerald: 'bg-emerald-50',
    blue: 'bg-blue-50',
    red: 'bg-red-50',
    slate: 'bg-slate-100',
  }
  return map[color] || 'bg-slate-100'
}

function iconClass(color: string) {
  const map: Record<string, string> = {
    emerald: 'text-emerald-600',
    blue: 'text-blue-600',
    red: 'text-red-600',
    slate: 'text-slate-600',
  }
  return map[color] || 'text-slate-600'
}
</script>

<template>
  <div class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
    <div
      v-for="card in cards"
      :key="card.label"
      class="bg-white shadow-sm border border-slate-200 rounded-xl p-5 flex items-center justify-between"
    >
      <div>
        <p class="text-sm font-medium text-slate-500">{{ card.label }}</p>
        <p class="text-2xl font-semibold text-slate-900 mt-1">{{ card.value }}</p>
        <p v-if="card.subtitle" class="text-xs text-slate-400 mt-0.5">{{ card.subtitle }}</p>
      </div>
      <div
        class="w-10 h-10 rounded-lg flex items-center justify-center"
        :class="bgClass(card.color)"
      >
        <span class="pi text-lg" :class="[card.icon, iconClass(card.color)]"></span>
      </div>
    </div>
  </div>
</template>
