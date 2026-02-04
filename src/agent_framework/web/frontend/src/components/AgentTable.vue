<script setup lang="ts">
import { computed } from 'vue'
import type { Agent } from '../types'

const props = defineProps<{
  agents: Agent[]
  onRestart: (agentId: string) => void
}>()

function formatElapsed(seconds: number | null): string {
  if (!seconds) return '-'
  const minutes = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${minutes}m ${secs}s`
}

function formatPhase(phase: string | null): string {
  if (!phase) return '-'
  return phase.replace(/_/g, ' ')
}

function getStatusClass(status: string): string {
  switch (status) {
    case 'working':
      return 'text-green-400'
    case 'completing':
      return 'text-blue-400'
    case 'idle':
      return 'text-gray-500'
    case 'dead':
      return 'text-red-400'
    default:
      return 'text-gray-600'
  }
}

function getStatusText(status: string): string {
  return status.toUpperCase()
}

function getJiraKey(agent: Agent): string {
  if (!agent.current_task) return '-'
  // Try to extract JIRA key from task context or ID
  const taskId = agent.current_task.id
  const match = taskId.match(/^([A-Z]+-\d+)/)
  return match ? match[1] : '-'
}
</script>

<template>
  <div class="font-mono text-sm">
    <!-- Header -->
    <div class="grid grid-cols-12 gap-2 px-2 py-1 text-gray-500 text-xs border-b border-gray-800">
      <div class="col-span-2">AGENT</div>
      <div class="col-span-1">STATUS</div>
      <div class="col-span-2">PHASE</div>
      <div class="col-span-4">TASK</div>
      <div class="col-span-1">JIRA</div>
      <div class="col-span-1">TIME</div>
      <div class="col-span-1"></div>
    </div>

    <!-- Agent rows -->
    <div
      v-for="agent in agents"
      :key="agent.id"
      class="grid grid-cols-12 gap-2 px-2 py-1 border-b border-gray-800/50 hover:bg-gray-800/30"
    >
      <!-- Agent name -->
      <div class="col-span-2 text-gray-300 truncate">
        {{ agent.id }}
      </div>

      <!-- Status -->
      <div class="col-span-1" :class="getStatusClass(agent.status)">
        {{ getStatusText(agent.status) }}
      </div>

      <!-- Phase -->
      <div class="col-span-2 text-cyan-400 truncate">
        {{ formatPhase(agent.current_phase) }}
      </div>

      <!-- Task title -->
      <div class="col-span-4 text-gray-400 truncate" :title="agent.current_task?.title">
        {{ agent.current_task?.title || '-' }}
      </div>

      <!-- JIRA key -->
      <div class="col-span-1 text-yellow-400">
        {{ getJiraKey(agent) }}
      </div>

      <!-- Elapsed time -->
      <div class="col-span-1 text-gray-500">
        {{ formatElapsed(agent.elapsed_seconds) }}
      </div>

      <!-- Actions -->
      <div class="col-span-1 text-right">
        <button
          v-if="agent.status === 'dead'"
          @click="onRestart(agent.id)"
          class="text-red-400 hover:text-red-300 text-xs"
        >
          restart
        </button>
      </div>
    </div>

    <!-- Empty state -->
    <div v-if="agents.length === 0" class="px-2 py-4 text-gray-500 text-center">
      No agents configured
    </div>
  </div>
</template>
