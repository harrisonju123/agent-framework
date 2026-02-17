<script setup lang="ts">
import type { Agent } from '../types'
import DataTable from 'primevue/datatable'
import Column from 'primevue/column'
import Tag from 'primevue/tag'
import Button from 'primevue/button'

defineProps<{
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

function getStatusSeverity(status: string): string {
  switch (status) {
    case 'working': return 'success'
    case 'completing': return 'info'
    case 'idle': return 'warn'
    case 'dead': return 'danger'
    default: return 'secondary'
  }
}

function getJiraKey(agent: Agent): string {
  if (!agent.current_task) return '-'
  const taskId = agent.current_task.id
  const match = taskId.match(/^([A-Z]+-\d+)/)
  return match ? match[1] : '-'
}
</script>

<template>
  <DataTable :value="agents" stripedRows class="text-sm">
    <template #empty>
      <div class="text-center py-6 text-slate-400">No agents configured</div>
    </template>
    <Column field="id" header="Agent">
      <template #body="{ data }">
        <span class="font-medium text-slate-800">{{ data.id }}</span>
      </template>
    </Column>
    <Column header="Status" style="width: 100px">
      <template #body="{ data }">
        <Tag :value="data.status.toUpperCase()" :severity="getStatusSeverity(data.status) as any" />
      </template>
    </Column>
    <Column header="Phase" style="width: 140px">
      <template #body="{ data }">
        <span class="text-blue-600">{{ formatPhase(data.current_phase) }}</span>
      </template>
    </Column>
    <Column header="Current Task">
      <template #body="{ data }">
        <span class="text-slate-600 truncate block max-w-xs" :title="data.current_task?.title">
          {{ data.current_task?.title || '-' }}
        </span>
      </template>
    </Column>
    <Column header="JIRA" style="width: 90px">
      <template #body="{ data }">
        <span class="text-amber-600 font-mono text-xs">{{ getJiraKey(data) }}</span>
      </template>
    </Column>
    <Column header="Elapsed" style="width: 80px">
      <template #body="{ data }">
        <span class="text-slate-400">{{ formatElapsed(data.elapsed_seconds) }}</span>
      </template>
    </Column>
    <Column header="Actions" style="width: 100px">
      <template #body="{ data }">
        <Button
          v-if="data.status === 'dead'"
          label="Restart"
          severity="danger"
          size="small"
          @click="onRestart(data.id)"
        />
      </template>
    </Column>
  </DataTable>
</template>
