<script setup lang="ts">
import { ref } from 'vue'
import { useAppState } from '../composables/useAppState'
import KpiCards from '../components/KpiCards.vue'
import AgentCard from '../components/AgentCard.vue'
import ActivityFeed from '../components/ActivityFeed.vue'
import FailedTasks from '../components/FailedTasks.vue'
import PendingCheckpoints from '../components/PendingCheckpoints.vue'
import CheckpointDetailDialog from '../components/dialogs/CheckpointDetailDialog.vue'
import type { CheckpointData } from '../types'

const {
  agents, queues, events, failedTasks, pendingCheckpoints, uptimeDisplay,
  handleRestart, handleRetryTask,
} = useAppState()

const selectedCheckpoint = ref<CheckpointData | null>(null)
const showCheckpointDialog = ref(false)

function handleSelectCheckpoint(checkpoint: CheckpointData) {
  selectedCheckpoint.value = checkpoint
  showCheckpointDialog.value = true
}
</script>

<template>
  <div class="space-y-6">
    <!-- KPI Cards -->
    <KpiCards
      :agents="agents"
      :queues="queues"
      :failed-tasks="failedTasks"
      :uptime-display="uptimeDisplay"
    />

    <!-- Agent Grid -->
    <div>
      <h2 class="text-lg font-semibold text-slate-800 mb-3">Agents</h2>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        <AgentCard
          v-for="agent in agents"
          :key="agent.id"
          :agent="agent"
          :on-restart="handleRestart"
        />
      </div>
      <div v-if="agents.length === 0" class="text-center py-8 text-slate-400">
        No agents configured
      </div>
    </div>

    <!-- Activity / Failed / Checkpoints panels -->
    <div
      class="grid grid-cols-1 gap-4"
      :class="pendingCheckpoints.length > 0 ? 'lg:grid-cols-3' : 'lg:grid-cols-2'"
    >
      <!-- Recent Activity -->
      <div class="bg-white shadow-sm border border-slate-200 rounded-xl overflow-hidden">
        <div class="px-4 py-3 border-b border-slate-200">
          <h3 class="text-sm font-medium text-slate-500">Recent Activity</h3>
        </div>
        <div class="max-h-64 overflow-y-auto">
          <ActivityFeed :events="events" />
        </div>
      </div>

      <!-- Failed Tasks -->
      <div class="bg-white shadow-sm border border-slate-200 rounded-xl overflow-hidden">
        <div class="px-4 py-3 border-b border-slate-200 flex items-center justify-between">
          <h3 class="text-sm font-medium text-slate-500">Failed Tasks</h3>
          <span v-if="failedTasks.length > 0" class="text-red-600 text-sm font-medium">({{ failedTasks.length }})</span>
        </div>
        <div class="max-h-64 overflow-y-auto">
          <FailedTasks :tasks="failedTasks" :on-retry="handleRetryTask" />
        </div>
      </div>

      <!-- Pending Checkpoints -->
      <div v-if="pendingCheckpoints.length > 0" class="bg-white shadow-sm border border-amber-200 rounded-xl overflow-hidden">
        <div class="px-4 py-3 border-b border-amber-200 flex items-center justify-between">
          <h3 class="text-sm font-medium text-amber-700">Awaiting Approval</h3>
          <span class="text-amber-600 text-sm font-medium">({{ pendingCheckpoints.length }})</span>
        </div>
        <div class="max-h-64 overflow-y-auto">
          <PendingCheckpoints :checkpoints="pendingCheckpoints" :on-select="handleSelectCheckpoint" />
        </div>
      </div>
    </div>

    <!-- Checkpoint Detail Dialog -->
    <CheckpointDetailDialog
      v-model:visible="showCheckpointDialog"
      :checkpoint="selectedCheckpoint"
      @close="selectedCheckpoint = null"
    />
  </div>
</template>
