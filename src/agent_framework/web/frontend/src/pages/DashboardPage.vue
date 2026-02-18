<script setup lang="ts">
import { useAppState } from '../composables/useAppState'
import KpiCards from '../components/KpiCards.vue'
import AgentCard from '../components/AgentCard.vue'
import ActivityFeed from '../components/ActivityFeed.vue'
import FailedTasks from '../components/FailedTasks.vue'

const {
  agents, queues, events, failedTasks,
  handleRestart, handleRetryTask,
} = useAppState()
</script>

<template>
  <div class="space-y-6">
    <!-- KPI Cards -->
    <KpiCards
      :agents="agents"
      :queues="queues"
      :failed-tasks="failedTasks"
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

    <!-- Activity / Failed panels -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
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
    </div>
  </div>
</template>
