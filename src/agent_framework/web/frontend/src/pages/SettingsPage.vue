<script setup lang="ts">
import { useAppState } from '../composables/useAppState'
import SetupWizard from '../components/SetupWizard.vue'
import HealthStatus from '../components/HealthStatus.vue'
import QueuePanel from '../components/QueuePanel.vue'

const { setupComplete, health, queues, handleSetupComplete } = useAppState()
</script>

<template>
  <div class="space-y-6 max-w-4xl">
    <!-- Health Status -->
    <HealthStatus :health="health" />

    <!-- Queue Status -->
    <QueuePanel :queues="queues" />

    <!-- Setup Wizard -->
    <div class="bg-white shadow-sm border border-slate-200 rounded-xl overflow-hidden">
      <div class="px-6 py-4 border-b border-slate-200">
        <h2 class="text-lg font-semibold text-slate-800">Configuration</h2>
        <p v-if="setupComplete" class="text-sm text-emerald-600 mt-1">System is configured and ready</p>
        <p v-else class="text-sm text-amber-600 mt-1">Setup required before starting agents</p>
      </div>
      <div class="p-6">
        <SetupWizard @complete="handleSetupComplete" />
      </div>
    </div>

    <!-- Keyboard Shortcuts Reference -->
    <div class="bg-white shadow-sm border border-slate-200 rounded-xl overflow-hidden">
      <div class="px-6 py-4 border-b border-slate-200">
        <h2 class="text-lg font-semibold text-slate-800">Keyboard Shortcuts</h2>
      </div>
      <div class="p-6">
        <div class="grid grid-cols-2 sm:grid-cols-3 gap-3 text-sm">
          <div v-for="s in [
            { key: 's', label: 'Start All Agents' },
            { key: 'x', label: 'Stop All Agents' },
            { key: 'p', label: 'Pause / Resume' },
            { key: 'w', label: 'New Work' },
            { key: 'a', label: 'Analyze Repo' },
            { key: 't', label: 'Run Ticket' },
            { key: 'r', label: 'Retry All Failed' },
            { key: 'c', label: 'Approve All Checkpoints' },
            { key: '1-5', label: 'Navigate Pages' },
          ]" :key="s.key" class="flex items-center gap-2">
            <kbd class="px-2 py-1 bg-slate-100 border border-slate-200 rounded text-xs font-mono text-slate-600 min-w-[28px] text-center">{{ s.key }}</kbd>
            <span class="text-slate-600">{{ s.label }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
