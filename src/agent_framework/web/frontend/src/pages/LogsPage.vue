<script setup lang="ts">
import { useAppState } from '../composables/useAppState'
import LogViewer from '../components/LogViewer.vue'
import Button from 'primevue/button'

const { logs, logsConnected, logsClear, logsReconnect, agentIds } = useAppState()
</script>

<template>
  <div class="space-y-3 h-full flex flex-col">
    <!-- Toolbar -->
    <div class="flex items-center justify-between shrink-0">
      <div class="flex items-center gap-3">
        <h2 class="text-lg font-semibold text-slate-800">Live Logs</h2>
        <span class="flex items-center gap-1.5 text-sm" :class="logsConnected ? 'text-emerald-600' : 'text-red-500'">
          <span class="w-2 h-2 rounded-full" :class="logsConnected ? 'bg-emerald-500' : 'bg-red-500'"></span>
          {{ logsConnected ? 'Connected' : 'Disconnected' }}
        </span>
      </div>
      <div class="flex gap-2">
        <Button label="Clear" severity="secondary" size="small" outlined @click="logsClear" />
        <Button v-if="!logsConnected" label="Reconnect" size="small" @click="logsReconnect" />
      </div>
    </div>

    <!-- Log Viewer (full remaining height) -->
    <div class="flex-1 min-h-0 rounded-xl overflow-hidden border border-slate-200">
      <LogViewer :logs="logs" :agents="agentIds" />
    </div>
  </div>
</template>
