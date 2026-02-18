<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useApi } from '../composables/useApi'
import type { AgenticMetrics } from '../types'
import MemoryPanel from '../components/panels/MemoryPanel.vue'
import SelfEvalPanel from '../components/panels/SelfEvalPanel.vue'
import ReplanPanel from '../components/panels/ReplanPanel.vue'

const { fetchAgenticMetrics, loading, error } = useApi()

const metrics = ref<AgenticMetrics | null>(null)

async function refresh() {
  const result = await fetchAgenticMetrics()
  if (result) {
    metrics.value = result
  }
}

onMounted(refresh)
</script>

<template>
  <div class="flex-1 overflow-y-auto p-4">
    <div class="max-w-5xl mx-auto">
      <!-- Page header -->
      <div class="flex items-center justify-between mb-6">
        <div>
          <h1 class="text-xl font-semibold text-gray-200 font-mono">Insights</h1>
          <p class="text-sm text-gray-500 mt-1">Agentic feature metrics — memory, self-evaluation, and replanning</p>
        </div>
        <button
          @click="refresh"
          :disabled="loading"
          class="px-3 py-1.5 text-sm font-mono bg-gray-700 hover:bg-gray-600 text-gray-200 rounded transition-colors disabled:opacity-50"
          aria-label="Refresh metrics"
        >
          {{ loading ? 'Loading…' : 'Refresh' }}
        </button>
      </div>

      <!-- Error state -->
      <div v-if="error" class="mb-4 px-4 py-3 bg-red-900/40 border border-red-700 rounded text-red-300 text-sm font-mono">
        Failed to load metrics: {{ error }}
      </div>

      <!-- Computed-at timestamp -->
      <div v-if="metrics" class="text-xs text-gray-600 font-mono mb-4">
        Computed at {{ new Date(metrics.computed_at).toLocaleTimeString() }}
      </div>

      <!-- Panel grid — the three required panels -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MemoryPanel :metrics="metrics?.memory ?? null" />
        <SelfEvalPanel :metrics="metrics?.self_eval ?? null" />
        <ReplanPanel :metrics="metrics?.replan ?? null" />
      </div>
    </div>
  </div>
</template>
