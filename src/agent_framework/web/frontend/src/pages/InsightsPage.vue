<script setup lang="ts">
import { useAppState } from '../composables/useAppState'
import MemoryPanel from '../components/insights/MemoryPanel.vue'
import SelfEvalPanel from '../components/insights/SelfEvalPanel.vue'
import ReplanPanel from '../components/insights/ReplanPanel.vue'
import SpecializationPanel from '../components/insights/SpecializationPanel.vue'
import DebatePanel from '../components/insights/DebatePanel.vue'
import ContextBudgetPanel from '../components/insights/ContextBudgetPanel.vue'

const { agenticMetrics } = useAppState()
</script>

<template>
  <div class="space-y-6">
    <!-- Page header -->
    <div>
      <h1 class="text-2xl font-bold text-slate-900">Agentic Observability</h1>
      <p class="text-slate-600 mt-1">
        Real-time metrics and insights into agent framework behavior and performance
      </p>
      <p v-if="agenticMetrics?.computed_at" class="text-xs text-slate-500 mt-2">
        Last updated: {{ new Date(agenticMetrics.computed_at).toLocaleString() }}
      </p>
    </div>

    <!-- 2x3 Grid of metric panels -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      <!-- Row 1 -->
      <MemoryPanel :metrics="agenticMetrics?.memory" />
      <SelfEvalPanel :metrics="agenticMetrics?.self_eval" />
      <ReplanPanel :metrics="agenticMetrics?.replan" />

      <!-- Row 2 -->
      <SpecializationPanel :metrics="agenticMetrics?.specialization" />
      <DebatePanel :metrics="agenticMetrics?.debates" />
      <ContextBudgetPanel :metrics="agenticMetrics?.context_budget" />
    </div>

    <!-- No data state -->
    <div v-if="!agenticMetrics" class="text-center py-12">
      <div class="text-slate-400">
        <p class="text-lg font-medium">Waiting for metrics data...</p>
        <p class="text-sm mt-2">Agentic observability metrics will appear as sessions are processed</p>
      </div>
    </div>
  </div>
</template>
