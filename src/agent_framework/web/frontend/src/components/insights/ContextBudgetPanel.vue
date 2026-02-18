<script setup lang="ts">
import type { ContextBudgetMetrics } from '../../types'

defineProps<{
  metrics: ContextBudgetMetrics | null | undefined
}>()

function getTotalExceeded(metrics: ContextBudgetMetrics): number {
  return Object.values(metrics.exceeded_by_agent).reduce((sum, count) => sum + count, 0)
}

function getAgentsWithExceeded(metrics: ContextBudgetMetrics) {
  return Object.entries(metrics.exceeded_by_agent)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5)
}
</script>

<template>
  <div class="bg-white shadow-sm border border-slate-200 rounded-xl p-5">
    <h2 class="text-lg font-semibold text-slate-800 mb-4">Context Budget Usage</h2>

    <!-- Empty state -->
    <div v-if="!metrics || metrics.budget_exceeded_count === 0" class="text-slate-400 text-sm">
      No context budget exceedances
    </div>

    <!-- Metrics loaded -->
    <div v-else class="space-y-4">
      <!-- Total exceeded count -->
      <div class="grid grid-cols-2 gap-4">
        <div>
          <p class="text-xs text-slate-500 uppercase tracking-wide">Times Exceeded</p>
          <p class="text-2xl font-bold text-red-600">{{ metrics.budget_exceeded_count }}</p>
        </div>
        <div>
          <p class="text-xs text-slate-500 uppercase tracking-wide">Affected Agents</p>
          <p class="text-2xl font-bold text-slate-800">{{ Object.keys(metrics.exceeded_by_agent).length }}</p>
        </div>
      </div>

      <!-- Breakdown by agent -->
      <div v-if="Object.keys(metrics.exceeded_by_agent).length > 0" class="pt-4 border-t border-slate-200">
        <p class="text-xs font-medium text-slate-600 mb-3">Exceedances by Agent</p>
        <div class="space-y-2">
          <div
            v-for="[agent, count] in getAgentsWithExceeded(metrics)"
            :key="agent"
            class="flex items-center justify-between text-sm"
          >
            <span class="text-slate-700">{{ agent }}</span>
            <span class="font-mono text-red-600 font-medium">{{ count }}</span>
          </div>
        </div>
      </div>

      <!-- Alert -->
      <div class="pt-4 border-t border-slate-200">
        <div class="flex items-start gap-2 p-3 bg-red-50 rounded-lg">
          <div class="w-4 h-4 text-red-600 flex-shrink-0 mt-0.5">⚠️</div>
          <div>
            <p class="text-xs font-medium text-red-900">Budget Exceeded</p>
            <p class="text-xs text-red-700 mt-0.5">
              {{ getTotalExceeded(metrics) }} task(s) exceeded context window budget
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
