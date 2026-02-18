<script setup lang="ts">
import type { SelfEvalMetrics } from '../../types'

defineProps<{
  metrics: SelfEvalMetrics | null | undefined
}>()

function getSuccessRate(metrics: SelfEvalMetrics): number {
  if (metrics.total_evals === 0) return 0
  return Math.round((metrics.passed / metrics.total_evals) * 100)
}
</script>

<template>
  <div class="bg-white shadow-sm border border-slate-200 rounded-xl p-5">
    <h2 class="text-lg font-semibold text-slate-800 mb-4">Self-Evaluation</h2>

    <!-- Empty state -->
    <div v-if="!metrics || metrics.total_evals === 0" class="text-slate-400 text-sm">
      No self-evaluation data available
    </div>

    <!-- Metrics loaded -->
    <div v-else class="space-y-4">
      <!-- Success rate -->
      <div class="grid grid-cols-3 gap-3">
        <div>
          <p class="text-xs text-slate-500 uppercase tracking-wide">Total</p>
          <p class="text-2xl font-bold text-slate-800">{{ metrics.total_evals }}</p>
        </div>
        <div>
          <p class="text-xs text-slate-500 uppercase tracking-wide">Passed</p>
          <p class="text-2xl font-bold text-green-600">{{ metrics.passed }}</p>
        </div>
        <div>
          <p class="text-xs text-slate-500 uppercase tracking-wide">Failed</p>
          <p class="text-2xl font-bold text-red-600">{{ metrics.failed }}</p>
        </div>
      </div>

      <!-- Success rate bar -->
      <div class="pt-4 border-t border-slate-200">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm font-medium text-slate-700">Success Rate</span>
          <span class="text-lg font-bold text-green-600">{{ getSuccessRate(metrics) }}%</span>
        </div>
        <div class="w-full bg-slate-100 rounded-full h-3">
          <div
            class="bg-green-500 h-3 rounded-full transition-all"
            :style="{ width: getSuccessRate(metrics) + '%' }"
          ></div>
        </div>
      </div>

      <!-- Sessions scanned -->
      <div class="pt-4 border-t border-slate-200">
        <p class="text-sm text-slate-600">Sessions Scanned</p>
        <p class="text-lg font-bold text-slate-800 mt-1">{{ metrics.sessions_scanned }}</p>
      </div>
    </div>
  </div>
</template>
