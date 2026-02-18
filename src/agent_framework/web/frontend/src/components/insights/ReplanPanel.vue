<script setup lang="ts">
import type { ReplanMetrics } from '../../types'

defineProps<{
  metrics: ReplanMetrics | null | undefined
}>()

function getReplanRate(metrics: ReplanMetrics): number {
  if (metrics.sessions_scanned === 0) return 0
  return Math.round((metrics.sessions_with_replans / metrics.sessions_scanned) * 100)
}
</script>

<template>
  <div class="bg-white shadow-sm border border-slate-200 rounded-xl p-5">
    <h2 class="text-lg font-semibold text-slate-800 mb-4">Replanning</h2>

    <!-- Empty state -->
    <div v-if="!metrics || metrics.total_replans === 0" class="text-slate-400 text-sm">
      No replanning data available
    </div>

    <!-- Metrics loaded -->
    <div v-else class="space-y-4">
      <!-- Total replans -->
      <div class="grid grid-cols-2 gap-4">
        <div>
          <p class="text-xs text-slate-500 uppercase tracking-wide">Total Replans</p>
          <p class="text-2xl font-bold text-slate-800">{{ metrics.total_replans }}</p>
        </div>
        <div>
          <p class="text-xs text-slate-500 uppercase tracking-wide">Sessions w/ Replans</p>
          <p class="text-2xl font-bold text-slate-800">{{ metrics.sessions_with_replans }}</p>
        </div>
      </div>

      <!-- Replan rate -->
      <div class="pt-4 border-t border-slate-200">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm font-medium text-slate-700">Replan Rate</span>
          <span class="text-lg font-bold text-amber-600">{{ getReplanRate(metrics) }}%</span>
        </div>
        <div class="w-full bg-slate-100 rounded-full h-3">
          <div
            class="bg-amber-500 h-3 rounded-full transition-all"
            :style="{ width: getReplanRate(metrics) + '%' }"
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
