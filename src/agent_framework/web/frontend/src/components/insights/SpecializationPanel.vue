<script setup lang="ts">
import type { SpecializationMetrics } from '../../types'

defineProps<{
  metrics: SpecializationMetrics | null | undefined
}>()

function getProfileLabel(profileId: string): string {
  // Convert profile IDs to readable names (e.g., 'frontend-engineer' -> 'Frontend Engineer')
  return profileId
    .split('-')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

function getTotalCount(metrics: SpecializationMetrics): number {
  return Object.values(metrics.profile_counts).reduce((sum, count) => sum + count, 0)
}

function getTopProfiles(metrics: SpecializationMetrics, limit: number = 5) {
  return Object.entries(metrics.profile_counts)
    .sort(([, a], [, b]) => b - a)
    .slice(0, limit)
}

function getPercentage(count: number, total: number): number {
  if (total === 0) return 0
  return Math.round((count / total) * 100)
}
</script>

<template>
  <div class="bg-white shadow-sm border border-slate-200 rounded-xl p-5">
    <h2 class="text-lg font-semibold text-slate-800 mb-4">Engineer Specializations</h2>

    <!-- Empty state -->
    <div v-if="!metrics || Object.keys(metrics.profile_counts).length === 0" class="text-slate-400 text-sm">
      No specialization data available
    </div>

    <!-- Metrics loaded -->
    <div v-else class="space-y-4">
      <!-- Total count -->
      <div class="grid grid-cols-2 gap-4">
        <div>
          <p class="text-xs text-slate-500 uppercase tracking-wide">Total Selections</p>
          <p class="text-2xl font-bold text-slate-800">{{ getTotalCount(metrics) }}</p>
        </div>
        <div>
          <p class="text-xs text-slate-500 uppercase tracking-wide">Sessions Scanned</p>
          <p class="text-2xl font-bold text-slate-800">{{ metrics.sessions_scanned }}</p>
        </div>
      </div>

      <!-- Top specializations -->
      <div class="pt-4 border-t border-slate-200">
        <p class="text-xs font-medium text-slate-600 mb-3">Top Specializations</p>
        <div class="space-y-3">
          <div
            v-for="[profileId, count] in getTopProfiles(metrics)"
            :key="profileId"
            class="flex items-center justify-between"
          >
            <div class="flex-1">
              <div class="flex items-center justify-between mb-1">
                <span class="text-sm font-medium text-slate-700">{{ getProfileLabel(profileId) }}</span>
                <span class="text-xs text-slate-500">{{ getPercentage(count, getTotalCount(metrics)) }}%</span>
              </div>
              <div class="w-full bg-slate-100 rounded-full h-2">
                <div
                  class="bg-blue-500 h-2 rounded-full"
                  :style="{ width: getPercentage(count, getTotalCount(metrics)) + '%' }"
                ></div>
              </div>
              <p class="text-xs text-slate-500 mt-1">{{ count }} selections</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
