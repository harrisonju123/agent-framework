<script setup lang="ts">
import type { MemoryMetrics } from '../../types'

defineProps<{
  metrics: MemoryMetrics | null | undefined
}>()

function getTopCategories(metrics: MemoryMetrics, limit: number = 5) {
  return Object.entries(metrics.categories)
    .sort(([, a], [, b]) => b - a)
    .slice(0, limit)
}

function getTotalEntries(metrics: MemoryMetrics): number {
  return Object.values(metrics.categories).reduce((sum, count) => sum + count, 0)
}
</script>

<template>
  <div class="bg-white shadow-sm border border-slate-200 rounded-xl p-5">
    <h2 class="text-lg font-semibold text-slate-800 mb-4">Agent Memory</h2>

    <!-- Empty state -->
    <div v-if="!metrics || metrics.total_entries === 0" class="text-slate-400 text-sm">
      No memory data available
    </div>

    <!-- Metrics loaded -->
    <div v-else class="space-y-4">
      <!-- Total entries and stores -->
      <div class="grid grid-cols-2 gap-4">
        <div>
          <p class="text-xs text-slate-500 uppercase tracking-wide">Total Entries</p>
          <p class="text-2xl font-bold text-slate-800">{{ metrics.total_entries }}</p>
        </div>
        <div>
          <p class="text-xs text-slate-500 uppercase tracking-wide">Memory Stores</p>
          <p class="text-2xl font-bold text-slate-800">{{ metrics.stores_count }}</p>
        </div>
      </div>

      <!-- Categories breakdown -->
      <div v-if="Object.keys(metrics.categories).length > 0" class="pt-4 border-t border-slate-200">
        <p class="text-xs font-medium text-slate-600 mb-3">Top Categories</p>
        <div class="space-y-2">
          <div
            v-for="[category, count] in getTopCategories(metrics)"
            :key="category"
            class="flex items-center justify-between text-sm"
          >
            <span class="text-slate-700 capitalize">{{ category }}</span>
            <span class="font-mono text-slate-800 font-medium">{{ count }}</span>
          </div>
        </div>
      </div>

      <!-- Status indicator -->
      <div class="pt-4 border-t border-slate-200">
        <div class="flex items-center gap-2">
          <div class="w-3 h-3 rounded-full bg-green-500"></div>
          <span class="text-sm text-slate-700">
            Memory actively storing {{ getTotalEntries(metrics) }} entries
          </span>
        </div>
      </div>
    </div>
  </div>
</template>
