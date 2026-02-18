<script setup lang="ts">
import { computed } from 'vue'
import type { MemoryMetrics } from '../../types'

const props = defineProps<{
  metrics: MemoryMetrics | null
}>()

// Sort categories by entry count descending so the most active ones appear first
const sortedCategories = computed(() => {
  if (!props.metrics) return []
  return Object.entries(props.metrics.categories).sort(([, a], [, b]) => (b as number) - (a as number))
})

// Average access count per entry as a proxy for hit rate
const avgAccessCount = computed(() => {
  if (!props.metrics || props.metrics.total_entries === 0) return 0
  return (props.metrics.total_entries / Math.max(props.metrics.stores_count, 1)).toFixed(1)
})

const isEmpty = computed(() => !props.metrics || props.metrics.total_entries === 0)
</script>

<template>
  <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
    <h2 class="text-lg font-semibold mb-4 text-gray-200">Memory System</h2>

    <div v-if="isEmpty" class="text-gray-500 text-sm py-6 text-center">
      No data yet
    </div>

    <template v-else>
      <!-- Summary stats -->
      <div class="grid grid-cols-2 gap-3 mb-4">
        <div class="bg-gray-900 rounded p-3">
          <div class="text-2xl font-mono font-bold text-cyan-400">{{ metrics!.total_entries }}</div>
          <div class="text-xs text-gray-400 mt-1">Total Memories</div>
        </div>
        <div class="bg-gray-900 rounded p-3">
          <div class="text-2xl font-mono font-bold text-purple-400">{{ avgAccessCount }}</div>
          <div class="text-xs text-gray-400 mt-1">Avg / Store</div>
        </div>
      </div>

      <!-- Stores count -->
      <div class="flex items-center justify-between text-sm mb-4">
        <span class="text-gray-400">Active stores</span>
        <span class="font-mono text-gray-200">{{ metrics!.stores_count }}</span>
      </div>

      <!-- Categories breakdown -->
      <div v-if="sortedCategories.length > 0">
        <div class="text-xs text-gray-500 uppercase tracking-wider mb-2">By Category</div>
        <div class="space-y-2">
          <div
            v-for="[category, count] in sortedCategories"
            :key="category"
            class="flex items-center gap-2"
          >
            <div class="flex-1 min-w-0">
              <div class="flex items-center justify-between mb-1">
                <span class="text-xs text-gray-300 truncate">{{ category }}</span>
                <span class="text-xs font-mono text-gray-400 ml-2 shrink-0">{{ count }}</span>
              </div>
              <!-- Bar scaled relative to the highest-count category -->
              <div class="h-1 bg-gray-700 rounded-full overflow-hidden">
                <div
                  class="h-full bg-purple-500 rounded-full"
                  :style="{ width: `${(count / sortedCategories[0][1]) * 100}%` }"
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>
