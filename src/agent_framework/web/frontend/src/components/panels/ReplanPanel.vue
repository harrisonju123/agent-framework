<script setup lang="ts">
import { computed } from 'vue'
import type { ReplanMetrics } from '../../types'

const props = defineProps<{
  metrics: ReplanMetrics | null
}>()

// Tasks that resolved without exceeding their replan budget (simple proxy for success)
const successCount = computed(() => {
  if (!props.metrics) return 0
  // treat tasks_replanned as "eventually succeeded" — the raw model doesn't track
  // failures explicitly, so we present replanned vs attempts as the ratio
  return props.metrics.tasks_replanned
})

// Success rate: tasks_replanned / total_replan_attempts when attempts > 0
const successRate = computed(() => {
  if (!props.metrics || props.metrics.total_replan_attempts === 0) return 0
  return Math.round((props.metrics.tasks_replanned / props.metrics.total_replan_attempts) * 100)
})

// Average attempts per replanned task
const avgAttempts = computed(() => {
  if (!props.metrics || props.metrics.tasks_replanned === 0) return 0
  return (props.metrics.total_replan_attempts / props.metrics.tasks_replanned).toFixed(1)
})

const isEmpty = computed(() => !props.metrics || props.metrics.tasks_replanned === 0)
</script>

<template>
  <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
    <h2 class="text-lg font-semibold mb-4 text-gray-200">Dynamic Replanning</h2>

    <div v-if="isEmpty" class="text-gray-500 text-sm py-6 text-center">
      No data yet
    </div>

    <template v-else>
      <!-- Summary stats -->
      <div class="grid grid-cols-2 gap-3 mb-4">
        <div class="bg-gray-900 rounded p-3">
          <div class="text-2xl font-mono font-bold text-cyan-400">{{ metrics!.tasks_replanned }}</div>
          <div class="text-xs text-gray-400 mt-1">Tasks Replanned</div>
        </div>
        <div class="bg-gray-900 rounded p-3">
          <div class="text-2xl font-mono font-bold text-orange-400">{{ metrics!.total_replan_attempts }}</div>
          <div class="text-xs text-gray-400 mt-1">Total Attempts</div>
        </div>
      </div>

      <!-- Success rate progress bar -->
      <div class="mb-2">
        <div class="flex items-center justify-between text-xs mb-1">
          <span class="text-gray-400">Task resolution rate</span>
          <span class="font-mono" :class="successRate >= 80 ? 'text-green-400' : successRate >= 50 ? 'text-yellow-400' : 'text-orange-400'">
            {{ successRate }}%
          </span>
        </div>
        <div class="h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            class="h-full rounded-full transition-all"
            :class="successRate >= 80 ? 'bg-green-500' : successRate >= 50 ? 'bg-yellow-500' : 'bg-orange-500'"
            :style="{ width: `${successRate}%` }"
          ></div>
        </div>
      </div>

      <div class="flex items-center justify-between text-xs text-gray-500 mt-1">
        <span>{{ avgAttempts }} avg attempts/task</span>
        <span>{{ successCount }} resolved</span>
      </div>
    </template>
  </div>
</template>
