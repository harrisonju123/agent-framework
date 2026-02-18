<script setup lang="ts">
import { computed } from 'vue'
import type { SelfEvalMetrics } from '../../types'

const props = defineProps<{
  metrics: SelfEvalMetrics | null
}>()

// Tasks that passed without any retries
const passCount = computed(() => {
  if (!props.metrics) return 0
  // tasks_evaluated includes all tasks that went through self-eval;
  // total_retries is the number of retry rounds triggered
  return Math.max(0, props.metrics.tasks_evaluated - props.metrics.total_retries)
})

// Pass rate as a 0-100 percentage
const passRate = computed(() => {
  if (!props.metrics || props.metrics.tasks_evaluated === 0) return 0
  return Math.round((passCount.value / props.metrics.tasks_evaluated) * 100)
})

const isEmpty = computed(() => !props.metrics || props.metrics.tasks_evaluated === 0)
</script>

<template>
  <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
    <h2 class="text-lg font-semibold mb-4 text-gray-200">Self-Evaluation</h2>

    <div v-if="isEmpty" class="text-gray-500 text-sm py-6 text-center">
      No data yet
    </div>

    <template v-else>
      <!-- Summary stats -->
      <div class="grid grid-cols-2 gap-3 mb-4">
        <div class="bg-gray-900 rounded p-3">
          <div class="text-2xl font-mono font-bold text-cyan-400">{{ metrics!.tasks_evaluated }}</div>
          <div class="text-xs text-gray-400 mt-1">Tasks Evaluated</div>
        </div>
        <div class="bg-gray-900 rounded p-3">
          <div class="text-2xl font-mono font-bold text-yellow-400">{{ metrics!.total_retries }}</div>
          <div class="text-xs text-gray-400 mt-1">Issues Caught</div>
        </div>
      </div>

      <!-- Pass / fail progress bar -->
      <div class="mb-2">
        <div class="flex items-center justify-between text-xs mb-1">
          <span class="text-gray-400">Pass rate</span>
          <span class="font-mono" :class="passRate >= 80 ? 'text-green-400' : passRate >= 50 ? 'text-yellow-400' : 'text-red-400'">
            {{ passRate }}%
          </span>
        </div>
        <div class="h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            class="h-full rounded-full transition-all"
            :class="passRate >= 80 ? 'bg-green-500' : passRate >= 50 ? 'bg-yellow-500' : 'bg-red-500'"
            :style="{ width: `${passRate}%` }"
          ></div>
        </div>
      </div>

      <div class="flex items-center justify-between text-xs text-gray-500 mt-1">
        <span>{{ passCount }} passed</span>
        <span>{{ metrics!.total_retries }} retried</span>
      </div>
    </template>
  </div>
</template>
