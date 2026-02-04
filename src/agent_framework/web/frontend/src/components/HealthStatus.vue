<script setup lang="ts">
import { computed } from 'vue'
import type { HealthReport } from '../types'

const props = defineProps<{
  health: HealthReport
}>()

function formatCheckName(name: string): string {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

const passedCount = computed(() => props.health.checks.filter(c => c.passed).length)
const totalCount = computed(() => props.health.checks.length)
</script>

<template>
  <div
    class="rounded-lg p-4 border transition-colors"
    :class="health.passed
      ? 'bg-green-500/10 border-green-500/30'
      : 'bg-red-500/10 border-red-500/30'"
  >
    <div class="flex items-center justify-between mb-3">
      <h2 class="text-lg font-semibold flex items-center gap-2">
        <span
          class="w-3 h-3 rounded-full"
          :class="health.passed ? 'bg-green-500' : 'bg-red-500'"
        ></span>
        System Health
      </h2>
      <span
        class="px-2 py-1 text-sm font-medium rounded"
        :class="health.passed
          ? 'bg-green-500/20 text-green-400'
          : 'bg-red-500/20 text-red-400'"
      >
        {{ health.passed ? 'HEALTHY' : 'DEGRADED' }}
      </span>
    </div>

    <div class="text-sm text-gray-400 mb-3">
      {{ passedCount }}/{{ totalCount }} checks passing
    </div>

    <div class="space-y-2">
      <div
        v-for="check in health.checks"
        :key="check.name"
        class="flex items-center gap-2 text-sm"
      >
        <span :class="check.passed ? 'text-green-500' : 'text-red-500'">
          {{ check.passed ? '&#10003;' : '&#10007;' }}
        </span>
        <span :class="check.passed ? 'text-gray-400' : 'text-red-400'">
          {{ formatCheckName(check.name) }}
        </span>
        <span v-if="!check.passed && check.message" class="text-xs text-red-400/70 truncate">
          - {{ check.message }}
        </span>
      </div>
    </div>

    <div v-if="health.warnings.length > 0" class="mt-3 pt-3 border-t border-gray-700">
      <p class="text-xs text-yellow-500 font-medium mb-1">Warnings:</p>
      <ul class="text-xs text-yellow-400/70 space-y-1">
        <li v-for="(warning, i) in health.warnings" :key="i">
          {{ warning }}
        </li>
      </ul>
    </div>
  </div>
</template>
