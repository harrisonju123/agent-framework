<script setup lang="ts">
import { computed } from 'vue'
import type { HealthReport } from '../types'
import Tag from 'primevue/tag'

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
    class="rounded-xl p-5 border transition-colors"
    :class="health.passed
      ? 'bg-emerald-50 border-emerald-200'
      : 'bg-red-50 border-red-200'"
  >
    <div class="flex items-center justify-between mb-3">
      <h2 class="text-lg font-semibold text-slate-800 flex items-center gap-2">
        <span
          class="w-3 h-3 rounded-full"
          :class="health.passed ? 'bg-emerald-500' : 'bg-red-500'"
        ></span>
        System Health
      </h2>
      <Tag
        :value="health.passed ? 'HEALTHY' : 'DEGRADED'"
        :severity="health.passed ? 'success' : 'danger'"
      />
    </div>

    <div class="text-sm text-slate-500 mb-3">
      {{ passedCount }}/{{ totalCount }} checks passing
    </div>

    <div class="space-y-2">
      <div
        v-for="check in health.checks"
        :key="check.name"
        class="flex items-center gap-2 text-sm"
      >
        <span :class="check.passed ? 'text-emerald-500' : 'text-red-500'">
          {{ check.passed ? '&#10003;' : '&#10007;' }}
        </span>
        <span :class="check.passed ? 'text-slate-600' : 'text-red-600'">
          {{ formatCheckName(check.name) }}
        </span>
        <span v-if="!check.passed && check.message" class="text-xs text-red-400 truncate">
          - {{ check.message }}
        </span>
      </div>
    </div>

    <div v-if="health.warnings.length > 0" class="mt-3 pt-3 border-t border-slate-200">
      <p class="text-xs text-amber-600 font-medium mb-1">Warnings:</p>
      <ul class="text-xs text-amber-500 space-y-1">
        <li v-for="(warning, i) in health.warnings" :key="i">
          {{ warning }}
        </li>
      </ul>
    </div>
  </div>
</template>
