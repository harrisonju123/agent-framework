<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import type { AgenticInsights } from '../types'

const data = ref<AgenticInsights | null>(null)
const error = ref<string | null>(null)
let pollInterval: ReturnType<typeof setInterval> | null = null

async function fetchMetrics() {
  try {
    const response = await fetch('/api/analytics/agentic')
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    data.value = await response.json()
    error.value = null
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load'
  }
}

onMounted(() => {
  fetchMetrics()
  // Poll every 30 seconds — these metrics don't change rapidly
  pollInterval = setInterval(fetchMetrics, 30_000)
})

onUnmounted(() => {
  if (pollInterval !== null) clearInterval(pollInterval)
})

// Memory hit rate formatted as percentage
const memoryHitPct = computed(() => {
  if (!data.value) return '—'
  return `${Math.round(data.value.memory.hit_rate * 100)}%`
})

// Self-eval retry rate formatted as percentage
const selfEvalRetryPct = computed(() => {
  if (!data.value) return '—'
  return `${Math.round(data.value.self_eval.retry_rate * 100)}%`
})

// Replan trigger rate formatted as percentage
const replanRatePct = computed(() => {
  if (!data.value) return '—'
  return `${Math.round(data.value.replan.trigger_rate * 100)}%`
})

// Specialization breakdown sorted by count descending
const specializationEntries = computed(() => {
  if (!data.value) return []
  return Object.entries(data.value.specialization_distribution).sort((a, b) => b[1] - a[1])
})

// Context budget utilization
const avgBudgetUtilPct = computed(() => {
  if (!data.value || data.value.context_budget.avg_utilization_pct === null) return '—'
  return `${data.value.context_budget.avg_utilization_pct.toFixed(0)}%`
})

function specializationLabel(key: string): string {
  if (key === 'none') return 'None'
  return key.charAt(0).toUpperCase() + key.slice(1)
}
</script>

<template>
  <div>
    <h2 class="text-lg font-semibold text-slate-800 mb-3">Agentic Feature Insights</h2>

    <div v-if="error" class="text-sm text-red-500 mb-3">Failed to load: {{ error }}</div>

    <div class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">

      <!-- Memory hit rate -->
      <div class="bg-white shadow-sm border border-slate-200 rounded-xl p-5">
        <div class="flex items-center justify-between mb-3">
          <p class="text-sm font-medium text-slate-500">Memory Hit Rate</p>
          <span class="pi pi-database text-indigo-400"></span>
        </div>
        <p class="text-2xl font-semibold text-slate-900">{{ memoryHitPct }}</p>
        <p class="text-xs text-slate-400 mt-1">
          {{ data?.memory.accessed_entries ?? '—' }} / {{ data?.memory.total_entries ?? '—' }} entries recalled
        </p>
      </div>

      <!-- Self-eval retry rate -->
      <div class="bg-white shadow-sm border border-slate-200 rounded-xl p-5">
        <div class="flex items-center justify-between mb-3">
          <p class="text-sm font-medium text-slate-500">Self-Eval Catch Rate</p>
          <span class="pi pi-check-circle text-emerald-400"></span>
        </div>
        <p class="text-2xl font-semibold text-slate-900">{{ selfEvalRetryPct }}</p>
        <p class="text-xs text-slate-400 mt-1">
          {{ data?.self_eval.failed_evaluations ?? '—' }} rewrites from {{ data?.self_eval.total_evaluations ?? '—' }} evals
        </p>
      </div>

      <!-- Replan trigger rate -->
      <div class="bg-white shadow-sm border border-slate-200 rounded-xl p-5">
        <div class="flex items-center justify-between mb-3">
          <p class="text-sm font-medium text-slate-500">Replan Rate</p>
          <span class="pi pi-refresh text-amber-400"></span>
        </div>
        <p class="text-2xl font-semibold text-slate-900">{{ replanRatePct }}</p>
        <p class="text-xs text-slate-400 mt-1">
          {{ data?.replan.sessions_with_replan ?? '—' }} / {{ data?.replan.total_sessions ?? '—' }} sessions replanned
        </p>
      </div>

      <!-- Specialization distribution -->
      <div class="bg-white shadow-sm border border-slate-200 rounded-xl p-5">
        <div class="flex items-center justify-between mb-3">
          <p class="text-sm font-medium text-slate-500">Specialization Profiles</p>
          <span class="pi pi-tag text-blue-400"></span>
        </div>
        <div v-if="specializationEntries.length > 0" class="space-y-1.5 mt-1">
          <div
            v-for="[profile, count] in specializationEntries"
            :key="profile"
            class="flex items-center justify-between text-sm"
          >
            <span class="text-slate-600">{{ specializationLabel(profile) }}</span>
            <span class="font-medium text-slate-800">{{ count }}</span>
          </div>
        </div>
        <p v-else class="text-xs text-slate-400 mt-1">No active agents</p>
      </div>

      <!-- Context budget utilization -->
      <div class="bg-white shadow-sm border border-slate-200 rounded-xl p-5">
        <div class="flex items-center justify-between mb-3">
          <p class="text-sm font-medium text-slate-500">Avg Token Utilization</p>
          <span class="pi pi-chart-bar text-violet-400"></span>
        </div>
        <p class="text-2xl font-semibold text-slate-900">{{ avgBudgetUtilPct }}</p>
        <p class="text-xs text-slate-400 mt-1">
          {{ data?.context_budget.budget_exceeded_events ?? '—' }} budget overruns
          across {{ data?.context_budget.total_tasks_with_tokens ?? '—' }} tasks
        </p>
      </div>

    </div>
  </div>
</template>
