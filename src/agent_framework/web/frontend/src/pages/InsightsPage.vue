<script setup lang="ts">
import { ref, onMounted, onUnmounted, computed } from 'vue'
import type { AgenticMetricsReport } from '../types'

const report = ref<AgenticMetricsReport | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)
const hours = ref(24)
let pollTimer: ReturnType<typeof setTimeout> | null = null
let polling = true

async function fetchMetrics() {
  loading.value = true
  error.value = null
  try {
    const res = await fetch(`/api/agentic-metrics?hours=${hours.value}`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    report.value = await res.json()
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to fetch metrics'
  } finally {
    loading.value = false
  }
}

async function pollLoop() {
  await fetchMetrics()
  if (polling) {
    pollTimer = setTimeout(pollLoop, 30_000)
  }
}

onMounted(() => { pollLoop() })
onUnmounted(() => {
  polling = false
  if (pollTimer) clearTimeout(pollTimer)
})

// Formatting helpers
function pct(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`
}

function fmtK(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`
}

const specializationRows = computed(() => {
  if (!report.value) return []
  const dist = report.value.specialization.distribution
  const total = report.value.specialization.total_active_agents
  return Object.entries(dist)
    .sort((a, b) => b[1] - a[1])
    .map(([profile, count]) => ({
      profile: profile === 'none' ? 'Unspecialized' : profile,
      count,
      pct: total > 0 ? pct(count / total) : '—',
    }))
})
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-xl font-semibold text-slate-800">Agentic Insights</h1>
        <p class="text-sm text-slate-500 mt-0.5">Memory, self-evaluation, replanning, and context budget signals</p>
      </div>
      <div class="flex items-center gap-3">
        <label class="text-sm text-slate-500">Lookback</label>
        <select
          v-model="hours"
          @change="fetchMetrics"
          class="text-sm border border-slate-200 rounded-lg px-2 py-1.5 bg-white text-slate-700"
        >
          <option :value="6">6 h</option>
          <option :value="24">24 h</option>
          <option :value="72">72 h</option>
          <option :value="168">7 d</option>
        </select>
        <span v-if="loading" class="pi pi-spin pi-spinner text-blue-500"></span>
      </div>
    </div>

    <!-- Error banner -->
    <div v-if="error" class="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
      {{ error }}
    </div>

    <!-- Skeleton while first load -->
    <div v-else-if="!report" class="text-center py-16 text-slate-400">
      <span class="pi pi-spin pi-spinner mr-2"></span> Loading insights…
    </div>

    <template v-else>
      <!-- Summary strip -->
      <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div class="bg-white border border-slate-200 rounded-xl px-4 py-4">
          <p class="text-xs font-medium text-slate-400 uppercase tracking-wide">Tasks observed</p>
          <p class="text-2xl font-semibold text-slate-800 mt-1">{{ report.total_observed_tasks }}</p>
          <p class="text-xs text-slate-400 mt-1">last {{ report.time_range_hours }}h</p>
        </div>
        <div class="bg-white border border-slate-200 rounded-xl px-4 py-4">
          <p class="text-xs font-medium text-slate-400 uppercase tracking-wide">Memory recall rate</p>
          <p class="text-2xl font-semibold text-blue-700 mt-1">{{ pct(report.memory.recall_rate) }}</p>
          <p class="text-xs text-slate-400 mt-1">{{ report.memory.total_recalls }} total recalls</p>
        </div>
        <div class="bg-white border border-slate-200 rounded-xl px-4 py-4">
          <p class="text-xs font-medium text-slate-400 uppercase tracking-wide">Self-eval catch rate</p>
          <p class="text-2xl font-semibold text-amber-600 mt-1">{{ pct(report.self_eval.catch_rate) }}</p>
          <p class="text-xs text-slate-400 mt-1">{{ report.self_eval.total_evals }} evals total</p>
        </div>
        <div class="bg-white border border-slate-200 rounded-xl px-4 py-4">
          <p class="text-xs font-medium text-slate-400 uppercase tracking-wide">Replan trigger rate</p>
          <p class="text-2xl font-semibold text-orange-600 mt-1">{{ pct(report.replan.trigger_rate) }}</p>
          <p class="text-xs text-slate-400 mt-1">{{ report.replan.tasks_with_replan }} tasks replanned</p>
        </div>
      </div>

      <!-- Panels grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">

        <!-- Memory panel -->
        <div class="bg-white border border-slate-200 rounded-xl overflow-hidden">
          <div class="px-4 py-3 border-b border-slate-200">
            <h3 class="text-sm font-medium text-slate-700">Memory Recall</h3>
          </div>
          <div class="px-4 py-4 space-y-3">
            <div class="flex items-center justify-between">
              <span class="text-sm text-slate-500">Recall rate</span>
              <span class="text-sm font-semibold text-slate-800">{{ pct(report.memory.recall_rate) }}</span>
            </div>
            <div class="w-full bg-slate-100 rounded-full h-2">
              <div class="bg-blue-500 h-2 rounded-full" :style="{ width: pct(report.memory.recall_rate) }"></div>
            </div>
            <div class="grid grid-cols-3 gap-2 pt-1">
              <div class="text-center">
                <p class="text-lg font-semibold text-slate-800">{{ report.memory.total_recalls }}</p>
                <p class="text-xs text-slate-400">Total recalls</p>
              </div>
              <div class="text-center">
                <p class="text-lg font-semibold text-slate-800">{{ report.memory.tasks_with_recall }}</p>
                <p class="text-xs text-slate-400">Tasks with recall</p>
              </div>
              <div class="text-center">
                <p class="text-lg font-semibold text-slate-800">{{ fmtK(Math.round(report.memory.avg_chars_injected)) }}</p>
                <p class="text-xs text-slate-400">Avg chars injected</p>
              </div>
            </div>
          </div>
        </div>

        <!-- Self-eval panel -->
        <div class="bg-white border border-slate-200 rounded-xl overflow-hidden">
          <div class="px-4 py-3 border-b border-slate-200">
            <h3 class="text-sm font-medium text-slate-700">Self-Evaluation</h3>
          </div>
          <div class="px-4 py-4 space-y-3">
            <div class="flex items-center justify-between">
              <span class="text-sm text-slate-500">Catch rate (FAIL / real evals)</span>
              <span class="text-sm font-semibold text-amber-600">{{ pct(report.self_eval.catch_rate) }}</span>
            </div>
            <div class="w-full bg-slate-100 rounded-full h-2">
              <div class="bg-amber-500 h-2 rounded-full" :style="{ width: pct(report.self_eval.catch_rate) }"></div>
            </div>
            <div class="grid grid-cols-3 gap-2 pt-1">
              <div class="text-center">
                <p class="text-lg font-semibold text-green-600">{{ report.self_eval.pass_count }}</p>
                <p class="text-xs text-slate-400">PASS</p>
              </div>
              <div class="text-center">
                <p class="text-lg font-semibold text-red-600">{{ report.self_eval.fail_count }}</p>
                <p class="text-xs text-slate-400">FAIL (caught)</p>
              </div>
              <div class="text-center">
                <p class="text-lg font-semibold text-slate-400">{{ report.self_eval.auto_pass_count }}</p>
                <p class="text-xs text-slate-400">Auto-pass</p>
              </div>
            </div>
          </div>
        </div>

        <!-- Replan panel -->
        <div class="bg-white border border-slate-200 rounded-xl overflow-hidden">
          <div class="px-4 py-3 border-b border-slate-200">
            <h3 class="text-sm font-medium text-slate-700">Replanning</h3>
          </div>
          <div class="px-4 py-4 space-y-3">
            <div class="flex items-center justify-between">
              <span class="text-sm text-slate-500">Trigger rate</span>
              <span class="text-sm font-semibold text-orange-600">{{ pct(report.replan.trigger_rate) }}</span>
            </div>
            <div class="w-full bg-slate-100 rounded-full h-2">
              <div class="bg-orange-500 h-2 rounded-full" :style="{ width: pct(report.replan.trigger_rate) }"></div>
            </div>
            <div class="grid grid-cols-3 gap-2 pt-1">
              <div class="text-center">
                <p class="text-lg font-semibold text-slate-800">{{ report.replan.tasks_with_replan }}</p>
                <p class="text-xs text-slate-400">Tasks replanned</p>
              </div>
              <div class="text-center">
                <p class="text-lg font-semibold text-slate-800">{{ report.replan.tasks_completed_after_replan }}</p>
                <p class="text-xs text-slate-400">Completed after</p>
              </div>
              <div class="text-center">
                <p class="text-lg font-semibold text-green-600">{{ pct(report.replan.success_rate_after_replan) }}</p>
                <p class="text-xs text-slate-400">Success rate</p>
              </div>
            </div>
          </div>
        </div>

        <!-- Context budget panel -->
        <div class="bg-white border border-slate-200 rounded-xl overflow-hidden">
          <div class="px-4 py-3 border-b border-slate-200">
            <h3 class="text-sm font-medium text-slate-700">Context Budget Utilization</h3>
            <p class="text-xs text-slate-400 mt-0.5">Prompt length (chars) as proxy for context window usage</p>
          </div>
          <div class="px-4 py-4">
            <div v-if="report.context_budget.sample_count === 0" class="text-center py-4 text-slate-400 text-sm">
              No prompt_built events in this window
            </div>
            <div v-else class="grid grid-cols-3 gap-2">
              <div class="text-center">
                <p class="text-lg font-semibold text-slate-800">{{ fmtK(report.context_budget.p50_prompt_length) }}</p>
                <p class="text-xs text-slate-400">p50</p>
              </div>
              <div class="text-center">
                <p class="text-lg font-semibold text-slate-800">{{ fmtK(report.context_budget.p90_prompt_length) }}</p>
                <p class="text-xs text-slate-400">p90</p>
              </div>
              <div class="text-center">
                <p class="text-lg font-semibold text-slate-800">{{ fmtK(report.context_budget.max_prompt_length) }}</p>
                <p class="text-xs text-slate-400">max</p>
              </div>
              <div class="text-center col-span-3 pt-1 border-t border-slate-100 mt-2">
                <p class="text-xs text-slate-400">{{ report.context_budget.sample_count }} samples · avg {{ fmtK(report.context_budget.avg_prompt_length) }} chars</p>
              </div>
            </div>
          </div>
        </div>

        <!-- Specialization distribution -->
        <div class="bg-white border border-slate-200 rounded-xl overflow-hidden">
          <div class="px-4 py-3 border-b border-slate-200">
            <h3 class="text-sm font-medium text-slate-700">Specialization Profile Distribution</h3>
            <p class="text-xs text-slate-400 mt-0.5">Current snapshot from agent activity files</p>
          </div>
          <div class="px-4 py-4">
            <div v-if="specializationRows.length === 0" class="text-center py-4 text-slate-400 text-sm">
              No active agents
            </div>
            <div v-else class="space-y-2">
              <div
                v-for="row in specializationRows"
                :key="row.profile"
                class="flex items-center gap-3"
              >
                <span class="text-sm text-slate-700 w-32 truncate capitalize">{{ row.profile }}</span>
                <div class="flex-1 bg-slate-100 rounded-full h-1.5">
                  <div
                    class="bg-blue-400 h-1.5 rounded-full"
                    :style="{ width: row.pct }"
                  ></div>
                </div>
                <span class="text-sm text-slate-500 w-12 text-right">{{ row.count }} ({{ row.pct }})</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Debate panel (stubbed) -->
        <div class="bg-white border border-slate-200 rounded-xl overflow-hidden opacity-60">
          <div class="px-4 py-3 border-b border-slate-200 flex items-center gap-2">
            <h3 class="text-sm font-medium text-slate-700">Debate Usage</h3>
            <span class="text-xs bg-slate-100 text-slate-400 px-2 py-0.5 rounded-full">Coming soon</span>
          </div>
          <div class="px-4 py-8 text-center text-slate-400 text-sm">
            {{ report.debate.note }}
          </div>
        </div>

      </div>
    </template>
  </div>
</template>
