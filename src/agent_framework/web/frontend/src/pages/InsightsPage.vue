<script setup lang="ts">
import { ref, onMounted } from 'vue'
import type { AgenticMetricsReport } from '../types'

const report = ref<AgenticMetricsReport | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)
const hours = ref(24)

async function fetchMetrics() {
  loading.value = true
  error.value = null
  try {
    const resp = await fetch(`/api/metrics/agentics?hours=${hours.value}`)
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}))
      throw new Error(data.detail || `HTTP ${resp.status}`)
    }
    report.value = await resp.json()
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to load metrics'
  } finally {
    loading.value = false
  }
}

onMounted(fetchMetrics)
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-xl font-semibold text-slate-800">Agentic Insights</h1>
        <p class="text-sm text-slate-500 mt-0.5">Memory, self-eval, replan, and context budget metrics</p>
      </div>
      <div class="flex items-center gap-3">
        <select
          v-model.number="hours"
          @change="fetchMetrics"
          class="text-sm border border-slate-200 rounded-lg px-3 py-1.5 text-slate-700 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option :value="6">Last 6 hours</option>
          <option :value="24">Last 24 hours</option>
          <option :value="72">Last 3 days</option>
          <option :value="168">Last 7 days</option>
          <option :value="720">Last 30 days</option>
        </select>
        <button
          @click="fetchMetrics"
          :disabled="loading"
          class="text-sm px-3 py-1.5 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          <span class="pi pi-refresh text-xs mr-1" :class="{ 'animate-spin': loading }"></span>
          Refresh
        </button>
      </div>
    </div>

    <!-- Error state -->
    <div v-if="error" class="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
      {{ error }}
    </div>

    <!-- Loading skeleton -->
    <div v-else-if="loading && !report" class="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <div v-for="i in 4" :key="i" class="bg-white border border-slate-200 rounded-xl p-5 animate-pulse">
        <div class="h-4 bg-slate-200 rounded w-1/3 mb-4"></div>
        <div class="h-8 bg-slate-200 rounded w-1/2 mb-2"></div>
        <div class="h-3 bg-slate-100 rounded w-2/3"></div>
      </div>
    </div>

    <!-- Metrics grid -->
    <div v-else-if="report" class="grid grid-cols-1 sm:grid-cols-2 gap-4">

      <!-- Memory hit rate -->
      <div class="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div class="px-5 py-4 border-b border-slate-200 flex items-center gap-2">
          <span class="pi pi-database text-blue-500"></span>
          <h3 class="text-sm font-semibold text-slate-700">Memory Recall</h3>
        </div>
        <div class="p-5 space-y-4">
          <!-- Hit rate bar -->
          <div>
            <div class="flex items-baseline justify-between mb-1">
              <span class="text-3xl font-bold text-slate-800">{{ report.memory.hit_rate_pct }}%</span>
              <span class="text-xs text-slate-400">hit rate</span>
            </div>
            <div class="w-full bg-slate-100 rounded-full h-2">
              <div
                class="bg-blue-500 h-2 rounded-full transition-all"
                :style="{ width: `${Math.min(report.memory.hit_rate_pct, 100)}%` }"
              ></div>
            </div>
          </div>
          <div class="grid grid-cols-3 gap-3 text-center">
            <div class="bg-slate-50 rounded-lg p-2">
              <p class="text-xs text-slate-400">With recall</p>
              <p class="text-sm font-semibold text-slate-700">{{ report.memory.sessions_with_recall }}</p>
            </div>
            <div class="bg-slate-50 rounded-lg p-2">
              <p class="text-xs text-slate-400">Total sessions</p>
              <p class="text-sm font-semibold text-slate-700">{{ report.memory.total_sessions }}</p>
            </div>
            <div class="bg-slate-50 rounded-lg p-2">
              <p class="text-xs text-slate-400">Avg chars</p>
              <p class="text-sm font-semibold text-slate-700">{{ report.memory.avg_chars_injected.toFixed(0) }}</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Self-evaluation catch rate -->
      <div class="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div class="px-5 py-4 border-b border-slate-200 flex items-center gap-2">
          <span class="pi pi-check-circle text-emerald-500"></span>
          <h3 class="text-sm font-semibold text-slate-700">Self-Evaluation</h3>
        </div>
        <div class="p-5 space-y-4">
          <div>
            <div class="flex items-baseline justify-between mb-1">
              <span class="text-3xl font-bold text-slate-800">{{ report.self_eval.catch_rate_pct }}%</span>
              <span class="text-xs text-slate-400">catch rate</span>
            </div>
            <div class="w-full bg-slate-100 rounded-full h-2">
              <div
                class="bg-emerald-500 h-2 rounded-full transition-all"
                :style="{ width: `${Math.min(report.self_eval.catch_rate_pct, 100)}%` }"
              ></div>
            </div>
          </div>
          <div class="grid grid-cols-3 gap-3 text-center">
            <div class="bg-slate-50 rounded-lg p-2">
              <p class="text-xs text-slate-400">Auto-pass</p>
              <p class="text-sm font-semibold text-emerald-600">{{ report.self_eval.auto_pass_count }}</p>
            </div>
            <div class="bg-slate-50 rounded-lg p-2">
              <p class="text-xs text-slate-400">Pass</p>
              <p class="text-sm font-semibold text-blue-600">{{ report.self_eval.pass_count }}</p>
            </div>
            <div class="bg-slate-50 rounded-lg p-2">
              <p class="text-xs text-slate-400">Caught (fail)</p>
              <p class="text-sm font-semibold text-amber-600">{{ report.self_eval.fail_count }}</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Replan trigger rate and success -->
      <div class="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div class="px-5 py-4 border-b border-slate-200 flex items-center gap-2">
          <span class="pi pi-directions text-amber-500"></span>
          <h3 class="text-sm font-semibold text-slate-700">Replan</h3>
        </div>
        <div class="p-5 space-y-4">
          <div>
            <div class="flex items-baseline justify-between mb-1">
              <span class="text-3xl font-bold text-slate-800">{{ report.replan.replan_success_rate_pct }}%</span>
              <span class="text-xs text-slate-400">success after replan</span>
            </div>
            <div class="w-full bg-slate-100 rounded-full h-2">
              <div
                class="bg-amber-500 h-2 rounded-full transition-all"
                :style="{ width: `${Math.min(report.replan.replan_success_rate_pct, 100)}%` }"
              ></div>
            </div>
          </div>
          <div class="grid grid-cols-3 gap-3 text-center">
            <div class="bg-slate-50 rounded-lg p-2">
              <p class="text-xs text-slate-400">Total replans</p>
              <p class="text-sm font-semibold text-slate-700">{{ report.replan.total_replans }}</p>
            </div>
            <div class="bg-slate-50 rounded-lg p-2">
              <p class="text-xs text-slate-400">Tasks affected</p>
              <p class="text-sm font-semibold text-slate-700">{{ report.replan.tasks_with_replans }}</p>
            </div>
            <div class="bg-slate-50 rounded-lg p-2">
              <p class="text-xs text-slate-400">Completed</p>
              <p class="text-sm font-semibold text-emerald-600">{{ report.replan.tasks_completed_after_replan }}</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Context budget utilization -->
      <div class="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
        <div class="px-5 py-4 border-b border-slate-200 flex items-center gap-2">
          <span class="pi pi-chart-bar text-violet-500"></span>
          <h3 class="text-sm font-semibold text-slate-700">Context Budget</h3>
        </div>
        <div class="p-5 space-y-4">
          <div>
            <div class="flex items-baseline justify-between mb-1">
              <span class="text-3xl font-bold text-slate-800">{{ report.context_budget.avg_utilization_pct }}%</span>
              <span class="text-xs text-slate-400">avg utilization</span>
            </div>
            <div class="w-full bg-slate-100 rounded-full h-2">
              <div
                class="h-2 rounded-full transition-all"
                :class="report.context_budget.avg_utilization_pct > 90 ? 'bg-red-500' : report.context_budget.avg_utilization_pct > 75 ? 'bg-amber-500' : 'bg-violet-500'"
                :style="{ width: `${Math.min(report.context_budget.avg_utilization_pct, 100)}%` }"
              ></div>
            </div>
          </div>
          <!-- Stacked band bars — each band as a proportional row -->
          <div v-if="report.context_budget.total_completions > 0" class="space-y-1.5">
            <template v-for="band in [
              { label: '0–25%', count: report.context_budget.band_0_25_pct, color: 'bg-emerald-400' },
              { label: '25–50%', count: report.context_budget.band_25_50_pct, color: 'bg-blue-400' },
              { label: '50–75%', count: report.context_budget.band_50_75_pct, color: 'bg-amber-400' },
              { label: '75–100%', count: report.context_budget.band_75_100_pct, color: 'bg-orange-400' },
              { label: '>100%', count: report.context_budget.band_over_100_pct, color: 'bg-red-400' },
            ]" :key="band.label">
              <div class="flex items-center gap-2 text-xs">
                <span class="w-12 text-slate-400 text-right">{{ band.label }}</span>
                <div class="flex-1 bg-slate-100 rounded-full h-1.5">
                  <div
                    class="h-1.5 rounded-full transition-all"
                    :class="band.color"
                    :style="{ width: `${(band.count / report.context_budget.total_completions * 100).toFixed(1)}%` }"
                  ></div>
                </div>
                <span class="w-6 text-slate-500 font-medium">{{ band.count }}</span>
              </div>
            </template>
          </div>
          <p v-else class="text-xs text-slate-400 text-center">No completions in window</p>
        </div>
      </div>

    </div>

    <!-- Empty state (no data yet) -->
    <div v-else-if="!loading" class="text-center py-16 text-slate-400">
      <span class="pi pi-chart-line text-4xl block mb-3 opacity-30"></span>
      <p class="text-sm">No session data found for this time window.</p>
      <p class="text-xs mt-1">Agentic metrics are collected from <code>logs/sessions/*.jsonl</code>.</p>
    </div>

    <!-- Footer: last generated -->
    <p v-if="report" class="text-xs text-slate-400 text-right">
      Generated {{ new Date(report.generated_at).toLocaleString() }} · {{ report.time_range_hours }}h window
    </p>
  </div>
</template>
