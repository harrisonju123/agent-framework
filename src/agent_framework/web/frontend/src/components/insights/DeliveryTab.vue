<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { Chart } from 'chart.js'
import type {
  PerformanceReport,
  GitMetricsReport,
  FailureAnalysisReport,
} from '../../types'
import {
  buildAgentPerformanceChart,
  buildFailureCategoriesChart,
} from '../../composables/useInsightsCharts'
import InsightsKpiStrip from './InsightsKpiStrip.vue'

const props = defineProps<{ hours: number }>()

const perfReport = ref<PerformanceReport | null>(null)
const gitReport = ref<GitMetricsReport | null>(null)
const failureReport = ref<FailureAnalysisReport | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

const agentPerfCanvas = ref<HTMLCanvasElement | null>(null)
const failureCatCanvas = ref<HTMLCanvasElement | null>(null)
let agentPerfChart: Chart | null = null
let failureCatChart: Chart | null = null

function pct(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`
}

function fmtCost(n: number): string {
  return `$${n.toFixed(2)}`
}

const kpiCards = computed(() => [
  {
    label: 'Success Rate',
    value: perfReport.value ? pct(perfReport.value.overall_success_rate) : '--',
    subtitle: perfReport.value ? `${perfReport.value.total_tasks} tasks` : '',
    color: perfReport.value && perfReport.value.overall_success_rate < 0.5 ? 'text-red-600' : 'text-green-600',
  },
  {
    label: 'Avg Commits/Task',
    value: gitReport.value ? gitReport.value.summary.avg_commits_per_task.toFixed(1) : '--',
    subtitle: gitReport.value ? `${gitReport.value.total_tasks} tasks with git` : '',
  },
  {
    label: 'Push Success',
    value: gitReport.value ? pct(gitReport.value.summary.push_success_rate) : '--',
    subtitle: 'of push attempts',
  },
  {
    label: 'Total Failures',
    value: failureReport.value ? `${failureReport.value.total_failures}` : '--',
    subtitle: failureReport.value ? `${failureReport.value.categories.length} categories` : '',
    color: failureReport.value && failureReport.value.total_failures > 0 ? 'text-red-600' : 'text-green-600',
  },
])

async function fetchData() {
  loading.value = true
  error.value = null
  try {
    const [perfRes, gitRes, failRes] = await Promise.allSettled([
      fetch(`/api/performance-metrics?hours=${props.hours}`),
      fetch(`/api/git-metrics?hours=${props.hours}`),
      fetch(`/api/failure-analysis?hours=${Math.min(props.hours * 2, 720)}`),
    ])
    if (perfRes.status === 'fulfilled' && perfRes.value.ok) {
      perfReport.value = await perfRes.value.json()
    }
    if (gitRes.status === 'fulfilled' && gitRes.value.ok) {
      gitReport.value = await gitRes.value.json()
    }
    if (failRes.status === 'fulfilled' && failRes.value.ok) {
      failureReport.value = await failRes.value.json()
    }
    if (!perfReport.value && !gitReport.value && !failureReport.value) {
      error.value = 'Failed to load delivery data'
    }
  } catch (e) {
    error.value = e instanceof Error ? e.message : 'Failed to fetch'
  } finally {
    loading.value = false
    await nextTick()
    renderCharts()
  }
}

function renderCharts() {
  if (agentPerfCanvas.value && perfReport.value && perfReport.value.agent_performance.length > 0) {
    agentPerfChart?.destroy()
    const cfg = buildAgentPerformanceChart(perfReport.value.agent_performance)
    agentPerfChart = new Chart(agentPerfCanvas.value, { type: 'bar', data: cfg.data, options: cfg.options })
  }
  if (failureCatCanvas.value && failureReport.value && failureReport.value.categories.length > 0) {
    failureCatChart?.destroy()
    const cfg = buildFailureCategoriesChart(failureReport.value.categories)
    failureCatChart = new Chart(failureCatCanvas.value, { type: 'bar', data: cfg.data, options: cfg.options })
  }
}

onMounted(() => { fetchData() })
onBeforeUnmount(() => {
  agentPerfChart?.destroy()
  failureCatChart?.destroy()
})
watch(() => props.hours, () => { fetchData() })
</script>

<template>
  <div class="space-y-6 pt-4">
    <div v-if="loading && !perfReport" class="text-center py-16 text-slate-400">
      <span class="pi pi-spin pi-spinner mr-2"></span> Loading delivery data...
    </div>

    <div v-else-if="error && !perfReport" class="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
      {{ error }}
    </div>

    <template v-else>
      <InsightsKpiStrip :cards="kpiCards" />

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- Agent Performance -->
        <div class="bg-white border border-slate-200 rounded-xl p-4">
          <h3 class="text-sm font-medium text-slate-600 mb-3">Agent Performance</h3>
          <div v-if="perfReport && perfReport.agent_performance.length > 0" class="h-48">
            <canvas ref="agentPerfCanvas"></canvas>
          </div>
          <div v-else class="h-48 flex items-center justify-center text-slate-400 text-sm">
            No agent performance data
          </div>
        </div>

        <!-- Failure Categories -->
        <div class="bg-white border border-slate-200 rounded-xl p-4">
          <h3 class="text-sm font-medium text-slate-600 mb-3">Failure Categories</h3>
          <div v-if="failureReport && failureReport.categories.length > 0" class="h-48">
            <canvas ref="failureCatCanvas"></canvas>
          </div>
          <div v-else class="h-48 flex items-center justify-center text-slate-400 text-sm">
            No failure data
          </div>
        </div>
      </div>

      <!-- Agent Detail Table -->
      <div v-if="perfReport && perfReport.agent_performance.length > 0" class="bg-white border border-slate-200 rounded-xl p-4">
        <h3 class="text-sm font-medium text-slate-600 mb-3">Agent Breakdown</h3>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="text-left text-slate-400 text-xs uppercase border-b border-slate-100">
                <th class="pb-2 pr-4">Agent</th>
                <th class="pb-2 pr-4">Tasks</th>
                <th class="pb-2 pr-4">Success</th>
                <th class="pb-2 pr-4">Avg Duration</th>
                <th class="pb-2 pr-4">Cost</th>
                <th class="pb-2">Retry Rate</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="agent in perfReport.agent_performance" :key="agent.agent_id" class="border-b border-slate-50">
                <td class="py-2 pr-4 text-slate-700 font-medium">{{ agent.agent_id }}</td>
                <td class="py-2 pr-4 text-slate-600">{{ agent.total_tasks }}</td>
                <td class="py-2 pr-4" :class="agent.success_rate < 0.5 ? 'text-red-600' : 'text-green-600'">{{ pct(agent.success_rate) }}</td>
                <td class="py-2 pr-4 text-slate-600">{{ agent.avg_duration_seconds.toFixed(0) }}s</td>
                <td class="py-2 pr-4 text-slate-600">{{ fmtCost(agent.total_cost) }}</td>
                <td class="py-2" :class="agent.retry_rate > 0.3 ? 'text-amber-600' : 'text-slate-600'">{{ pct(agent.retry_rate) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Handoff Latency -->
      <div v-if="perfReport && perfReport.handoff_summaries.length > 0" class="bg-white border border-slate-200 rounded-xl p-4">
        <h3 class="text-sm font-medium text-slate-600 mb-3">Handoff Latency</h3>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="text-left text-slate-400 text-xs uppercase border-b border-slate-100">
                <th class="pb-2 pr-4">Transition</th>
                <th class="pb-2 pr-4">Count</th>
                <th class="pb-2 pr-4">Avg Total</th>
                <th class="pb-2 pr-4">p50</th>
                <th class="pb-2 pr-4">p90</th>
                <th class="pb-2">Delayed</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="h in perfReport.handoff_summaries" :key="h.transition" class="border-b border-slate-50">
                <td class="py-2 pr-4 text-slate-700 font-medium">{{ h.transition }}</td>
                <td class="py-2 pr-4 text-slate-600">{{ h.count }}</td>
                <td class="py-2 pr-4 text-slate-600">{{ (h.avg_total_ms / 1000).toFixed(1) }}s</td>
                <td class="py-2 pr-4 text-slate-600">{{ (h.p50_total_ms / 1000).toFixed(1) }}s</td>
                <td class="py-2 pr-4 text-slate-600">{{ (h.p90_total_ms / 1000).toFixed(1) }}s</td>
                <td class="py-2" :class="h.delayed_count > 0 ? 'text-amber-600' : 'text-slate-600'">{{ h.delayed_count }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Failure Trends -->
      <div v-if="failureReport && failureReport.trends.length > 0" class="bg-white border border-slate-200 rounded-xl p-4">
        <h3 class="text-sm font-medium text-slate-600 mb-3">Failure Trends</h3>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="text-left text-slate-400 text-xs uppercase border-b border-slate-100">
                <th class="pb-2 pr-4">Category</th>
                <th class="pb-2 pr-4">Weekly Count</th>
                <th class="pb-2">Trend</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="trend in failureReport.trends" :key="trend.category" class="border-b border-slate-50">
                <td class="py-2 pr-4 text-slate-700">{{ trend.category }}</td>
                <td class="py-2 pr-4 text-slate-600">{{ trend.weekly_count }}</td>
                <td class="py-2">
                  <span :class="trend.is_increasing ? 'text-red-600' : 'text-green-600'">
                    {{ trend.is_increasing ? '&#9650;' : '&#9660;' }} {{ Math.abs(trend.weekly_change_pct).toFixed(0) }}%
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Recommendations -->
      <div v-if="failureReport && failureReport.top_recommendations.length > 0" class="bg-white border border-slate-200 rounded-xl p-4">
        <h3 class="text-sm font-medium text-slate-600 mb-3">Recommendations</h3>
        <ul class="space-y-2">
          <li v-for="(rec, i) in failureReport.top_recommendations" :key="i" class="flex items-start gap-2 text-sm text-slate-700">
            <span class="text-blue-500 mt-0.5 shrink-0">&#8226;</span>
            {{ rec }}
          </li>
        </ul>
      </div>
    </template>
  </div>
</template>
