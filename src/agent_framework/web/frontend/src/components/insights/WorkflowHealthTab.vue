<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { Chart } from 'chart.js'
import type {
  ChainMetricsReport,
  DecompositionReport,
  ReviewCycleMetricsReport,
  VerdictMetricsReport,
} from '../../types'
import {
  buildStepSuccessChart,
  buildVerdictDoughnutChart,
  buildSubtaskDistributionChart,
} from '../../composables/useInsightsCharts'
import InsightsKpiStrip from './InsightsKpiStrip.vue'

const props = defineProps<{ hours: number }>()

const chainReport = ref<ChainMetricsReport | null>(null)
const decompReport = ref<DecompositionReport | null>(null)
const reviewReport = ref<ReviewCycleMetricsReport | null>(null)
const verdictReport = ref<VerdictMetricsReport | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

const stepSuccessCanvas = ref<HTMLCanvasElement | null>(null)
const verdictCanvas = ref<HTMLCanvasElement | null>(null)
const subtaskCanvas = ref<HTMLCanvasElement | null>(null)
let stepSuccessChart: Chart | null = null
let verdictChart: Chart | null = null
let subtaskChart: Chart | null = null

function pct(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`
}

const kpiCards = computed(() => [
  {
    label: 'Chain Completion',
    value: chainReport.value ? pct(chainReport.value.chain_completion_rate) : '--',
    subtitle: chainReport.value ? `${chainReport.value.completed_chains}/${chainReport.value.total_chains} chains` : '',
    color: chainReport.value && chainReport.value.chain_completion_rate < 0.5 ? 'text-red-600' : 'text-green-600',
  },
  {
    label: 'Decomposition Rate',
    value: decompReport.value ? pct(decompReport.value.rate.decomposition_rate) : '--',
    subtitle: decompReport.value ? `${decompReport.value.rate.tasks_decomposed} decomposed` : '',
  },
  {
    label: 'Review Enforcement',
    value: reviewReport.value ? pct(reviewReport.value.metrics.enforcement_rate) : '--',
    subtitle: reviewReport.value ? `${reviewReport.value.metrics.cap_violations} cap violations` : '',
    color: reviewReport.value && reviewReport.value.metrics.cap_violations > 0 ? 'text-red-600' : 'text-green-600',
  },
  {
    label: 'Verdict Fallback',
    value: verdictReport.value ? pct(verdictReport.value.distribution.fallback_rate) : '--',
    subtitle: 'keyword fallback rate',
    color: verdictReport.value && verdictReport.value.distribution.fallback_rate > 0.2 ? 'text-amber-600' : 'text-green-600',
  },
])

async function fetchData() {
  loading.value = true
  error.value = null
  try {
    const [chainRes, decompRes, reviewRes, verdictRes] = await Promise.allSettled([
      fetch(`/api/chain-metrics?hours=${props.hours}`),
      fetch(`/api/decomposition-metrics?hours=${props.hours}`),
      fetch(`/api/review-cycle-metrics?hours=${props.hours}`),
      fetch(`/api/verdict-metrics?hours=${props.hours}`),
    ])
    if (chainRes.status === 'fulfilled' && chainRes.value.ok) {
      chainReport.value = await chainRes.value.json()
    }
    if (decompRes.status === 'fulfilled' && decompRes.value.ok) {
      decompReport.value = await decompRes.value.json()
    }
    if (reviewRes.status === 'fulfilled' && reviewRes.value.ok) {
      reviewReport.value = await reviewRes.value.json()
    }
    if (verdictRes.status === 'fulfilled' && verdictRes.value.ok) {
      verdictReport.value = await verdictRes.value.json()
    }
    if (!chainReport.value && !decompReport.value && !reviewReport.value && !verdictReport.value) {
      error.value = 'Failed to load workflow data'
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
  if (stepSuccessCanvas.value && chainReport.value && chainReport.value.step_type_metrics.length > 0) {
    stepSuccessChart?.destroy()
    const cfg = buildStepSuccessChart(chainReport.value.step_type_metrics)
    stepSuccessChart = new Chart(stepSuccessCanvas.value, { type: 'bar', data: cfg.data, options: cfg.options })
  }
  if (verdictCanvas.value && verdictReport.value && verdictReport.value.distribution.total_verdicts > 0) {
    verdictChart?.destroy()
    const cfg = buildVerdictDoughnutChart(verdictReport.value.distribution.by_method)
    verdictChart = new Chart(verdictCanvas.value, { type: 'doughnut', data: cfg.data, options: cfg.options })
  }
  if (subtaskCanvas.value && decompReport.value && Object.keys(decompReport.value.distribution.distribution).length > 0) {
    subtaskChart?.destroy()
    const cfg = buildSubtaskDistributionChart(decompReport.value.distribution.distribution)
    subtaskChart = new Chart(subtaskCanvas.value, { type: 'bar', data: cfg.data, options: cfg.options })
  }
}

onMounted(() => { fetchData() })
onBeforeUnmount(() => {
  stepSuccessChart?.destroy()
  verdictChart?.destroy()
  subtaskChart?.destroy()
})
watch(() => props.hours, () => { fetchData() })
</script>

<template>
  <div class="space-y-6 pt-4">
    <div v-if="loading && !chainReport" class="text-center py-16 text-slate-400">
      <span class="pi pi-spin pi-spinner mr-2"></span> Loading workflow data...
    </div>

    <div v-else-if="error && !chainReport" class="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
      {{ error }}
    </div>

    <template v-else>
      <InsightsKpiStrip :cards="kpiCards" />

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- Step Success Rates -->
        <div class="bg-white border border-slate-200 rounded-xl p-4">
          <h3 class="text-sm font-medium text-slate-600 mb-3">Step Success Rates</h3>
          <div v-if="chainReport && chainReport.step_type_metrics.length > 0" class="h-48">
            <canvas ref="stepSuccessCanvas"></canvas>
          </div>
          <div v-else class="h-48 flex items-center justify-center text-slate-400 text-sm">
            No chain step data
          </div>
        </div>

        <!-- Verdict Method Distribution -->
        <div class="bg-white border border-slate-200 rounded-xl p-4">
          <h3 class="text-sm font-medium text-slate-600 mb-3">Verdict Methods</h3>
          <div v-if="verdictReport && verdictReport.distribution.total_verdicts > 0" class="h-48">
            <canvas ref="verdictCanvas"></canvas>
          </div>
          <div v-else class="h-48 flex items-center justify-center text-slate-400 text-sm">
            No verdict data
          </div>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- Subtask Distribution -->
        <div class="bg-white border border-slate-200 rounded-xl p-4">
          <h3 class="text-sm font-medium text-slate-600 mb-3">Subtask Distribution</h3>
          <div v-if="decompReport && Object.keys(decompReport.distribution.distribution).length > 0" class="h-48">
            <canvas ref="subtaskCanvas"></canvas>
          </div>
          <div v-else class="h-48 flex items-center justify-center text-slate-400 text-sm">
            No decomposition data
          </div>
        </div>

        <!-- Estimation Accuracy -->
        <div class="bg-white border border-slate-200 rounded-xl p-4">
          <h3 class="text-sm font-medium text-slate-600 mb-3">Estimation Accuracy</h3>
          <div v-if="decompReport && decompReport.estimation.sample_count > 0">
            <div class="grid grid-cols-2 gap-4 text-center">
              <div>
                <p class="text-xs text-slate-400">Avg Estimated</p>
                <p class="text-lg font-semibold text-slate-700">{{ decompReport.estimation.avg_estimated.toFixed(0) }} lines</p>
              </div>
              <div>
                <p class="text-xs text-slate-400">Avg Actual</p>
                <p class="text-lg font-semibold text-slate-700">{{ decompReport.estimation.avg_actual.toFixed(0) }} lines</p>
              </div>
              <div>
                <p class="text-xs text-slate-400">Avg Ratio</p>
                <p class="text-lg font-semibold" :class="decompReport.estimation.avg_ratio > 1.5 ? 'text-red-600' : 'text-slate-700'">{{ decompReport.estimation.avg_ratio.toFixed(2) }}x</p>
              </div>
              <div>
                <p class="text-xs text-slate-400">Samples</p>
                <p class="text-lg font-semibold text-slate-700">{{ decompReport.estimation.sample_count }}</p>
              </div>
            </div>
          </div>
          <div v-else class="h-32 flex items-center justify-center text-slate-400 text-sm">
            No estimation data
          </div>
        </div>
      </div>

      <!-- Step Duration Table -->
      <div v-if="chainReport && chainReport.step_type_metrics.length > 0" class="bg-white border border-slate-200 rounded-xl p-4">
        <h3 class="text-sm font-medium text-slate-600 mb-3">Step Duration Summary</h3>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="text-left text-slate-400 text-xs uppercase border-b border-slate-100">
                <th class="pb-2 pr-4">Step</th>
                <th class="pb-2 pr-4">Count</th>
                <th class="pb-2 pr-4">Success Rate</th>
                <th class="pb-2 pr-4">Avg Duration</th>
                <th class="pb-2 pr-4">p50</th>
                <th class="pb-2">p90</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="step in chainReport.step_type_metrics" :key="step.step_id" class="border-b border-slate-50">
                <td class="py-2 pr-4 text-slate-700 font-medium">{{ step.step_id }}</td>
                <td class="py-2 pr-4 text-slate-600">{{ step.total_count }}</td>
                <td class="py-2 pr-4" :class="step.success_rate < 0.5 ? 'text-red-600' : 'text-green-600'">{{ pct(step.success_rate) }}</td>
                <td class="py-2 pr-4 text-slate-600">{{ step.avg_duration_seconds.toFixed(0) }}s</td>
                <td class="py-2 pr-4 text-slate-600">{{ step.p50_duration_seconds.toFixed(0) }}s</td>
                <td class="py-2 text-slate-600">{{ step.p90_duration_seconds.toFixed(0) }}s</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Review Cycle by Step -->
      <div v-if="reviewReport && reviewReport.metrics.by_step.length > 0" class="bg-white border border-slate-200 rounded-xl p-4">
        <h3 class="text-sm font-medium text-slate-600 mb-3">Review Cycles by Step</h3>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="text-left text-slate-400 text-xs uppercase border-b border-slate-100">
                <th class="pb-2 pr-4">Step</th>
                <th class="pb-2 pr-4">Checks</th>
                <th class="pb-2 pr-4">Enforcements</th>
                <th class="pb-2">Phase Resets</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="step in reviewReport.metrics.by_step" :key="step.workflow_step" class="border-b border-slate-50">
                <td class="py-2 pr-4 text-slate-700 font-medium">{{ step.workflow_step }}</td>
                <td class="py-2 pr-4 text-slate-600">{{ step.checks }}</td>
                <td class="py-2 pr-4 text-slate-600">{{ step.enforcements }}</td>
                <td class="py-2 text-slate-600">{{ step.phase_resets }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div v-if="reviewReport.metrics.cap_violations > 0" class="mt-3 bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-sm text-red-700">
          <span class="font-medium">{{ reviewReport.metrics.cap_violations }} cap violation(s)</span>
          in tasks: {{ reviewReport.metrics.violation_task_ids.join(', ') }}
        </div>
      </div>
    </template>
  </div>
</template>
