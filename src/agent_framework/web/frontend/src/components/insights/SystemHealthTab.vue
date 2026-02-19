<script setup lang="ts">
import { computed, ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { Chart } from 'chart.js'
import type { AgenticMetricsReport } from '../../types'
import InsightsKpiStrip from './InsightsKpiStrip.vue'
import { buildBudgetTrendChart, buildSpecializationBarChart } from '../../composables/useInsightsCharts'

const props = defineProps<{ report: AgenticMetricsReport }>()

function fmtK(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`
}

const kpiCards = computed(() => {
  const cb = props.report.context_budget
  const sp = props.report.specialization
  return [
    { label: 'p50 prompt length', value: fmtK(cb.p50_prompt_length), subtitle: 'chars' },
    { label: 'p90 prompt length', value: fmtK(cb.p90_prompt_length), subtitle: 'chars' },
    { label: 'Active agents', value: `${sp.total_active_agents}`, subtitle: 'current snapshot' },
    { label: 'Prompt samples', value: `${cb.sample_count}`, subtitle: `avg ${fmtK(cb.avg_prompt_length)} chars` },
  ]
})

const specCanvas = ref<HTMLCanvasElement | null>(null)
const budgetCanvas = ref<HTMLCanvasElement | null>(null)
let specChart: Chart | null = null
let budgetChart: Chart | null = null

function renderCharts() {
  if (specCanvas.value && Object.keys(props.report.specialization.distribution).length > 0) {
    specChart?.destroy()
    const cfg = buildSpecializationBarChart(props.report.specialization.distribution)
    specChart = new Chart(specCanvas.value, { type: 'bar', data: cfg.data, options: cfg.options })
  }
  if (budgetCanvas.value && props.report.trends.length > 0) {
    budgetChart?.destroy()
    const cfg = buildBudgetTrendChart(props.report.trends)
    budgetChart = new Chart(budgetCanvas.value, { type: 'line', data: cfg.data, options: cfg.options })
  }
}

onMounted(() => nextTick(renderCharts))
onBeforeUnmount(() => {
  specChart?.destroy()
  budgetChart?.destroy()
})
watch(() => props.report, () => nextTick(renderCharts))
</script>

<template>
  <div class="space-y-4">
    <InsightsKpiStrip :cards="kpiCards" />

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <!-- Context Budget panel -->
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
              <p class="text-xs text-slate-400">
                {{ report.context_budget.sample_count }} samples · avg {{ fmtK(report.context_budget.avg_prompt_length) }} chars
              </p>
            </div>
          </div>
        </div>
      </div>

      <!-- Specialization panel -->
      <div class="bg-white border border-slate-200 rounded-xl overflow-hidden">
        <div class="px-4 py-3 border-b border-slate-200">
          <h3 class="text-sm font-medium text-slate-700">Specialization Profile Distribution</h3>
          <p class="text-xs text-slate-400 mt-0.5">Current snapshot from agent activity files</p>
        </div>
        <div class="px-4 py-4">
          <div v-if="Object.keys(report.specialization.distribution).length === 0" class="text-center py-4 text-slate-400 text-sm">
            No active agents
          </div>
          <div v-else class="h-40">
            <canvas ref="specCanvas"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- Budget Trend chart -->
    <div v-if="report.trends.length > 0" class="bg-white border border-slate-200 rounded-xl overflow-hidden">
      <div class="px-4 py-3 border-b border-slate-200">
        <h3 class="text-sm font-medium text-slate-700">Budget Trend</h3>
      </div>
      <div class="px-4 py-4 h-64">
        <canvas ref="budgetCanvas"></canvas>
      </div>
    </div>
  </div>
</template>
