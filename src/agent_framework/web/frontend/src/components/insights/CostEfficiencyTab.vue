<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { Chart } from 'chart.js'
import type { LlmMetricsReport, WasteMetricsReport } from '../../types'
import { buildCostTrendChart, buildModelTierChart } from '../../composables/useInsightsCharts'
import InsightsKpiStrip from './InsightsKpiStrip.vue'

const props = defineProps<{ hours: number }>()

const llmReport = ref<LlmMetricsReport | null>(null)
const wasteReport = ref<WasteMetricsReport | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)

const costTrendCanvas = ref<HTMLCanvasElement | null>(null)
const modelTierCanvas = ref<HTMLCanvasElement | null>(null)
let costTrendChart: Chart | null = null
let modelTierChart: Chart | null = null

function fmtCost(n: number): string {
  return `$${n.toFixed(2)}`
}

function pct(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`
}

const kpiCards = computed(() => {
  const llm = llmReport.value
  const waste = wasteReport.value
  return [
    {
      label: 'Total LLM Cost',
      value: llm ? fmtCost(llm.total_cost) : '--',
      subtitle: llm ? `${llm.total_llm_calls} calls` : '',
      color: 'text-red-600',
    },
    {
      label: 'Waste Ratio',
      value: waste ? pct(waste.aggregate_waste_ratio) : '--',
      subtitle: waste ? `${waste.roots_analyzed} roots` : '',
      color: waste && waste.aggregate_waste_ratio > 0.3 ? 'text-red-600' : 'text-green-600',
    },
    {
      label: 'Token Efficiency',
      value: llm ? `${llm.overall_token_efficiency.toFixed(2)}` : '--',
      subtitle: 'out / in ratio',
    },
    {
      label: 'Zero-Delivery Roots',
      value: waste ? `${waste.roots_with_zero_delivery}` : '--',
      subtitle: 'tasks with no PR',
      color: waste && waste.roots_with_zero_delivery > 0 ? 'text-red-600' : 'text-green-600',
    },
  ]
})

async function fetchData() {
  loading.value = true
  error.value = null
  try {
    const [llmRes, wasteRes] = await Promise.allSettled([
      fetch(`/api/llm-metrics?hours=${props.hours}`),
      fetch(`/api/waste-metrics?hours=${props.hours}`),
    ])
    if (llmRes.status === 'fulfilled' && llmRes.value.ok) {
      llmReport.value = await llmRes.value.json()
    }
    if (wasteRes.status === 'fulfilled' && wasteRes.value.ok) {
      wasteReport.value = await wasteRes.value.json()
    }
    if (!llmReport.value && !wasteReport.value) {
      error.value = 'Failed to load cost data'
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
  if (costTrendCanvas.value && llmReport.value && llmReport.value.trends.length > 0) {
    costTrendChart?.destroy()
    const cfg = buildCostTrendChart(llmReport.value.trends)
    costTrendChart = new Chart(costTrendCanvas.value, { type: 'line', data: cfg.data, options: cfg.options })
  }
  if (modelTierCanvas.value && llmReport.value && llmReport.value.model_tiers.length > 0) {
    modelTierChart?.destroy()
    const cfg = buildModelTierChart(llmReport.value.model_tiers)
    modelTierChart = new Chart(modelTierCanvas.value, { type: 'bar', data: cfg.data, options: cfg.options })
  }
}

onMounted(() => { fetchData() })
onBeforeUnmount(() => {
  costTrendChart?.destroy()
  modelTierChart?.destroy()
})
watch(() => props.hours, () => { fetchData() })
</script>

<template>
  <div class="space-y-6 pt-4">
    <div v-if="loading && !llmReport" class="text-center py-16 text-slate-400">
      <span class="pi pi-spin pi-spinner mr-2"></span> Loading cost data...
    </div>

    <div v-else-if="error && !llmReport" class="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
      {{ error }}
    </div>

    <template v-else>
      <InsightsKpiStrip :cards="kpiCards" />

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- Cost Trend -->
        <div class="bg-white border border-slate-200 rounded-xl p-4">
          <h3 class="text-sm font-medium text-slate-600 mb-3">Cost Trend</h3>
          <div v-if="llmReport && llmReport.trends.length > 0" class="h-48">
            <canvas ref="costTrendCanvas"></canvas>
          </div>
          <div v-else class="h-48 flex items-center justify-center text-slate-400 text-sm">
            No trend data available
          </div>
        </div>

        <!-- Model Tier Breakdown -->
        <div class="bg-white border border-slate-200 rounded-xl p-4">
          <h3 class="text-sm font-medium text-slate-600 mb-3">Cost by Model Tier</h3>
          <div v-if="llmReport && llmReport.model_tiers.length > 0" class="h-48">
            <canvas ref="modelTierCanvas"></canvas>
          </div>
          <div v-else class="h-48 flex items-center justify-center text-slate-400 text-sm">
            No model tier data
          </div>
        </div>
      </div>

      <!-- Latency Stats -->
      <div v-if="llmReport" class="bg-white border border-slate-200 rounded-xl p-4">
        <h3 class="text-sm font-medium text-slate-600 mb-3">Latency Percentiles</h3>
        <div class="grid grid-cols-2 sm:grid-cols-5 gap-4 text-center">
          <div>
            <p class="text-xs text-slate-400">Avg</p>
            <p class="text-lg font-semibold text-slate-700">{{ llmReport.latency.avg_ms.toFixed(0) }}ms</p>
          </div>
          <div>
            <p class="text-xs text-slate-400">p50</p>
            <p class="text-lg font-semibold text-slate-700">{{ llmReport.latency.p50_ms.toFixed(0) }}ms</p>
          </div>
          <div>
            <p class="text-xs text-slate-400">p90</p>
            <p class="text-lg font-semibold text-slate-700">{{ llmReport.latency.p90_ms.toFixed(0) }}ms</p>
          </div>
          <div>
            <p class="text-xs text-slate-400">p99</p>
            <p class="text-lg font-semibold text-slate-700">{{ llmReport.latency.p99_ms.toFixed(0) }}ms</p>
          </div>
          <div>
            <p class="text-xs text-slate-400">Max</p>
            <p class="text-lg font-semibold text-slate-700">{{ llmReport.latency.max_ms.toFixed(0) }}ms</p>
          </div>
        </div>
      </div>

      <!-- Top Waste Tasks -->
      <div v-if="wasteReport && wasteReport.top_waste_roots.length > 0" class="bg-white border border-slate-200 rounded-xl p-4">
        <h3 class="text-sm font-medium text-slate-600 mb-3">Top Waste Tasks</h3>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="text-left text-slate-400 text-xs uppercase border-b border-slate-100">
                <th class="pb-2 pr-4">Task</th>
                <th class="pb-2 pr-4">Total</th>
                <th class="pb-2 pr-4">Wasted</th>
                <th class="pb-2 pr-4">Ratio</th>
                <th class="pb-2">PR</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="task in wasteReport.top_waste_roots.slice(0, 10)" :key="task.root_task_id" class="border-b border-slate-50">
                <td class="py-2 pr-4 text-slate-700 truncate max-w-[240px]" :title="task.title">{{ task.title || task.root_task_id }}</td>
                <td class="py-2 pr-4 text-slate-600">{{ fmtCost(task.total_cost) }}</td>
                <td class="py-2 pr-4 text-red-600">{{ fmtCost(task.wasted_cost) }}</td>
                <td class="py-2 pr-4" :class="task.waste_ratio > 0.5 ? 'text-red-600 font-medium' : 'text-slate-600'">{{ pct(task.waste_ratio) }}</td>
                <td class="py-2">
                  <span :class="task.has_pr ? 'text-green-600' : 'text-slate-400'">{{ task.has_pr ? 'Yes' : 'No' }}</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </template>
  </div>
</template>
