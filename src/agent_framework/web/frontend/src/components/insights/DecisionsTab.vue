<script setup lang="ts">
import { computed, ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { Chart } from 'chart.js'
import type { AgenticMetricsReport } from '../../types'
import InsightsKpiStrip from './InsightsKpiStrip.vue'
import { buildConfidenceBarChart } from '../../composables/useInsightsCharts'

const props = defineProps<{ report: AgenticMetricsReport }>()

function pct(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`
}

const highPct = computed(() => {
  const d = props.report.debate
  if (d.total_debates === 0) return '0.0%'
  const high = d.confidence_distribution['high'] || 0
  return pct(high / d.total_debates)
})

const kpiCards = computed(() => {
  const d = props.report.debate
  return [
    { label: 'Total debates', value: `${d.total_debates}`, subtitle: `${d.successful_debates} successful` },
    { label: 'Success rate', value: pct(d.success_rate), subtitle: 'of all debates', color: 'text-purple-600' },
    { label: 'High confidence', value: highPct.value, subtitle: 'of debates', color: 'text-green-600' },
    { label: 'Avg trade-offs', value: d.avg_trade_offs_count.toFixed(1), subtitle: 'per debate' },
  ]
})

const confCanvas = ref<HTMLCanvasElement | null>(null)
let confChart: Chart | null = null

function renderCharts() {
  if (confCanvas.value && props.report.debate.total_debates > 0) {
    confChart?.destroy()
    const cfg = buildConfidenceBarChart(props.report.debate.confidence_distribution)
    confChart = new Chart(confCanvas.value, { type: 'bar', data: cfg.data, options: cfg.options })
  }
}

onMounted(() => nextTick(renderCharts))
onBeforeUnmount(() => { confChart?.destroy() })
watch(() => props.report, () => nextTick(renderCharts))
</script>

<template>
  <div class="space-y-4">
    <template v-if="!report.debate.available">
      <div class="bg-white border border-slate-200 rounded-xl px-6 py-12 text-center text-slate-400">
        <span class="pi pi-info-circle mr-2"></span>
        No debates directory found. Debates are logged when agents use the structured debate system.
      </div>
    </template>

    <template v-else-if="report.debate.total_debates === 0">
      <div class="bg-white border border-slate-200 rounded-xl px-6 py-12 text-center text-slate-400">
        <span class="pi pi-comments mr-2"></span>
        No debates in this time window.
      </div>
    </template>

    <template v-else>
      <InsightsKpiStrip :cards="kpiCards" />

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <!-- Success rate + stats -->
        <div class="bg-white border border-slate-200 rounded-xl overflow-hidden">
          <div class="px-4 py-3 border-b border-slate-200">
            <h3 class="text-sm font-medium text-slate-700">Debate Outcomes</h3>
          </div>
          <div class="px-4 py-4 space-y-3">
            <div class="flex items-center justify-between">
              <span class="text-sm text-slate-500">Success rate</span>
              <span class="text-sm font-semibold text-purple-600">{{ pct(report.debate.success_rate) }}</span>
            </div>
            <div class="w-full bg-slate-100 rounded-full h-2">
              <div class="bg-purple-500 h-2 rounded-full" :style="{ width: pct(report.debate.success_rate) }"></div>
            </div>
            <div class="grid grid-cols-3 gap-2 pt-1">
              <div class="text-center">
                <p class="text-lg font-semibold text-slate-800">{{ report.debate.total_debates }}</p>
                <p class="text-xs text-slate-400">Total</p>
              </div>
              <div class="text-center">
                <p class="text-lg font-semibold text-green-600">{{ report.debate.successful_debates }}</p>
                <p class="text-xs text-slate-400">Successful</p>
              </div>
              <div class="text-center">
                <p class="text-lg font-semibold text-slate-800">{{ report.debate.avg_trade_offs_count.toFixed(1) }}</p>
                <p class="text-xs text-slate-400">Avg trade-offs</p>
              </div>
            </div>
          </div>
        </div>

        <!-- Confidence distribution chart -->
        <div class="bg-white border border-slate-200 rounded-xl overflow-hidden">
          <div class="px-4 py-3 border-b border-slate-200">
            <h3 class="text-sm font-medium text-slate-700">Confidence Distribution</h3>
          </div>
          <div class="px-4 py-4 h-48">
            <canvas ref="confCanvas"></canvas>
          </div>
        </div>
      </div>
    </template>
  </div>
</template>
