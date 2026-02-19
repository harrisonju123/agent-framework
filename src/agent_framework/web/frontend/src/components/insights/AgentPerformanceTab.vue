<script setup lang="ts">
import { computed, ref, onMounted, onBeforeUnmount, watch } from 'vue'
import { Chart } from 'chart.js'
import type { AgenticMetricsReport } from '../../types'
import InsightsKpiStrip from './InsightsKpiStrip.vue'
import { buildPerformanceTrendChart, buildSelfEvalBarChart } from '../../composables/useInsightsCharts'

const props = defineProps<{ report: AgenticMetricsReport }>()

function pct(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`
}

function fmtK(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`
}

const kpiCards = computed(() => [
  { label: 'Tasks observed', value: `${props.report.total_observed_tasks}`, subtitle: `last ${props.report.time_range_hours}h` },
  { label: 'Memory recall rate', value: pct(props.report.memory.recall_rate), subtitle: `${props.report.memory.total_recalls} total recalls`, color: 'text-blue-700' },
  { label: 'Self-eval catch rate', value: pct(props.report.self_eval.catch_rate), subtitle: `${props.report.self_eval.total_evals} evals total`, color: 'text-amber-600' },
  { label: 'Replan trigger rate', value: pct(props.report.replan.trigger_rate), subtitle: `${props.report.replan.tasks_with_replan} tasks replanned`, color: 'text-orange-600' },
])

const trendCanvas = ref<HTMLCanvasElement | null>(null)
const seBarCanvas = ref<HTMLCanvasElement | null>(null)
let trendChart: Chart | null = null
let seChart: Chart | null = null

function renderCharts() {
  if (trendCanvas.value) {
    trendChart?.destroy()
    const cfg = buildPerformanceTrendChart(props.report.trends)
    trendChart = new Chart(trendCanvas.value, { type: 'line', data: cfg.data, options: cfg.options })
  }
  if (seBarCanvas.value) {
    seChart?.destroy()
    const se = props.report.self_eval
    const cfg = buildSelfEvalBarChart(se.pass_count, se.fail_count, se.auto_pass_count)
    seChart = new Chart(seBarCanvas.value, { type: 'bar', data: cfg.data, options: cfg.options })
  }
}

onMounted(renderCharts)
onBeforeUnmount(() => {
  trendChart?.destroy()
  seChart?.destroy()
})
watch(() => props.report, renderCharts)
</script>

<template>
  <div class="space-y-4">
    <InsightsKpiStrip :cards="kpiCards" />

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
          <!-- Usefulness comparison -->
          <div class="border-t border-slate-100 pt-3 mt-2">
            <p class="text-xs font-medium text-slate-400 uppercase tracking-wide mb-2">Recall Usefulness</p>
            <div class="grid grid-cols-3 gap-2">
              <div class="text-center">
                <p class="text-lg font-semibold text-slate-800">{{ pct(report.memory.completion_rate_with_recall) }}</p>
                <p class="text-xs text-slate-400">With recall</p>
              </div>
              <div class="text-center">
                <p class="text-lg font-semibold text-slate-800">{{ pct(report.memory.completion_rate_without_recall) }}</p>
                <p class="text-xs text-slate-400">Without recall</p>
              </div>
              <div class="text-center">
                <p
                  class="text-lg font-semibold"
                  :class="report.memory.recall_usefulness_delta >= 0 ? 'text-green-600' : 'text-red-600'"
                >
                  {{ report.memory.recall_usefulness_delta >= 0 ? '+' : '' }}{{ pct(report.memory.recall_usefulness_delta) }}
                </p>
                <p class="text-xs text-slate-400">Delta</p>
              </div>
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
          <div class="h-40">
            <canvas ref="seBarCanvas"></canvas>
          </div>
        </div>
      </div>

      <!-- Replan panel -->
      <div class="bg-white border border-slate-200 rounded-xl overflow-hidden lg:col-span-2">
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
    </div>

    <!-- Performance Trends chart -->
    <div v-if="report.trends.length > 0" class="bg-white border border-slate-200 rounded-xl overflow-hidden">
      <div class="px-4 py-3 border-b border-slate-200">
        <h3 class="text-sm font-medium text-slate-700">Performance Trends</h3>
      </div>
      <div class="px-4 py-4 h-64">
        <canvas ref="trendCanvas"></canvas>
      </div>
    </div>
  </div>
</template>
