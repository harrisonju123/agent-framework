import {
  Chart,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js'
import type { ChartData, ChartOptions } from 'chart.js'
import type { TrendBucket } from '../types'

let registered = false

export function initChartJs() {
  if (registered) return
  Chart.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Tooltip, Legend, Filler)
  registered = true
}

function baseLineOptions(yLabel: string): ChartOptions<'line'> {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, padding: 16 } } },
    scales: {
      x: { grid: { display: false } },
      y: { beginAtZero: true, title: { display: true, text: yLabel } },
    },
  }
}

function baseBarOptions(indexAxis: 'x' | 'y' = 'x'): ChartOptions<'bar'> {
  return {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis,
    plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, padding: 16 } } },
    scales: {
      x: { grid: { display: indexAxis === 'y' }, stacked: indexAxis === 'x' },
      y: { beginAtZero: true, stacked: indexAxis === 'x', grid: { display: indexAxis !== 'y' } },
    },
  }
}

function formatHour(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export function buildPerformanceTrendChart(trends: TrendBucket[]): {
  data: ChartData<'line'>
  options: ChartOptions<'line'>
} {
  const labels = trends.map(t => formatHour(t.timestamp))
  return {
    data: {
      labels,
      datasets: [
        {
          label: 'Memory Recall',
          data: trends.map(t => t.memory_recall_rate * 100),
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59,130,246,0.1)',
          fill: false,
          tension: 0.3,
        },
        {
          label: 'Self-Eval Catch',
          data: trends.map(t => t.self_eval_catch_rate * 100),
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245,158,11,0.1)',
          fill: false,
          tension: 0.3,
        },
        {
          label: 'Replan Trigger',
          data: trends.map(t => t.replan_trigger_rate * 100),
          borderColor: '#f97316',
          backgroundColor: 'rgba(249,115,22,0.1)',
          fill: false,
          tension: 0.3,
        },
      ],
    },
    options: baseLineOptions('Rate (%)'),
  }
}

export function buildSelfEvalBarChart(pass: number, fail: number, autoPass: number): {
  data: ChartData<'bar'>
  options: ChartOptions<'bar'>
} {
  return {
    data: {
      labels: ['Self-Eval Outcomes'],
      datasets: [
        { label: 'Pass', data: [pass], backgroundColor: '#22c55e' },
        { label: 'Fail (Caught)', data: [fail], backgroundColor: '#ef4444' },
        { label: 'Auto-Pass', data: [autoPass], backgroundColor: '#94a3b8' },
      ],
    },
    options: baseBarOptions('x'),
  }
}

export function buildBudgetTrendChart(trends: TrendBucket[]): {
  data: ChartData<'line'>
  options: ChartOptions<'line'>
} {
  const labels = trends.map(t => formatHour(t.timestamp))
  return {
    data: {
      labels,
      datasets: [
        {
          label: 'Avg Prompt Length',
          data: trends.map(t => t.avg_prompt_length),
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99,102,241,0.15)',
          fill: true,
          tension: 0.3,
        },
      ],
    },
    options: baseLineOptions('Characters'),
  }
}

export function buildSpecializationBarChart(distribution: Record<string, number>): {
  data: ChartData<'bar'>
  options: ChartOptions<'bar'>
} {
  const entries = Object.entries(distribution).sort((a, b) => b[1] - a[1])
  return {
    data: {
      labels: entries.map(([k]) => k === 'none' ? 'Unspecialized' : k),
      datasets: [
        {
          label: 'Agents',
          data: entries.map(([, v]) => v),
          backgroundColor: '#60a5fa',
        },
      ],
    },
    options: baseBarOptions('y'),
  }
}

export function buildConfidenceBarChart(distribution: Record<string, number>): {
  data: ChartData<'bar'>
  options: ChartOptions<'bar'>
} {
  const high = distribution['high'] || 0
  const medium = distribution['medium'] || 0
  const low = distribution['low'] || 0
  return {
    data: {
      labels: ['Confidence'],
      datasets: [
        { label: 'High', data: [high], backgroundColor: '#22c55e' },
        { label: 'Medium', data: [medium], backgroundColor: '#f59e0b' },
        { label: 'Low', data: [low], backgroundColor: '#ef4444' },
      ],
    },
    options: baseBarOptions('x'),
  }
}
