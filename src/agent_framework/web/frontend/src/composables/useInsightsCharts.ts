import {
  Chart,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  DoughnutController,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js'
import type { ChartData, ChartOptions } from 'chart.js'
import type {
  TrendBucket,
  CostTrendBucket,
  ModelTierMetrics,
  StepTypeMetrics,
  AgentPerformance,
  FailureCategory,
} from '../types'

let registered = false

export function initChartJs() {
  if (registered) return
  Chart.register(
    CategoryScale, LinearScale, PointElement, LineElement,
    BarElement, ArcElement, DoughnutController, Tooltip, Legend, Filler,
  )
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

// ============== Existing Charts (Agentic Insights) ==============

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

// ============== Cost & Efficiency Charts ==============

export function buildCostTrendChart(trends: CostTrendBucket[]): {
  data: ChartData<'line'>
  options: ChartOptions<'line'>
} {
  const labels = trends.map(t => formatHour(t.timestamp))
  return {
    data: {
      labels,
      datasets: [
        {
          label: 'Cost ($)',
          data: trends.map(t => t.total_cost),
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239,68,68,0.1)',
          fill: true,
          tension: 0.3,
        },
      ],
    },
    options: baseLineOptions('Cost ($)'),
  }
}

export function buildModelTierChart(tiers: ModelTierMetrics[]): {
  data: ChartData<'bar'>
  options: ChartOptions<'bar'>
} {
  return {
    data: {
      labels: tiers.map(t => t.tier),
      datasets: [
        {
          label: 'Cost ($)',
          data: tiers.map(t => t.total_cost),
          backgroundColor: ['#3b82f6', '#8b5cf6', '#f59e0b', '#22c55e', '#94a3b8'],
        },
      ],
    },
    options: baseBarOptions('y'),
  }
}

// ============== Workflow Health Charts ==============

export function buildStepSuccessChart(steps: StepTypeMetrics[]): {
  data: ChartData<'bar'>
  options: ChartOptions<'bar'>
} {
  return {
    data: {
      labels: steps.map(s => s.step_id),
      datasets: [
        { label: 'Success', data: steps.map(s => s.success_count), backgroundColor: '#22c55e' },
        { label: 'Failure', data: steps.map(s => s.failure_count), backgroundColor: '#ef4444' },
      ],
    },
    options: baseBarOptions('x'),
  }
}

export function buildVerdictDoughnutChart(byMethod: Record<string, number>): {
  data: ChartData<'doughnut'>
  options: ChartOptions<'doughnut'>
} {
  const entries = Object.entries(byMethod).filter(([, v]) => v > 0)
  const colors = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#94a3b8']
  return {
    data: {
      labels: entries.map(([k]) => k),
      datasets: [{
        data: entries.map(([, v]) => v),
        backgroundColor: colors.slice(0, entries.length),
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, padding: 16 } } },
    },
  }
}

export function buildSubtaskDistributionChart(distribution: Record<string, number>): {
  data: ChartData<'bar'>
  options: ChartOptions<'bar'>
} {
  const entries = Object.entries(distribution).sort(([a], [b]) => Number(a) - Number(b))
  return {
    data: {
      labels: entries.map(([k]) => `${k} subtasks`),
      datasets: [{
        label: 'Tasks',
        data: entries.map(([, v]) => v),
        backgroundColor: '#8b5cf6',
      }],
    },
    options: baseBarOptions('x'),
  }
}

// ============== Delivery Charts ==============

export function buildAgentPerformanceChart(agents: AgentPerformance[]): {
  data: ChartData<'bar'>
  options: ChartOptions<'bar'>
} {
  return {
    data: {
      labels: agents.map(a => a.agent_id),
      datasets: [
        { label: 'Completed', data: agents.map(a => a.completed_tasks), backgroundColor: '#22c55e' },
        { label: 'Failed', data: agents.map(a => a.failed_tasks), backgroundColor: '#ef4444' },
      ],
    },
    options: baseBarOptions('x'),
  }
}

export function buildFailureCategoriesChart(categories: FailureCategory[]): {
  data: ChartData<'bar'>
  options: ChartOptions<'bar'>
} {
  const sorted = [...categories].sort((a, b) => b.count - a.count).slice(0, 8)
  return {
    data: {
      labels: sorted.map(c => c.category),
      datasets: [{
        label: 'Count',
        data: sorted.map(c => c.count),
        backgroundColor: '#f59e0b',
      }],
    },
    options: baseBarOptions('y'),
  }
}
