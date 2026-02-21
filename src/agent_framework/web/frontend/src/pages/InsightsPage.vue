<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import Tabs from 'primevue/tabs'
import TabList from 'primevue/tablist'
import Tab from 'primevue/tab'
import TabPanels from 'primevue/tabpanels'
import TabPanel from 'primevue/tabpanel'
import type { AgenticMetricsReport } from '../types'
import { initChartJs } from '../composables/useInsightsCharts'
import AgentPerformanceTab from '../components/insights/AgentPerformanceTab.vue'
import SystemHealthTab from '../components/insights/SystemHealthTab.vue'
import DecisionsTab from '../components/insights/DecisionsTab.vue'
import CostEfficiencyTab from '../components/insights/CostEfficiencyTab.vue'
import WorkflowHealthTab from '../components/insights/WorkflowHealthTab.vue'
import DeliveryTab from '../components/insights/DeliveryTab.vue'

initChartJs()

const report = ref<AgenticMetricsReport | null>(null)
const loading = ref(false)
const error = ref<string | null>(null)
const hours = ref(24)
const activeTab = ref('performance')
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
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-xl font-semibold text-slate-800">Agentic Insights</h1>
        <p class="text-sm text-slate-500 mt-0.5">Memory, self-evaluation, replanning, context budget, and debate signals</p>
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

    <!-- Error banner (only for agentic-metrics, new tabs handle their own errors) -->
    <div v-if="error && ['performance', 'health', 'decisions'].includes(activeTab)" class="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-700">
      {{ error }}
    </div>

    <Tabs v-model:value="activeTab">
      <TabList>
        <Tab value="performance">Agent Performance</Tab>
        <Tab value="health">System Health</Tab>
        <Tab value="decisions">Decisions</Tab>
        <Tab value="cost">Cost & Efficiency</Tab>
        <Tab value="workflow">Workflow Health</Tab>
        <Tab value="delivery">Delivery</Tab>
      </TabList>

      <TabPanels>
        <!-- Existing tabs: gated on report loading -->
        <TabPanel value="performance">
          <div v-if="!report && activeTab === 'performance'" class="text-center py-16 text-slate-400">
            <span class="pi pi-spin pi-spinner mr-2"></span> Loading insights...
          </div>
          <AgentPerformanceTab v-else-if="report && activeTab === 'performance'" :report="report" />
        </TabPanel>
        <TabPanel value="health">
          <div v-if="!report && activeTab === 'health'" class="text-center py-16 text-slate-400">
            <span class="pi pi-spin pi-spinner mr-2"></span> Loading insights...
          </div>
          <SystemHealthTab v-else-if="report && activeTab === 'health'" :report="report" />
        </TabPanel>
        <TabPanel value="decisions">
          <div v-if="!report && activeTab === 'decisions'" class="text-center py-16 text-slate-400">
            <span class="pi pi-spin pi-spinner mr-2"></span> Loading insights...
          </div>
          <DecisionsTab v-else-if="report && activeTab === 'decisions'" :report="report" />
        </TabPanel>

        <!-- New self-fetching tabs: pass hours so they refetch on lookback change -->
        <TabPanel value="cost">
          <CostEfficiencyTab v-if="activeTab === 'cost'" :hours="hours" />
        </TabPanel>
        <TabPanel value="workflow">
          <WorkflowHealthTab v-if="activeTab === 'workflow'" :hours="hours" />
        </TabPanel>
        <TabPanel value="delivery">
          <DeliveryTab v-if="activeTab === 'delivery'" :hours="hours" />
        </TabPanel>
      </TabPanels>
    </Tabs>
  </div>
</template>
