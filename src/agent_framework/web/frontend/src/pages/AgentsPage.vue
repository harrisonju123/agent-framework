<script setup lang="ts">
import { ref } from 'vue'
import { useAppState } from '../composables/useAppState'
import AgentCard from '../components/AgentCard.vue'
import AgentTable from '../components/AgentTable.vue'
import SelectButton from 'primevue/selectbutton'

const { agents, handleRestart } = useAppState()

const viewMode = ref('cards')
const viewOptions = [
  { label: 'Cards', value: 'cards' },
  { label: 'Table', value: 'table' },
]
</script>

<template>
  <div class="space-y-4">
    <div class="flex items-center justify-between">
      <h2 class="text-lg font-semibold text-slate-800">All Agents</h2>
      <SelectButton v-model="viewMode" :options="viewOptions" optionLabel="label" optionValue="value" />
    </div>

    <!-- Card View -->
    <div v-if="viewMode === 'cards'" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      <AgentCard
        v-for="agent in agents"
        :key="agent.id"
        :agent="agent"
        :on-restart="handleRestart"
      />
    </div>

    <!-- Table View -->
    <div v-else class="bg-white shadow-sm border border-slate-200 rounded-xl overflow-hidden">
      <AgentTable :agents="agents" :on-restart="handleRestart" />
    </div>

    <div v-if="agents.length === 0" class="text-center py-8 text-slate-400">
      No agents configured
    </div>
  </div>
</template>
