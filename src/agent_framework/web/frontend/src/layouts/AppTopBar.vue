<script setup lang="ts">
import { ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAppState } from '../composables/useAppState'
import Button from 'primevue/button'
import Menu from 'primevue/menu'
import NewWorkDialog from '../components/dialogs/NewWorkDialog.vue'
import AnalyzeRepoDialog from '../components/dialogs/AnalyzeRepoDialog.vue'
import RunTicketDialog from '../components/dialogs/RunTicketDialog.vue'

const route = useRoute()
const router = useRouter()
const {
  connected, isPaused, loading,
  setupComplete,
  showWorkDialog, showAnalyzeDialog, showTicketDialog,
  handlePause, handleStart, handleStop,
  showConfirm,
} = useAppState()

const menu = ref()
const menuItems = ref([
  { label: 'New Work', icon: 'pi pi-plus', command: () => { showWorkDialog.value = true } },
  { label: 'Analyze Repo', icon: 'pi pi-search', command: () => { showAnalyzeDialog.value = true } },
  { label: 'Run Ticket', icon: 'pi pi-ticket', command: () => { showTicketDialog.value = true } },
])

function toggleMenu(event: Event) {
  menu.value.toggle(event)
}
</script>

<template>
  <header class="bg-white border-b border-slate-200 px-6 py-3 shrink-0">
    <div class="flex items-center justify-between">
      <!-- Page title -->
      <h1 class="text-2xl font-semibold text-slate-900">
        {{ (route.meta as any).title || 'Dashboard' }}
      </h1>

      <!-- Actions -->
      <div class="flex items-center gap-3">
        <!-- Connection status -->
        <span class="flex items-center gap-1.5 text-sm" :class="connected ? 'text-emerald-600' : 'text-red-500'">
          <span class="w-2 h-2 rounded-full" :class="connected ? 'bg-emerald-500' : 'bg-red-500'"></span>
          {{ connected ? 'Connected' : 'Disconnected' }}
        </span>

        <!-- Paused badge -->
        <span v-if="isPaused" class="px-2 py-1 bg-amber-50 text-amber-700 text-xs font-medium rounded-md border border-amber-200">
          PAUSED
        </span>

        <div class="w-px h-6 bg-slate-200"></div>

        <!-- Pause/Resume -->
        <Button
          :label="isPaused ? 'Resume' : 'Pause'"
          :severity="isPaused ? 'success' : 'warn'"
          size="small"
          outlined
          @click="handlePause"
        />

        <!-- Start All -->
        <Button
          label="Start All"
          severity="info"
          size="small"
          :disabled="loading"
          @click="handleStart"
        />

        <!-- Stop All -->
        <Button
          label="Stop All"
          severity="danger"
          size="small"
          outlined
          :disabled="loading"
          @click="showConfirm('Stop All Agents', 'Are you sure you want to stop all agents? This will terminate all running tasks.', handleStop, true)"
        />

        <div class="w-px h-6 bg-slate-200"></div>

        <!-- New action dropdown -->
        <Button
          label="+ New"
          size="small"
          @click="toggleMenu"
          aria-haspopup="true"
        />
        <Menu ref="menu" :model="menuItems" :popup="true" />

        <!-- Setup button -->
        <Button
          v-if="!setupComplete"
          label="Setup"
          icon="pi pi-cog"
          severity="success"
          size="small"
          @click="router.push('/settings')"
        />
      </div>
    </div>
  </header>

  <!-- Dialogs -->
  <NewWorkDialog v-model:visible="showWorkDialog" />
  <AnalyzeRepoDialog v-model:visible="showAnalyzeDialog" />
  <RunTicketDialog v-model:visible="showTicketDialog" />
</template>
