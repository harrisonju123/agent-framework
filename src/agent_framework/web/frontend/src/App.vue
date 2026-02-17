<script setup lang="ts">
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { provideAppState } from './composables/useAppState'
import { useKeyboard } from './composables/useKeyboard'
import AppLayout from './layouts/AppLayout.vue'
import Toast from 'primevue/toast'
import ConfirmDialog from 'primevue/confirmdialog'

const appState = provideAppState()
const router = useRouter()

useKeyboard({
  onStart: () => appState.handleStart(),
  onStop: () => appState.showConfirm('Stop All Agents', 'Are you sure you want to stop all agents?', () => appState.handleStop(), true),
  onPause: () => appState.handlePause(),
  onWork: () => { appState.showWorkDialog.value = true },
  onAnalyze: () => { appState.showAnalyzeDialog.value = true },
  onTicket: () => { appState.showTicketDialog.value = true },
  onRetry: () => appState.handleRetryAll(),
  onApprove: () => appState.handleApproveAll(),
  onEscape: () => {},
  onNavigate: (page: number) => {
    const routes = ['/', '/agents', '/tasks', '/logs', '/settings']
    if (page >= 1 && page <= routes.length) {
      router.push(routes[page - 1])
    }
  },
})

onMounted(() => appState.checkSetupStatus())
</script>

<template>
  <AppLayout />
  <Toast position="top-right" />
  <ConfirmDialog />
</template>
