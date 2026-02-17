<script setup lang="ts">
import { useAppState } from '../composables/useAppState'
import AppSidebar from './AppSidebar.vue'
import AppTopBar from './AppTopBar.vue'

const { wsError, reconnecting, reconnectAttempt, reconnect, state } = useAppState()
</script>

<template>
  <div class="h-screen flex overflow-hidden bg-slate-50">
    <AppSidebar />

    <div class="flex-1 flex flex-col min-w-0">
      <AppTopBar />

      <!-- Connection error -->
      <div v-if="wsError && !reconnecting" class="flex flex-col items-center justify-center flex-1 gap-4">
        <div class="text-red-600 text-sm">{{ wsError }}</div>
        <button
          @click="reconnect"
          class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-lg transition-colors"
        >
          Reconnect
        </button>
      </div>

      <!-- Reconnecting state -->
      <div v-else-if="reconnecting" class="flex items-center justify-center flex-1">
        <div class="text-amber-600 text-sm animate-pulse">
          Reconnecting ({{ reconnectAttempt }}/10)...
        </div>
      </div>

      <!-- Loading state -->
      <div v-else-if="!state" class="flex items-center justify-center flex-1">
        <div class="text-slate-400 text-sm">Connecting...</div>
      </div>

      <!-- Main content -->
      <main v-else class="flex-1 min-h-0 overflow-y-auto p-6">
        <router-view />
      </main>
    </div>
  </div>
</template>
