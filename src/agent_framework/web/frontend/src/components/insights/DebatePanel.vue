<script setup lang="ts">
import type { DebateMetrics } from '../../types'

defineProps<{
  metrics: DebateMetrics | null | undefined
}>()
</script>

<template>
  <div class="bg-white shadow-sm border border-slate-200 rounded-xl p-5">
    <h2 class="text-lg font-semibold text-slate-800 mb-4">Multi-Agent Debates</h2>

    <!-- Empty state -->
    <div v-if="!metrics" class="text-slate-400 text-sm">
      No debate data available
    </div>

    <!-- Metrics loaded -->
    <div v-else class="space-y-4">
      <!-- Total count -->
      <div class="grid grid-cols-2 gap-4">
        <div>
          <p class="text-xs text-slate-500 uppercase tracking-wide">Total Debates</p>
          <p class="text-2xl font-bold text-slate-800">{{ metrics.total_debates }}</p>
        </div>
        <div>
          <p class="text-xs text-slate-500 uppercase tracking-wide">Sessions Scanned</p>
          <p class="text-2xl font-bold text-slate-800">{{ metrics.sessions_scanned }}</p>
        </div>
      </div>

      <!-- Debate stats -->
      <div class="pt-4 border-t border-slate-200">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm text-slate-600">Debate Frequency</p>
            <p class="text-xs text-slate-500 mt-1">
              {{ metrics.sessions_scanned > 0 ? (metrics.total_debates / metrics.sessions_scanned).toFixed(2) : '0' }} debates per session
            </p>
          </div>
          <div class="text-right">
            <p class="text-sm text-slate-600">Debate Rate</p>
            <p
              class="text-lg font-bold mt-1"
              :class="metrics.total_debates > 0 ? 'text-green-600' : 'text-slate-400'"
            >
              {{ metrics.sessions_scanned > 0 ? ((metrics.total_debates / metrics.sessions_scanned) * 100).toFixed(1) : '0' }}%
            </p>
          </div>
        </div>
      </div>

      <!-- Status indicator -->
      <div class="pt-4 border-t border-slate-200">
        <div class="flex items-center gap-2">
          <div
            class="w-3 h-3 rounded-full"
            :class="metrics.total_debates > 0 ? 'bg-green-500' : 'bg-slate-300'"
          ></div>
          <span class="text-sm text-slate-700">
            {{ metrics.total_debates > 0 ? 'Debates are occurring' : 'No debates recorded' }}
          </span>
        </div>
      </div>
    </div>
  </div>
</template>
