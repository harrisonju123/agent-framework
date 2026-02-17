<script setup lang="ts">
import type { QueueStats } from '../types'

defineProps<{
  queues: QueueStats[]
}>()
</script>

<template>
  <div class="bg-white shadow-sm border border-slate-200 rounded-xl p-5">
    <h2 class="text-lg font-semibold text-slate-800 mb-4">Queue Status</h2>

    <div v-if="queues.length === 0" class="text-slate-400 text-sm">
      No queues configured
    </div>

    <div v-else class="space-y-3">
      <div
        v-for="queue in queues"
        :key="queue.queue_id"
        class="flex items-center justify-between"
      >
        <span class="text-blue-600 font-medium text-sm">{{ queue.agent_name }}</span>
        <span
          class="font-mono text-sm"
          :class="queue.pending_count > 0 ? 'text-amber-600 font-medium' : 'text-slate-400'"
        >
          {{ queue.pending_count }} pending
        </span>
      </div>
    </div>

    <div class="mt-4 pt-3 border-t border-slate-200">
      <div class="flex items-center justify-between text-sm">
        <span class="text-slate-500">Total</span>
        <span class="font-mono text-slate-800 font-medium">
          {{ queues.reduce((sum, q) => sum + q.pending_count, 0) }}
        </span>
      </div>
    </div>
  </div>
</template>
