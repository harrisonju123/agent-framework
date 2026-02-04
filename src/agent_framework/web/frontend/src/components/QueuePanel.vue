<script setup lang="ts">
import type { QueueStats } from '../types'

defineProps<{
  queues: QueueStats[]
}>()
</script>

<template>
  <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
    <h2 class="text-lg font-semibold mb-4 text-gray-200">Queue Status</h2>

    <div v-if="queues.length === 0" class="text-gray-500 text-sm">
      No queues configured
    </div>

    <div v-else class="space-y-3">
      <div
        v-for="queue in queues"
        :key="queue.queue_id"
        class="flex items-center justify-between"
      >
        <span class="text-cyan-400">{{ queue.agent_name }}</span>
        <span
          class="font-mono text-sm"
          :class="queue.pending_count > 0 ? 'text-yellow-400' : 'text-gray-500'"
        >
          {{ queue.pending_count }} pending
        </span>
      </div>
    </div>

    <div class="mt-4 pt-3 border-t border-gray-700">
      <div class="flex items-center justify-between text-sm">
        <span class="text-gray-400">Total</span>
        <span class="font-mono text-gray-200">
          {{ queues.reduce((sum, q) => sum + q.pending_count, 0) }}
        </span>
      </div>
    </div>
  </div>
</template>
