<script setup lang="ts">
import { computed } from 'vue'
import type { ActivityEvent } from '../types'

const props = defineProps<{
  events: ActivityEvent[]
}>()

function formatTime(timestamp: string): string {
  const date = new Date(timestamp)
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

function formatDuration(ms: number | null): string {
  if (!ms) return ''
  const seconds = Math.floor(ms / 1000)
  const minutes = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${minutes}m ${secs}s`
}

function truncateTitle(title: string, maxLen: number = 40): string {
  return title.length > maxLen ? title.slice(0, maxLen) + '...' : title
}
</script>

<template>
  <div class="font-mono text-sm">
    <div v-if="events.length === 0" class="px-4 py-3 text-gray-500 text-center">
      No recent activity
    </div>

    <div v-else>
      <div
        v-for="(event, index) in events"
        :key="index"
        class="px-4 py-2 border-b border-gray-800/50 last:border-b-0 hover:bg-gray-800/30 flex items-start gap-2"
      >
        <!-- Event icon -->
        <span
          class="flex-shrink-0 mt-0.5 text-xs"
          :class="{
            'text-green-500': event.type === 'complete',
            'text-red-500': event.type === 'fail',
            'text-cyan-500': event.type === 'start',
            'text-gray-500': event.type === 'phase',
          }"
          aria-hidden="true"
        >
          <template v-if="event.type === 'complete'">+</template>
          <template v-else-if="event.type === 'fail'">!</template>
          <template v-else-if="event.type === 'start'">></template>
          <template v-else>.</template>
        </span>

        <!-- Event content -->
        <div class="flex-1 min-w-0">
          <div class="flex items-baseline gap-2 text-xs">
            <span class="text-gray-600">
              {{ formatTime(event.timestamp) }}
            </span>
            <span class="text-cyan-400">{{ event.agent }}</span>
            <span v-if="event.type === 'complete' && event.duration_ms" class="text-gray-500">
              {{ formatDuration(event.duration_ms) }}
            </span>
            <span v-if="event.type === 'fail' && event.retry_count" class="text-red-400">
              x{{ event.retry_count }}
            </span>
          </div>
          <p class="text-xs text-gray-400 truncate" :title="event.title">
            {{ truncateTitle(event.title) }}
          </p>
        </div>
      </div>
    </div>
  </div>
</template>
