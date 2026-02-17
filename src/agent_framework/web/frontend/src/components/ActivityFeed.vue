<script setup lang="ts">
import type { ActivityEvent } from '../types'

defineProps<{
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
  <div class="text-sm">
    <div v-if="events.length === 0" class="px-4 py-3 text-slate-400 text-center">
      No recent activity
    </div>

    <div v-else>
      <div
        v-for="(event, index) in events"
        :key="index"
        class="px-4 py-2 border-b border-slate-100 last:border-b-0 hover:bg-slate-50 flex items-start gap-2"
      >
        <!-- Event icon -->
        <span
          class="flex-shrink-0 mt-0.5 pi text-xs"
          :class="{
            'pi-check-circle text-emerald-500': event.type === 'complete',
            'pi-times-circle text-red-500': event.type === 'fail',
            'pi-play text-blue-500': event.type === 'start',
            'pi-circle text-slate-400': event.type === 'phase',
          }"
          aria-hidden="true"
        ></span>

        <!-- Event content -->
        <div class="flex-1 min-w-0">
          <div class="flex items-baseline gap-2 text-xs">
            <span class="text-slate-400">
              {{ formatTime(event.timestamp) }}
            </span>
            <span class="text-blue-600 font-medium">{{ event.agent }}</span>
            <span v-if="event.type === 'complete' && event.duration_ms" class="text-slate-400">
              {{ formatDuration(event.duration_ms) }}
            </span>
            <span v-if="event.type === 'fail' && event.retry_count" class="text-red-500">
              x{{ event.retry_count }}
            </span>
          </div>
          <p class="text-xs text-slate-600 truncate" :title="event.title">
            {{ truncateTitle(event.title) }}
          </p>
        </div>
      </div>
    </div>
  </div>
</template>
