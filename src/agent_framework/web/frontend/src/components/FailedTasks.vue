<script setup lang="ts">
import type { FailedTask } from '../types'

const props = defineProps<{
  tasks: FailedTask[]
  onRetry: (taskId: string) => void
}>()

function truncateError(error: string | null, maxLen: number = 50): string {
  if (!error) return ''
  return error.length > maxLen ? error.slice(0, maxLen) + '...' : error
}

function getTaskKey(task: FailedTask): string {
  return task.jira_key || task.id.slice(0, 12)
}
</script>

<template>
  <div class="font-mono text-sm">
    <div v-if="tasks.length === 0" class="px-4 py-3 text-gray-500 text-center">
      No failed tasks
    </div>

    <div v-else>
      <div
        v-for="task in tasks"
        :key="task.id"
        class="px-4 py-2 border-b border-gray-800/50 last:border-b-0 hover:bg-gray-800/30"
      >
        <div class="flex items-center justify-between mb-1">
          <span class="font-mono text-red-400 font-medium text-xs">
            {{ getTaskKey(task) }}
          </span>
          <div class="flex items-center gap-2">
            <span class="text-xs text-gray-500" :title="`${task.retry_count} retry attempts`">
              x{{ task.retry_count }}
            </span>
            <button
              @click="onRetry(task.id)"
              class="px-2 py-0.5 text-xs bg-red-600/80 hover:bg-red-600 rounded transition-colors"
              :aria-label="`Retry task ${getTaskKey(task)}`"
            >
              Retry
            </button>
          </div>
        </div>

        <p class="text-xs text-gray-400 truncate" :title="task.title">
          {{ task.title }}
        </p>

        <div v-if="task.last_error" class="text-xs text-red-400/60 mt-1 truncate" :title="task.last_error">
          {{ truncateError(task.last_error, 60) }}
        </div>
      </div>
    </div>
  </div>
</template>
