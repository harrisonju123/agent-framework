<script setup lang="ts">
import type { CheckpointData } from '../types'

const props = defineProps<{
  checkpoints: CheckpointData[]
  onSelect: (checkpoint: CheckpointData) => void
}>()

function truncateText(text: string, maxLen: number = 50): string {
  return text.length > maxLen ? text.slice(0, maxLen) + '...' : text
}
</script>

<template>
  <div class="font-mono text-sm">
    <div v-if="checkpoints.length === 0" class="px-4 py-3 text-gray-500 text-center">
      No pending checkpoints
    </div>

    <div v-else>
      <div
        v-for="cp in checkpoints"
        :key="cp.id"
        class="px-4 py-2 border-b border-gray-800/50 last:border-b-0 hover:bg-gray-800/30 cursor-pointer"
        @click="onSelect(cp)"
      >
        <div class="flex items-center justify-between mb-1">
          <span class="font-mono text-yellow-400 font-medium text-xs">
            {{ cp.checkpoint_id }}
          </span>
          <span class="text-xs text-gray-600">click to review</span>
        </div>

        <p class="text-xs text-gray-400 truncate" :title="cp.title">
          {{ truncateText(cp.title, 60) }}
        </p>

        <div v-if="cp.checkpoint_message" class="text-xs text-yellow-400/60 mt-1 truncate" :title="cp.checkpoint_message">
          {{ truncateText(cp.checkpoint_message, 80) }}
        </div>
      </div>
    </div>
  </div>
</template>
