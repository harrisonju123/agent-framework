<script setup lang="ts">
import { useAppState } from '../../composables/useAppState'
import Dialog from 'primevue/dialog'
import Button from 'primevue/button'
import type { CheckpointData } from '../../types'

const props = defineProps<{
  checkpoint: CheckpointData | null
}>()

const emit = defineEmits<{
  close: []
}>()

const { approveCheckpoint, loading, showToast, apiError } = useAppState()

const visible = defineModel<boolean>('visible', { default: false })

async function handleApprove() {
  if (!props.checkpoint) return
  const result = await approveCheckpoint(props.checkpoint.id)
  if (result?.success) {
    visible.value = false
    emit('close')
    showToast(`Checkpoint approved for ${props.checkpoint.id}`, 'success')
  } else if (apiError.value) {
    showToast(apiError.value, 'error')
  }
}
</script>

<template>
  <Dialog v-model:visible="visible" header="Checkpoint Review" modal :style="{ width: '520px' }">
    <div v-if="checkpoint" class="space-y-4">
      <div>
        <label class="block text-xs font-medium text-slate-500 mb-1">Task</label>
        <p class="text-sm text-slate-700">{{ checkpoint.title }}</p>
      </div>
      <div>
        <label class="block text-xs font-medium text-slate-500 mb-1">Checkpoint</label>
        <p class="text-sm text-amber-700 font-mono">{{ checkpoint.checkpoint_id }}</p>
      </div>
      <div>
        <label class="block text-xs font-medium text-slate-500 mb-1">Message</label>
        <p class="text-sm text-slate-700 whitespace-pre-wrap bg-slate-50 rounded-lg p-3 border border-slate-200">{{ checkpoint.checkpoint_message }}</p>
      </div>
      <div class="flex items-center gap-4 text-xs text-slate-500">
        <span>Agent: <span class="text-slate-700">{{ checkpoint.assigned_to }}</span></span>
        <span>Task ID: <span class="font-mono text-slate-700">{{ checkpoint.id }}</span></span>
      </div>
      <div class="flex justify-end gap-2 pt-2 border-t border-slate-200">
        <Button label="Cancel" severity="secondary" text @click="visible = false" />
        <Button label="Approve" severity="warn" :disabled="loading" @click="handleApprove" />
      </div>
    </div>
  </Dialog>
</template>
