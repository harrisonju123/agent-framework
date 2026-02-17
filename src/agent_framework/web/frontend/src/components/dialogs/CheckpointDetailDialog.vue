<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useAppState } from '../../composables/useAppState'
import Dialog from 'primevue/dialog'
import Button from 'primevue/button'
import Textarea from 'primevue/textarea'
import type { CheckpointData } from '../../types'

const props = defineProps<{
  checkpoint: CheckpointData | null
}>()

const emit = defineEmits<{
  close: []
}>()

const { approveCheckpoint, rejectCheckpoint, loading, showToast, apiError } = useAppState()

const visible = defineModel<boolean>('visible', { default: false })
const expanded = ref(false)
const rejecting = ref(false)
const feedback = ref('')

const dialogStyle = computed(() =>
  expanded.value
    ? { width: '90vw', height: '85vh' }
    : { width: '520px' }
)

watch(visible, (val) => {
  if (!val) {
    expanded.value = false
    rejecting.value = false
    feedback.value = ''
  }
})

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

async function handleReject() {
  if (!props.checkpoint || !feedback.value.trim()) return
  const result = await rejectCheckpoint(props.checkpoint.id, feedback.value.trim())
  if (result?.success) {
    visible.value = false
    emit('close')
    showToast(`Checkpoint rejected — feedback sent to ${props.checkpoint.assigned_to}`, 'info')
  } else if (apiError.value) {
    showToast(apiError.value, 'error')
  }
}
</script>

<template>
  <Dialog
    v-model:visible="visible"
    modal
    :style="dialogStyle"
    :contentStyle="expanded ? { flex: '1', overflow: 'hidden', display: 'flex', flexDirection: 'column' } : {}"
  >
    <template #header>
      <div class="flex items-center gap-2 w-full">
        <span class="font-semibold text-lg">Checkpoint Review</span>
        <button
          class="ml-auto p-1 rounded hover:bg-slate-100 text-slate-500 hover:text-slate-700 transition-colors"
          :title="expanded ? 'Collapse' : 'Expand'"
          @click="expanded = !expanded"
        >
          <svg v-if="!expanded" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="15 3 21 3 21 9" /><polyline points="9 21 3 21 3 15" />
            <line x1="21" y1="3" x2="14" y2="10" /><line x1="3" y1="21" x2="10" y2="14" />
          </svg>
          <svg v-else xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="4 14 10 14 10 20" /><polyline points="20 10 14 10 14 4" />
            <line x1="14" y1="10" x2="21" y2="3" /><line x1="3" y1="21" x2="10" y2="14" />
          </svg>
        </button>
      </div>
    </template>

    <div v-if="checkpoint" class="space-y-4" :class="expanded ? 'flex flex-col flex-1 min-h-0' : ''">
      <div>
        <label class="block text-xs font-medium text-slate-500 mb-1">Task</label>
        <p class="text-sm text-slate-700">{{ checkpoint.title }}</p>
      </div>
      <div>
        <label class="block text-xs font-medium text-slate-500 mb-1">Checkpoint</label>
        <p class="text-sm text-amber-700 font-mono">{{ checkpoint.checkpoint_id }}</p>
      </div>
      <div :class="expanded ? 'flex-1 min-h-0 flex flex-col' : ''">
        <label class="block text-xs font-medium text-slate-500 mb-1">Message</label>
        <p
          class="text-sm text-slate-700 whitespace-pre-wrap bg-slate-50 rounded-lg p-3 border border-slate-200"
          :class="expanded ? 'flex-1 overflow-y-auto' : 'max-h-64 overflow-y-auto'"
        >{{ checkpoint.checkpoint_message }}</p>
      </div>
      <div class="flex items-center gap-4 text-xs text-slate-500">
        <span>Agent: <span class="text-slate-700">{{ checkpoint.assigned_to }}</span></span>
        <span>Task ID: <span class="font-mono text-slate-700">{{ checkpoint.id }}</span></span>
      </div>

      <!-- Reject feedback area -->
      <div v-if="rejecting" class="space-y-2 pt-2 border-t border-slate-200">
        <label class="block text-xs font-medium text-slate-500">Feedback for the agent</label>
        <Textarea
          v-model="feedback"
          rows="4"
          class="w-full"
          placeholder="Explain what needs to change..."
          autofocus
        />
        <div class="flex justify-end gap-2">
          <Button label="Cancel" severity="secondary" text size="small" @click="rejecting = false; feedback = ''" />
          <Button
            label="Send Back"
            severity="danger"
            size="small"
            :disabled="loading || !feedback.trim()"
            @click="handleReject"
          />
        </div>
      </div>

      <!-- Default footer -->
      <div v-else class="flex justify-end gap-2 pt-2 border-t border-slate-200">
        <Button label="Reject" severity="danger" text @click="rejecting = true" />
        <Button label="Approve" severity="warn" :disabled="loading" @click="handleApprove" />
      </div>
    </div>
  </Dialog>
</template>
