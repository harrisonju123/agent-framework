<script setup lang="ts">
import { ref, computed } from 'vue'
import { useAppState } from '../../composables/useAppState'
import Dialog from 'primevue/dialog'
import InputText from 'primevue/inputtext'
import Select from 'primevue/select'
import Button from 'primevue/button'

const visible = defineModel<boolean>('visible', { default: false })
const { runTicket, loading, apiError, showToast } = useAppState()

const form = ref({ ticket_id: '', agent: '' })
const ticketPattern = /^[A-Z]+-\d+$/

const agentOptions = [
  { label: 'Auto-assign based on ticket type', value: '' },
  { label: 'Architect', value: 'architect' },
  { label: 'Engineer', value: 'engineer' },
  { label: 'QA', value: 'qa' },
]

const isValid = computed(() => ticketPattern.test(form.value.ticket_id))

const ticketError = computed(() =>
  form.value.ticket_id && !ticketPattern.test(form.value.ticket_id)
    ? 'Ticket must be in PROJ-123 format'
    : ''
)

async function submit() {
  if (!isValid.value) return
  const result = await runTicket({
    ticket_id: form.value.ticket_id,
    agent: form.value.agent || undefined,
  })
  if (result?.success) {
    visible.value = false
    form.value = { ticket_id: '', agent: '' }
    showToast(result.message, 'success')
  } else if (apiError.value) {
    showToast(apiError.value, 'error')
  }
}
</script>

<template>
  <Dialog v-model:visible="visible" header="Run JIRA Ticket" modal :style="{ width: '440px' }">
    <form @submit.prevent="submit" class="space-y-4">
      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1" for="ticket-id">Ticket ID (e.g., PROJ-123)</label>
        <InputText
          id="ticket-id"
          v-model="form.ticket_id"
          class="w-full uppercase"
          :class="{ 'p-invalid': ticketError }"
          placeholder="PROJ-123"
        />
        <small v-if="ticketError" class="text-red-600">{{ ticketError }}</small>
      </div>

      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1" for="ticket-agent">Agent (optional)</label>
        <Select
          id="ticket-agent"
          v-model="form.agent"
          :options="agentOptions"
          optionLabel="label"
          optionValue="value"
          class="w-full"
        />
        <small class="text-slate-400">Leave empty to auto-assign based on JIRA ticket type</small>
      </div>

      <div class="flex justify-end gap-2 pt-2">
        <Button label="Cancel" severity="secondary" text @click="visible = false" />
        <Button type="submit" label="Run" :disabled="!isValid || loading" />
      </div>
    </form>
  </Dialog>
</template>
