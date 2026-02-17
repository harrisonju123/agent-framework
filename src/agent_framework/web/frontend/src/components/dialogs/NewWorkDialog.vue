<script setup lang="ts">
import { ref, computed } from 'vue'
import { useAppState } from '../../composables/useAppState'
import Dialog from 'primevue/dialog'
import InputText from 'primevue/inputtext'
import Textarea from 'primevue/textarea'
import Button from 'primevue/button'

const visible = defineModel<boolean>('visible', { default: false })
const { createWork, loading, apiError, showToast } = useAppState()

const form = ref({ goal: '', repository: '', workflow: 'default' })
const repoPattern = /^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_.-]+$/

const isValid = computed(() =>
  form.value.goal.trim().length > 0 && repoPattern.test(form.value.repository)
)

const repoError = computed(() =>
  form.value.repository && !repoPattern.test(form.value.repository)
    ? 'Repository must be in owner/repo format'
    : ''
)

async function submit() {
  if (!isValid.value) return
  const result = await createWork({
    goal: form.value.goal,
    repository: form.value.repository,
    workflow: form.value.workflow,
  })
  if (result?.success) {
    visible.value = false
    form.value = { goal: '', repository: '', workflow: 'default' }
    showToast(result.message, 'success')
  } else if (apiError.value) {
    showToast(apiError.value, 'error')
  }
}
</script>

<template>
  <Dialog v-model:visible="visible" header="New Work" modal :style="{ width: '480px' }">
    <form @submit.prevent="submit" class="space-y-4">
      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1" for="work-goal">Goal</label>
        <Textarea
          id="work-goal"
          v-model="form.goal"
          rows="3"
          class="w-full"
          placeholder="Describe what you want to build..."
        />
      </div>
      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1" for="work-repo">Repository (owner/repo)</label>
        <InputText
          id="work-repo"
          v-model="form.repository"
          class="w-full"
          :class="{ 'p-invalid': repoError }"
          placeholder="justworkshr/pto"
        />
        <small v-if="repoError" class="text-red-600">{{ repoError }}</small>
      </div>
      <div class="flex justify-end gap-2 pt-2">
        <Button label="Cancel" severity="secondary" text @click="visible = false" />
        <Button type="submit" label="Create" :disabled="!isValid || loading" />
      </div>
    </form>
  </Dialog>
</template>
