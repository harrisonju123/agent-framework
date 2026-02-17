<script setup lang="ts">
import { ref, computed } from 'vue'
import { useAppState } from '../../composables/useAppState'
import Dialog from 'primevue/dialog'
import InputText from 'primevue/inputtext'
import Textarea from 'primevue/textarea'
import Select from 'primevue/select'
import InputNumber from 'primevue/inputnumber'
import Button from 'primevue/button'

const visible = defineModel<boolean>('visible', { default: false })
const { createTask, loading, apiError, showToast, agents: agentsState } = useAppState()

const form = ref({
  title: '',
  description: '',
  task_type: 'implementation',
  assigned_to: 'engineer',
  repository: '',
  priority: 1,
})

const taskTypes = [
  { label: 'Implementation', value: 'implementation' },
  { label: 'Fix', value: 'fix' },
  { label: 'Enhancement', value: 'enhancement' },
  { label: 'Planning', value: 'planning' },
  { label: 'Review', value: 'review' },
  { label: 'Documentation', value: 'documentation' },
  { label: 'Testing', value: 'testing' },
]

const agentOptions = computed(() =>
  agentsState.value.map((a: any) => ({ label: a.name, value: a.queue }))
)

const repoPattern = /^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_.-]+$/

const isValid = computed(() =>
  form.value.title.trim().length > 0 && form.value.description.trim().length > 0
)

const repoError = computed(() =>
  form.value.repository && !repoPattern.test(form.value.repository)
    ? 'Repository must be in owner/repo format'
    : ''
)

function resetForm() {
  form.value = {
    title: '',
    description: '',
    task_type: 'implementation',
    assigned_to: 'engineer',
    repository: '',
    priority: 1,
  }
}

async function submit() {
  if (!isValid.value || repoError.value) return
  const result = await createTask({
    title: form.value.title,
    description: form.value.description,
    task_type: form.value.task_type,
    assigned_to: form.value.assigned_to,
    ...(form.value.repository ? { repository: form.value.repository } : {}),
    priority: form.value.priority,
  })
  if (result?.success) {
    visible.value = false
    resetForm()
    showToast(result.message, 'success')
  } else if (apiError.value) {
    showToast(apiError.value, 'error')
  }
}
</script>

<template>
  <Dialog v-model:visible="visible" header="Add Task" modal :style="{ width: '520px' }">
    <form @submit.prevent="submit" class="space-y-4">
      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1" for="task-title">Title</label>
        <InputText
          id="task-title"
          v-model="form.title"
          class="w-full"
          placeholder="Task title"
        />
      </div>
      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1" for="task-desc">Description</label>
        <Textarea
          id="task-desc"
          v-model="form.description"
          rows="3"
          class="w-full"
          placeholder="What should be done..."
        />
      </div>
      <div class="grid grid-cols-2 gap-4">
        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1" for="task-type">Type</label>
          <Select
            id="task-type"
            v-model="form.task_type"
            :options="taskTypes"
            optionLabel="label"
            optionValue="value"
            class="w-full"
          />
        </div>
        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1" for="task-agent">Assign to</label>
          <Select
            id="task-agent"
            v-model="form.assigned_to"
            :options="agentOptions"
            optionLabel="label"
            optionValue="value"
            class="w-full"
          />
        </div>
      </div>
      <div class="grid grid-cols-2 gap-4">
        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1" for="task-repo">Repository (optional)</label>
          <InputText
            id="task-repo"
            v-model="form.repository"
            class="w-full"
            :class="{ 'p-invalid': repoError }"
            placeholder="owner/repo"
          />
          <small v-if="repoError" class="text-red-600">{{ repoError }}</small>
        </div>
        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1" for="task-priority">Priority</label>
          <InputNumber
            id="task-priority"
            v-model="form.priority"
            :min="1"
            :max="10"
            class="w-full"
          />
        </div>
      </div>
      <div class="flex justify-end gap-2 pt-2">
        <Button label="Cancel" severity="secondary" text @click="visible = false" />
        <Button type="submit" label="Create" :disabled="!isValid || !!repoError || loading" />
      </div>
    </form>
  </Dialog>
</template>
