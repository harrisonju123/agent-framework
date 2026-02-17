<script setup lang="ts">
import { ref, computed } from 'vue'
import { useAppState } from '../../composables/useAppState'
import Dialog from 'primevue/dialog'
import InputText from 'primevue/inputtext'
import InputNumber from 'primevue/inputnumber'
import Textarea from 'primevue/textarea'
import Select from 'primevue/select'
import Checkbox from 'primevue/checkbox'
import Button from 'primevue/button'

const visible = defineModel<boolean>('visible', { default: false })
const { analyzeRepo, loading, apiError, showToast } = useAppState()

const form = ref({
  repository: '',
  severity: 'high' as 'all' | 'critical' | 'high' | 'medium',
  max_issues: 50,
  dry_run: false,
  focus: '',
})

const repoPattern = /^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_.-]+$/
const severityOptions = [
  { label: 'All', value: 'all' },
  { label: 'Critical', value: 'critical' },
  { label: 'High', value: 'high' },
  { label: 'Medium', value: 'medium' },
]

const isValid = computed(() => repoPattern.test(form.value.repository))

const repoError = computed(() =>
  form.value.repository && !repoPattern.test(form.value.repository)
    ? 'Repository must be in owner/repo format'
    : ''
)

async function submit() {
  if (!isValid.value) return
  const result = await analyzeRepo({
    repository: form.value.repository,
    severity: form.value.severity,
    max_issues: form.value.max_issues,
    dry_run: form.value.dry_run,
    focus: form.value.focus || undefined,
  })
  if (result?.success) {
    visible.value = false
    form.value = { repository: '', severity: 'high', max_issues: 50, dry_run: false, focus: '' }
    showToast(result.message, 'success')
  } else if (apiError.value) {
    showToast(apiError.value, 'error')
  }
}
</script>

<template>
  <Dialog v-model:visible="visible" header="Analyze Repository" modal :style="{ width: '520px' }">
    <form @submit.prevent="submit" class="space-y-4">
      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1" for="analyze-repo">Repository (owner/repo)</label>
        <InputText
          id="analyze-repo"
          v-model="form.repository"
          class="w-full"
          :class="{ 'p-invalid': repoError }"
          placeholder="justworkshr/pto"
        />
        <small v-if="repoError" class="text-red-600">{{ repoError }}</small>
      </div>

      <div class="grid grid-cols-2 gap-4">
        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1" for="analyze-severity">Severity</label>
          <Select
            id="analyze-severity"
            v-model="form.severity"
            :options="severityOptions"
            optionLabel="label"
            optionValue="value"
            class="w-full"
          />
        </div>
        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1" for="analyze-max">Max Issues</label>
          <InputNumber
            id="analyze-max"
            v-model="form.max_issues"
            :min="1"
            :max="500"
            class="w-full"
          />
        </div>
      </div>

      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1" for="analyze-focus">Focus (optional)</label>
        <Textarea
          id="analyze-focus"
          v-model="form.focus"
          rows="2"
          class="w-full"
          placeholder="e.g., review PTO accrual flow for tech debt"
        />
      </div>

      <div class="flex items-center gap-2">
        <Checkbox v-model="form.dry_run" :binary="true" inputId="dry-run" />
        <label for="dry-run" class="text-sm text-slate-600">Dry run (no JIRA tickets)</label>
      </div>

      <div class="flex justify-end gap-2 pt-2">
        <Button label="Cancel" severity="secondary" text @click="visible = false" />
        <Button type="submit" label="Analyze" :disabled="!isValid || loading" />
      </div>
    </form>
  </Dialog>
</template>
