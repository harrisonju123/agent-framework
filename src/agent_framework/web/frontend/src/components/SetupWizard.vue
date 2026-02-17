<script setup lang="ts">
import { ref, computed } from 'vue'
import InputText from 'primevue/inputtext'
import Button from 'primevue/button'

const emit = defineEmits<{
  complete: []
}>()

type Step = 'welcome' | 'jira' | 'github' | 'repos' | 'review'

function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    if (currentStep.value === 'jira' && canProceedFromJira.value) {
      nextStep()
    } else if (currentStep.value === 'github' && canProceedFromGitHub.value) {
      nextStep()
    }
  }
}

const currentStep = ref<Step>('welcome')
const loading = ref(false)
const error = ref<string>('')

// Form data
const jiraForm = ref({
  server: '',
  email: '',
  api_token: '',
  project: ''
})

const githubForm = ref({
  token: ''
})

const repositories = ref<Array<{ github_repo: string; jira_project: string; name: string }>>([])

// Validation states
const jiraValid = ref<boolean | null>(null)
const githubValid = ref<boolean | null>(null)

const jiraValidating = ref(false)
const githubValidating = ref(false)

// Computed
const canProceedFromJira = computed(() => {
  return jiraValid.value === true &&
    jiraForm.value.server &&
    jiraForm.value.email &&
    jiraForm.value.api_token
})

const canProceedFromGitHub = computed(() => {
  return githubValid.value === true && githubForm.value.token
})

const canProceedFromRepos = computed(() => {
  return repositories.value.length > 0
})

const progressPercentage = computed(() => {
  const steps = ['welcome', 'jira', 'github', 'repos', 'review']
  const currentIndex = steps.indexOf(currentStep.value)
  return ((currentIndex + 1) / steps.length) * 100
})

// Navigation
function nextStep() {
  const steps: Step[] = ['welcome', 'jira', 'github', 'repos', 'review']
  const currentIndex = steps.indexOf(currentStep.value)
  if (currentIndex < steps.length - 1) {
    currentStep.value = steps[currentIndex + 1]
  }
}

function previousStep() {
  const steps: Step[] = ['welcome', 'jira', 'github', 'repos', 'review']
  const currentIndex = steps.indexOf(currentStep.value)
  if (currentIndex > 0) {
    currentStep.value = steps[currentIndex - 1]
  }
}

// JIRA validation
async function validateJira() {
  if (!jiraForm.value.server || !jiraForm.value.email || !jiraForm.value.api_token) {
    return
  }

  jiraValidating.value = true
  jiraValid.value = null
  error.value = ''

  try {
    const response = await fetch('/api/setup/validate-jira', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        server: jiraForm.value.server,
        email: jiraForm.value.email,
        api_token: jiraForm.value.api_token,
        project: jiraForm.value.project || null
      })
    })

    const data = await response.json()

    if (data.valid) {
      jiraValid.value = true
    } else {
      jiraValid.value = false
      error.value = data.error || 'JIRA validation failed'
    }
  } catch (e) {
    jiraValid.value = false
    error.value = 'Failed to validate JIRA credentials'
  } finally {
    jiraValidating.value = false
  }
}

// GitHub validation
async function validateGitHub() {
  if (!githubForm.value.token) {
    return
  }

  githubValidating.value = true
  githubValid.value = null
  error.value = ''

  try {
    const response = await fetch('/api/setup/validate-github', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        token: githubForm.value.token
      })
    })

    const data = await response.json()

    if (data.valid) {
      githubValid.value = true
    } else {
      githubValid.value = false
      error.value = data.error || 'GitHub validation failed'
    }
  } catch (e) {
    githubValid.value = false
    error.value = 'Failed to validate GitHub token'
  } finally {
    githubValidating.value = false
  }
}

// Repository management
function addRepository() {
  repositories.value.push({
    github_repo: '',
    jira_project: '',
    name: ''
  })
}

function removeRepository(index: number) {
  repositories.value.splice(index, 1)
}

// Submit configuration
async function submitConfiguration() {
  loading.value = true
  error.value = ''

  try {
    const response = await fetch('/api/setup/save-config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jira: {
          server: jiraForm.value.server,
          email: jiraForm.value.email,
          api_token: jiraForm.value.api_token,
          project: jiraForm.value.project || null
        },
        github: {
          token: githubForm.value.token
        },
        repositories: repositories.value,
        enable_mcp: false
      })
    })

    if (response.ok) {
      emit('complete')
    } else {
      const data = await response.json()
      error.value = data.detail || 'Failed to save configuration'
    }
  } catch (e) {
    error.value = 'Failed to save configuration'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="min-h-[500px] flex flex-col">
    <!-- Progress bar -->
    <div class="h-1.5 bg-slate-200 rounded-full mb-6">
      <div
        class="h-full bg-blue-600 rounded-full transition-all duration-300"
        :style="{ width: `${progressPercentage}%` }"
      ></div>
    </div>

    <!-- Error message -->
    <div v-if="error" class="mb-4 px-4 py-3 bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg">
      {{ error }}
    </div>

    <!-- Welcome Step -->
    <div v-if="currentStep === 'welcome'" class="space-y-6">
      <div>
        <h3 class="text-lg font-semibold text-slate-800 mb-2">Agent Framework Setup</h3>
        <p class="text-sm text-slate-500">
          Configure JIRA and GitHub credentials, then register repositories.
        </p>
      </div>

      <div class="space-y-3 text-sm text-slate-500">
        <p>Required:</p>
        <ul class="list-disc list-inside space-y-1 ml-2">
          <li>JIRA API token</li>
          <li>GitHub personal access token</li>
          <li>Repository list</li>
        </ul>
      </div>

      <div class="text-xs text-slate-400">
        <p>Takes 5-10 minutes</p>
      </div>

      <div class="flex justify-end pt-4">
        <Button label="Continue" @click="nextStep" />
      </div>
    </div>

    <!-- JIRA Configuration Step -->
    <div v-if="currentStep === 'jira'" class="space-y-4" @keydown="handleKeydown">
      <h3 class="text-lg font-semibold text-slate-800">JIRA Configuration</h3>

      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1">Server URL</label>
        <InputText
          v-model="jiraForm.server"
          type="url"
          placeholder="https://your-domain.atlassian.net"
          class="w-full"
          @input="jiraValid = null"
          autofocus
        />
      </div>

      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1">Email</label>
        <InputText
          v-model="jiraForm.email"
          type="email"
          placeholder="you@example.com"
          class="w-full"
          @input="jiraValid = null"
        />
      </div>

      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1">API Token</label>
        <InputText
          v-model="jiraForm.api_token"
          type="password"
          placeholder="Your JIRA API token"
          class="w-full font-mono"
          @input="jiraValid = null"
        />
        <a
          href="https://id.atlassian.com/manage-profile/security/api-tokens"
          target="_blank"
          class="text-xs text-blue-600 hover:text-blue-700 mt-1 inline-block"
        >
          How to generate API token
        </a>
      </div>

      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1">Default Project Key (optional)</label>
        <InputText
          v-model="jiraForm.project"
          type="text"
          placeholder="PROJ"
          class="w-full uppercase"
        />
      </div>

      <div class="flex items-center gap-3">
        <Button
          :label="jiraValidating ? 'Testing...' : 'Test Connection'"
          severity="secondary"
          outlined
          :disabled="jiraValidating || !jiraForm.server || !jiraForm.email || !jiraForm.api_token"
          @click="validateJira"
        />
        <span v-if="jiraValid === true" class="text-emerald-600 text-sm">Connected</span>
        <span v-else-if="jiraValid === false" class="text-red-600 text-sm">Failed</span>
      </div>

      <div class="flex justify-between pt-4">
        <Button label="Back" severity="secondary" text @click="previousStep" />
        <Button label="Continue" :disabled="!canProceedFromJira" @click="nextStep" />
      </div>
    </div>

    <!-- GitHub Configuration Step -->
    <div v-if="currentStep === 'github'" class="space-y-4" @keydown="handleKeydown">
      <h3 class="text-lg font-semibold text-slate-800">GitHub Configuration</h3>

      <div>
        <label class="block text-sm font-medium text-slate-700 mb-1">Personal Access Token</label>
        <InputText
          v-model="githubForm.token"
          type="password"
          placeholder="ghp_xxxxxxxxxxxxxxxxxxxx"
          class="w-full font-mono"
          @input="githubValid = null"
        />
        <a
          href="https://github.com/settings/tokens"
          target="_blank"
          class="text-xs text-blue-600 hover:text-blue-700 mt-1 inline-block"
        >
          Generate new token (needs 'repo' scope)
        </a>
      </div>

      <div class="flex items-center gap-3">
        <Button
          :label="githubValidating ? 'Testing...' : 'Test Connection'"
          severity="secondary"
          outlined
          :disabled="githubValidating || !githubForm.token"
          @click="validateGitHub"
        />
        <span v-if="githubValid === true" class="text-emerald-600 text-sm">Connected</span>
        <span v-else-if="githubValid === false" class="text-red-600 text-sm">Failed</span>
      </div>

      <div class="flex justify-between pt-4">
        <Button label="Back" severity="secondary" text @click="previousStep" />
        <Button label="Continue" :disabled="!canProceedFromGitHub" @click="nextStep" />
      </div>
    </div>

    <!-- Repository Configuration Step -->
    <div v-if="currentStep === 'repos'" class="space-y-4">
      <div class="flex items-center justify-between">
        <h3 class="text-lg font-semibold text-slate-800">Repository Configuration</h3>
        <Button label="+ Add Repository" severity="secondary" size="small" @click="addRepository" />
      </div>

      <div v-if="repositories.length === 0" class="text-center py-8 text-slate-400 text-sm">
        No repositories configured. Click "Add Repository" to get started.
      </div>

      <div v-for="(repo, index) in repositories" :key="index" class="p-4 bg-slate-50 border border-slate-200 rounded-lg space-y-3">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs text-slate-500 font-medium">Repository {{ index + 1 }}</span>
          <button @click="removeRepository(index)" class="text-xs text-red-500 hover:text-red-600">Remove</button>
        </div>

        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">GitHub Repository (owner/repo)</label>
          <InputText
            v-model="repo.github_repo"
            type="text"
            placeholder="justworkshr/pto"
            class="w-full"
            :class="{ 'p-invalid': repo.github_repo && !/^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_.-]+$/.test(repo.github_repo) }"
          />
          <small v-if="repo.github_repo && !/^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_.-]+$/.test(repo.github_repo)" class="text-red-600">
            Must be owner/repo format
          </small>
        </div>

        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">JIRA Project Key</label>
          <InputText v-model="repo.jira_project" type="text" placeholder="PROJ" class="w-full uppercase" />
        </div>

        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">Name</label>
          <InputText v-model="repo.name" type="text" placeholder="pto" class="w-full" />
        </div>
      </div>

      <div class="flex justify-between pt-4">
        <Button label="Back" severity="secondary" text @click="previousStep" />
        <Button label="Review" :disabled="!canProceedFromRepos" @click="nextStep" />
      </div>
    </div>

    <!-- Review Step -->
    <div v-if="currentStep === 'review'" class="space-y-4">
      <h3 class="text-lg font-semibold text-slate-800">Review Configuration</h3>

      <div class="space-y-3 text-sm">
        <div class="p-4 bg-slate-50 border border-slate-200 rounded-lg">
          <div class="text-xs text-slate-500 font-medium mb-2">JIRA</div>
          <div class="space-y-1 text-slate-700">
            <div>Server: {{ jiraForm.server }}</div>
            <div>Email: {{ jiraForm.email }}</div>
            <div v-if="jiraForm.project">Project: {{ jiraForm.project }}</div>
          </div>
        </div>

        <div class="p-4 bg-slate-50 border border-slate-200 rounded-lg">
          <div class="text-xs text-slate-500 font-medium mb-2">GitHub</div>
          <div class="text-slate-700">
            Token: {{ githubForm.token.substring(0, 8) }}...
          </div>
        </div>

        <div class="p-4 bg-slate-50 border border-slate-200 rounded-lg">
          <div class="text-xs text-slate-500 font-medium mb-2">Repositories ({{ repositories.length }})</div>
          <div class="space-y-2">
            <div v-for="(repo, index) in repositories" :key="index" class="text-slate-700">
              {{ repo.github_repo }} → {{ repo.jira_project }}
            </div>
          </div>
        </div>
      </div>

      <div class="flex justify-between pt-4">
        <Button label="Back to Edit" severity="secondary" text @click="previousStep" />
        <Button
          :label="loading ? 'Saving...' : 'Save Configuration'"
          severity="success"
          :disabled="loading"
          @click="submitConfiguration"
        />
      </div>
    </div>
  </div>
</template>
