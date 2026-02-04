<script setup lang="ts">
import { ref, computed } from 'vue'

const emit = defineEmits<{
  complete: []
}>()

type Step = 'welcome' | 'jira' | 'github' | 'repos' | 'review'

// Keyboard support
function handleKeydown(e: KeyboardEvent) {
  // Enter on forms submits/continues
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
function goToStep(step: Step) {
  currentStep.value = step
  error.value = ''
}

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
    <div class="h-1 bg-gray-800 mb-6">
      <div
        class="h-full bg-cyan-600 transition-all duration-300"
        :style="{ width: `${progressPercentage}%` }"
      ></div>
    </div>

    <!-- Error message -->
    <div v-if="error" class="mb-4 px-3 py-2 bg-red-500/20 border border-red-500 text-red-400 text-sm rounded">
      {{ error }}
    </div>

    <!-- Welcome Step -->
    <div v-if="currentStep === 'welcome'" class="space-y-6">
      <div>
        <h3 class="text-lg text-gray-200 mb-2">Agent Framework Setup</h3>
        <p class="text-sm text-gray-400">
          Configure JIRA and GitHub credentials, then register repositories.
        </p>
      </div>

      <div class="space-y-3 text-sm text-gray-400">
        <p>Required:</p>
        <ul class="list-disc list-inside space-y-1 ml-2">
          <li>JIRA API token</li>
          <li>GitHub personal access token</li>
          <li>Repository list</li>
        </ul>
      </div>

      <div class="text-xs text-gray-600">
        <p>Takes 5-10 minutes</p>
      </div>

      <div class="flex justify-end pt-4">
        <button
          @click="nextStep"
          class="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-sm rounded"
        >
          Continue
        </button>
      </div>
    </div>

    <!-- JIRA Configuration Step -->
    <div v-if="currentStep === 'jira'" class="space-y-4" @keydown="handleKeydown">
      <h3 class="text-lg text-gray-200">JIRA Configuration</h3>

      <div>
        <label class="block text-xs text-gray-500 mb-1">Server URL</label>
        <input
          v-model="jiraForm.server"
          type="url"
          placeholder="https://your-domain.atlassian.net"
          class="w-full bg-black border border-gray-700 px-2 py-1.5 text-sm focus:outline-none focus:border-cyan-500 rounded"
          @input="jiraValid = null"
          autofocus
        />
      </div>

      <div>
        <label class="block text-xs text-gray-500 mb-1">Email</label>
        <input
          v-model="jiraForm.email"
          type="email"
          placeholder="you@example.com"
          class="w-full bg-black border border-gray-700 px-2 py-1.5 text-sm focus:outline-none focus:border-cyan-500 rounded"
          @input="jiraValid = null"
        />
      </div>

      <div>
        <label class="block text-xs text-gray-500 mb-1">API Token</label>
        <input
          v-model="jiraForm.api_token"
          type="password"
          placeholder="Your JIRA API token"
          class="w-full bg-black border border-gray-700 px-2 py-1.5 text-sm focus:outline-none focus:border-cyan-500 rounded font-mono"
          @input="jiraValid = null"
        />
        <a
          href="https://id.atlassian.com/manage-profile/security/api-tokens"
          target="_blank"
          class="text-xs text-cyan-500 hover:text-cyan-400 mt-1 inline-block"
        >
          How to generate API token →
        </a>
      </div>

      <div>
        <label class="block text-xs text-gray-500 mb-1">Default Project Key (optional)</label>
        <input
          v-model="jiraForm.project"
          type="text"
          placeholder="PROJ"
          class="w-full bg-black border border-gray-700 px-2 py-1.5 text-sm focus:outline-none focus:border-cyan-500 rounded uppercase"
        />
      </div>

      <div>
        <button
          @click="validateJira"
          :disabled="jiraValidating || !jiraForm.server || !jiraForm.email || !jiraForm.api_token"
          class="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 text-sm rounded disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {{ jiraValidating ? 'Testing...' : 'Test Connection' }}
        </button>

        <span v-if="jiraValid === true" class="ml-3 text-green-400 text-sm">✓ Connected</span>
        <span v-else-if="jiraValid === false" class="ml-3 text-red-400 text-sm">✗ Failed</span>
      </div>

      <div class="flex justify-between pt-4">
        <button
          @click="previousStep"
          class="px-4 py-2 text-sm text-gray-500 hover:text-gray-300"
        >
          Back
        </button>
        <button
          @click="nextStep"
          :disabled="!canProceedFromJira"
          class="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-sm rounded disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Continue
        </button>
      </div>
    </div>

    <!-- GitHub Configuration Step -->
    <div v-if="currentStep === 'github'" class="space-y-4" @keydown="handleKeydown">
      <h3 class="text-lg text-gray-200">GitHub Configuration</h3>

      <div>
        <label class="block text-xs text-gray-500 mb-1">Personal Access Token</label>
        <input
          v-model="githubForm.token"
          type="password"
          placeholder="ghp_xxxxxxxxxxxxxxxxxxxx"
          class="w-full bg-black border border-gray-700 px-2 py-1.5 text-sm focus:outline-none focus:border-cyan-500 rounded font-mono"
          @input="githubValid = null"
        />
        <a
          href="https://github.com/settings/tokens"
          target="_blank"
          class="text-xs text-cyan-500 hover:text-cyan-400 mt-1 inline-block"
        >
          Generate new token (needs 'repo' scope) →
        </a>
      </div>

      <div>
        <button
          @click="validateGitHub"
          :disabled="githubValidating || !githubForm.token"
          class="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 text-sm rounded disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {{ githubValidating ? 'Testing...' : 'Test Connection' }}
        </button>

        <span v-if="githubValid === true" class="ml-3 text-green-400 text-sm">✓ Connected</span>
        <span v-else-if="githubValid === false" class="ml-3 text-red-400 text-sm">✗ Failed</span>
      </div>

      <div class="flex justify-between pt-4">
        <button
          @click="previousStep"
          class="px-4 py-2 text-sm text-gray-500 hover:text-gray-300"
        >
          Back
        </button>
        <button
          @click="nextStep"
          :disabled="!canProceedFromGitHub"
          class="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-sm rounded disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Continue
        </button>
      </div>
    </div>

    <!-- Repository Configuration Step -->
    <div v-if="currentStep === 'repos'" class="space-y-4">
      <div class="flex items-center justify-between">
        <h3 class="text-lg text-gray-200">Repository Configuration</h3>
        <button
          @click="addRepository"
          class="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-gray-200 text-sm rounded"
        >
          + Add Repository
        </button>
      </div>

      <div v-if="repositories.length === 0" class="text-center py-8 text-gray-600 text-sm">
        No repositories configured. Click "Add Repository" to get started.
      </div>

      <div v-for="(repo, index) in repositories" :key="index" class="p-3 bg-gray-800/50 border border-gray-700 rounded space-y-3">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs text-gray-500">Repository {{ index + 1 }}</span>
          <button
            @click="removeRepository(index)"
            class="text-xs text-red-400 hover:text-red-300"
          >
            Remove
          </button>
        </div>

        <div>
          <label class="block text-xs text-gray-500 mb-1">GitHub Repository (owner/repo)</label>
          <input
            v-model="repo.github_repo"
            type="text"
            placeholder="justworkshr/pto"
            class="w-full bg-black border px-2 py-1.5 text-sm focus:outline-none rounded"
            :class="repo.github_repo && !/^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_.-]+$/.test(repo.github_repo) ? 'border-red-500' : 'border-gray-700 focus:border-cyan-500'"
          />
          <span v-if="repo.github_repo && !/^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_.-]+$/.test(repo.github_repo)" class="text-xs text-red-400">
            Must be owner/repo format
          </span>
        </div>

        <div>
          <label class="block text-xs text-gray-500 mb-1">JIRA Project Key</label>
          <input
            v-model="repo.jira_project"
            type="text"
            placeholder="PROJ"
            class="w-full bg-black border border-gray-700 px-2 py-1.5 text-sm focus:outline-none focus:border-cyan-500 rounded uppercase"
          />
        </div>

        <div>
          <label class="block text-xs text-gray-500 mb-1">Name</label>
          <input
            v-model="repo.name"
            type="text"
            placeholder="pto"
            class="w-full bg-black border border-gray-700 px-2 py-1.5 text-sm focus:outline-none focus:border-cyan-500 rounded"
          />
        </div>
      </div>

      <div class="flex justify-between pt-4">
        <button
          @click="previousStep"
          class="px-4 py-2 text-sm text-gray-500 hover:text-gray-300"
        >
          Back
        </button>
        <button
          @click="nextStep"
          :disabled="!canProceedFromRepos"
          class="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-sm rounded disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Review
        </button>
      </div>
    </div>

    <!-- Review Step -->
    <div v-if="currentStep === 'review'" class="space-y-4">
      <h3 class="text-lg text-gray-200">Review Configuration</h3>

      <div class="space-y-3 text-sm">
        <div class="p-3 bg-gray-800/50 border border-gray-700 rounded">
          <div class="text-xs text-gray-500 mb-2">JIRA</div>
          <div class="space-y-1 text-gray-300">
            <div>Server: {{ jiraForm.server }}</div>
            <div>Email: {{ jiraForm.email }}</div>
            <div v-if="jiraForm.project">Project: {{ jiraForm.project }}</div>
          </div>
        </div>

        <div class="p-3 bg-gray-800/50 border border-gray-700 rounded">
          <div class="text-xs text-gray-500 mb-2">GitHub</div>
          <div class="text-gray-300">
            Token: {{ githubForm.token.substring(0, 8) }}...
          </div>
        </div>

        <div class="p-3 bg-gray-800/50 border border-gray-700 rounded">
          <div class="text-xs text-gray-500 mb-2">Repositories ({{ repositories.length }})</div>
          <div class="space-y-2">
            <div v-for="(repo, index) in repositories" :key="index" class="text-gray-300">
              {{ repo.github_repo }} → {{ repo.jira_project }}
            </div>
          </div>
        </div>
      </div>

      <div class="flex justify-between pt-4">
        <button
          @click="previousStep"
          class="px-4 py-2 text-sm text-gray-500 hover:text-gray-300"
        >
          Back to Edit
        </button>
        <button
          @click="submitConfiguration"
          :disabled="loading"
          class="px-4 py-2 bg-green-600 hover:bg-green-500 text-white text-sm rounded disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {{ loading ? 'Saving...' : 'Save Configuration' }}
        </button>
      </div>
    </div>
  </div>
</template>
