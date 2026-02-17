<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useAppState } from '../composables/useAppState'
import DataTable from 'primevue/datatable'
import Column from 'primevue/column'
import Tag from 'primevue/tag'
import Tabs from 'primevue/tabs'
import TabList from 'primevue/tablist'
import Tab from 'primevue/tab'
import TabPanels from 'primevue/tabpanels'
import TabPanel from 'primevue/tabpanel'
import Button from 'primevue/button'
import CheckpointDetailDialog from '../components/dialogs/CheckpointDetailDialog.vue'
import CreateTaskDialog from '../components/dialogs/CreateTaskDialog.vue'
import type { CheckpointData, ActiveTask } from '../types'

const {
  failedTasks, pendingCheckpoints, queues,
  handleRetryTask, handleCancelTask, handleDeleteTask,
  getActiveTasks, showConfirm, showCreateTaskDialog,
} = useAppState()

const selectedCheckpoint = ref<CheckpointData | null>(null)
const showCheckpointDialog = ref(false)
const activeTasks = ref<ActiveTask[]>([])
let pollTimer: ReturnType<typeof setTimeout> | null = null
let polling = true

async function fetchActiveTasks() {
  activeTasks.value = await getActiveTasks()
}

async function pollLoop() {
  await fetchActiveTasks()
  if (polling) {
    pollTimer = setTimeout(pollLoop, 5000)
  }
}

async function onCancelTask(taskId: string) {
  await handleCancelTask(taskId)
  await fetchActiveTasks()
}

function onDeleteActiveTask(taskId: string) {
  showConfirm('Delete Task', `Permanently delete task ${taskId}?`, async () => {
    await handleDeleteTask(taskId)
    await fetchActiveTasks()
  }, true)
}

function onDeleteFailedTask(taskId: string) {
  showConfirm('Delete Task', `Permanently delete failed task ${taskId}?`, async () => {
    await handleDeleteTask(taskId)
  }, true)
}

onMounted(() => {
  pollLoop()
})

onUnmounted(() => {
  polling = false
  if (pollTimer) clearTimeout(pollTimer)
})

function truncateError(error: string | null, maxLen = 60): string {
  if (!error) return ''
  return error.length > maxLen ? error.slice(0, maxLen) + '...' : error
}

function getTaskKey(task: any): string {
  return task.jira_key || task.id.slice(0, 12)
}

function reviewCheckpoint(cp: CheckpointData) {
  selectedCheckpoint.value = cp
  showCheckpointDialog.value = true
}

function statusSeverity(status: string): string {
  return status === 'in_progress' ? 'info' : 'secondary'
}

function statusLabel(status: string): string {
  return status === 'in_progress' ? 'Running' : 'Pending'
}
</script>

<template>
  <div class="space-y-4">
    <Tabs value="active">
      <TabList>
        <Tab value="active">Active Tasks</Tab>
        <Tab value="failed">Failed Tasks</Tab>
        <Tab value="checkpoints">Checkpoints</Tab>
        <Tab value="queues">Queues</Tab>
      </TabList>

      <TabPanels>
        <!-- Active Tasks Tab -->
        <TabPanel value="active">
          <div class="flex justify-end mb-2">
            <Button label="Add Task" icon="pi pi-plus" size="small" @click="showCreateTaskDialog = true" />
          </div>
          <DataTable :value="activeTasks" stripedRows :paginator="activeTasks.length > 10" :rows="10" class="text-sm">
            <template #empty>
              <div class="text-center py-6 text-slate-400">No active tasks</div>
            </template>
            <Column header="Key/ID" style="width: 120px">
              <template #body="{ data }">
                <span class="font-mono text-blue-600 font-medium text-xs">{{ getTaskKey(data) }}</span>
              </template>
            </Column>
            <Column field="title" header="Title">
              <template #body="{ data }">
                <span class="text-slate-700 text-sm" :title="data.title">{{ data.title }}</span>
              </template>
            </Column>
            <Column header="Status" style="width: 100px">
              <template #body="{ data }">
                <Tag :value="statusLabel(data.status)" :severity="statusSeverity(data.status)" />
              </template>
            </Column>
            <Column field="assigned_to" header="Agent" style="width: 100px" />
            <Column field="task_type" header="Type" style="width: 120px" />
            <Column header="Actions" style="width: 150px">
              <template #body="{ data }">
                <div class="flex gap-1">
                  <Button label="Cancel" severity="danger" size="small" @click="onCancelTask(data.id)" />
                  <Button
                    v-if="data.status === 'pending'"
                    label="Delete"
                    severity="secondary"
                    size="small"
                    @click="onDeleteActiveTask(data.id)"
                  />
                </div>
              </template>
            </Column>
          </DataTable>
        </TabPanel>

        <!-- Failed Tasks Tab -->
        <TabPanel value="failed">
          <DataTable :value="failedTasks" stripedRows :paginator="failedTasks.length > 10" :rows="10" class="text-sm">
            <template #empty>
              <div class="text-center py-6 text-slate-400">No failed tasks</div>
            </template>
            <Column header="Key/ID" style="width: 120px">
              <template #body="{ data }">
                <span class="font-mono text-red-600 font-medium text-xs">{{ getTaskKey(data) }}</span>
              </template>
            </Column>
            <Column field="title" header="Title">
              <template #body="{ data }">
                <span class="text-slate-700 text-sm" :title="data.title">{{ data.title }}</span>
              </template>
            </Column>
            <Column field="assigned_to" header="Agent" style="width: 100px" />
            <Column header="Retries" style="width: 80px">
              <template #body="{ data }">
                <Tag :value="`x${data.retry_count}`" severity="danger" />
              </template>
            </Column>
            <Column header="Error">
              <template #body="{ data }">
                <span class="text-red-500 text-xs" :title="data.last_error">{{ truncateError(data.last_error) }}</span>
              </template>
            </Column>
            <Column header="Actions" style="width: 150px">
              <template #body="{ data }">
                <div class="flex gap-1">
                  <Button label="Retry" severity="danger" size="small" @click="handleRetryTask(data.id)" />
                  <Button label="Delete" severity="secondary" size="small" @click="onDeleteFailedTask(data.id)" />
                </div>
              </template>
            </Column>
          </DataTable>
        </TabPanel>

        <!-- Checkpoints Tab -->
        <TabPanel value="checkpoints">
          <DataTable :value="pendingCheckpoints" stripedRows class="text-sm">
            <template #empty>
              <div class="text-center py-6 text-slate-400">No pending checkpoints</div>
            </template>
            <Column header="Checkpoint ID" style="width: 160px">
              <template #body="{ data }">
                <span class="font-mono text-amber-700 text-xs">{{ data.checkpoint_id }}</span>
              </template>
            </Column>
            <Column field="title" header="Task Title">
              <template #body="{ data }">
                <span class="text-slate-700 text-sm">{{ data.title }}</span>
              </template>
            </Column>
            <Column header="Message">
              <template #body="{ data }">
                <span class="text-slate-500 text-xs" :title="data.checkpoint_message">
                  {{ data.checkpoint_message?.slice(0, 60) }}{{ (data.checkpoint_message?.length || 0) > 60 ? '...' : '' }}
                </span>
              </template>
            </Column>
            <Column field="assigned_to" header="Agent" style="width: 100px" />
            <Column header="Actions" style="width: 100px">
              <template #body="{ data }">
                <Button label="Review" severity="warn" size="small" @click="reviewCheckpoint(data)" />
              </template>
            </Column>
          </DataTable>
        </TabPanel>

        <!-- Queues Tab -->
        <TabPanel value="queues">
          <DataTable :value="queues" stripedRows class="text-sm">
            <template #empty>
              <div class="text-center py-6 text-slate-400">No queues configured</div>
            </template>
            <Column field="queue_id" header="Queue" />
            <Column field="agent_name" header="Agent" />
            <Column header="Pending">
              <template #body="{ data }">
                <Tag
                  :value="String(data.pending_count)"
                  :severity="data.pending_count > 0 ? 'warn' : 'secondary'"
                />
              </template>
            </Column>
            <Column header="Oldest Task Age">
              <template #body="{ data }">
                <span class="text-slate-500 text-sm">
                  {{ data.oldest_task_age ? `${Math.floor(data.oldest_task_age / 60)}m` : '-' }}
                </span>
              </template>
            </Column>
          </DataTable>
        </TabPanel>
      </TabPanels>
    </Tabs>

    <CheckpointDetailDialog
      v-model:visible="showCheckpointDialog"
      :checkpoint="selectedCheckpoint"
      @close="selectedCheckpoint = null"
    />

    <CreateTaskDialog v-model:visible="showCreateTaskDialog" />
  </div>
</template>
