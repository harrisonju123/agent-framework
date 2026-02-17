<script setup lang="ts">
import { ref } from 'vue'
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
import type { CheckpointData } from '../types'

const { failedTasks, pendingCheckpoints, queues, handleRetryTask } = useAppState()

const selectedCheckpoint = ref<CheckpointData | null>(null)
const showCheckpointDialog = ref(false)

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
</script>

<template>
  <div class="space-y-4">
    <Tabs value="failed">
      <TabList>
        <Tab value="failed">Failed Tasks</Tab>
        <Tab value="checkpoints">Checkpoints</Tab>
        <Tab value="queues">Queues</Tab>
      </TabList>

      <TabPanels>
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
            <Column header="Actions" style="width: 80px">
              <template #body="{ data }">
                <Button label="Retry" severity="danger" size="small" @click="handleRetryTask(data.id)" />
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
  </div>
</template>
