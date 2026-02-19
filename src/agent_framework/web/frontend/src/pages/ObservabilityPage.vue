<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useApi } from '../composables/useApi'
import type { AgenticMetrics } from '../types'

const { getAgenticMetrics } = useApi()

const metrics = ref<AgenticMetrics | null>(null)
const hours = ref(24)
let pollTimer: ReturnType<typeof setInterval> | null = null

async function refresh() {
  const result = await getAgenticMetrics(hours.value)
  if (result) metrics.value = result
}

function fmt(n: number, decimals = 1): string {
  return n.toFixed(decimals)
}

// Specialization profile entries sorted by usage descending
const sortedProfiles = computed(() => {
  if (!metrics.value) return []
  return Object.entries(metrics.value.specialization.profiles).sort(
    ([, a], [, b]) => b - a
  )
})

// Memory category entries sorted by count descending (top 5)
const topCategories = computed(() => {
  if (!metrics.value) return []
  return Object.entries(metrics.value.memory.by_category)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5)
})

onMounted(() => {
  refresh()
  pollTimer = setInterval(refresh, 30_000)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})
</script>

<template>
  <div class="space-y-6">
    <!-- Header + controls -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-xl font-semibold text-slate-800">Observability</h1>
        <p class="text-sm text-slate-500 mt-0.5">Agentic feature metrics — refreshes every 30 s</p>
      </div>
      <div class="flex items-center gap-3">
        <label class="text-sm text-slate-600">Window:</label>
        <select
          v-model="hours"
          @change="refresh"
          class="text-sm border border-slate-300 rounded-lg px-3 py-1.5 bg-white text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option :value="1">1 hour</option>
          <option :value="6">6 hours</option>
          <option :value="24">24 hours</option>
          <option :value="72">3 days</option>
          <option :value="168">7 days</option>
        </select>
        <button
          @click="refresh"
          class="text-sm px-3 py-1.5 rounded-lg border border-slate-300 hover:bg-slate-100 text-slate-600 transition-colors"
        >
          Refresh
        </button>
      </div>
    </div>

    <!-- Loading state -->
    <div v-if="!metrics" class="text-center py-16 text-slate-400">
      Loading metrics…
    </div>

    <template v-else>
      <!-- Row 1: Memory + Self-eval -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">

        <!-- Memory hit rate -->
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
          <h2 class="text-sm font-medium text-slate-500 mb-3">Memory Hit Rate</h2>
          <div class="flex items-end gap-3 mb-4">
            <span class="text-4xl font-bold text-blue-600">{{ fmt(metrics.memory.hit_rate) }}%</span>
            <span class="text-sm text-slate-400 mb-1">of stored memories recalled at least once</span>
          </div>
          <div class="text-xs text-slate-400 mb-3">
            {{ metrics.memory.accessed_memories }} / {{ metrics.memory.total_memories }} entries accessed
          </div>
          <!-- Category breakdown -->
          <div v-if="topCategories.length" class="space-y-1.5">
            <p class="text-xs font-medium text-slate-400 uppercase tracking-wide mb-1">Top categories</p>
            <div
              v-for="[cat, count] in topCategories"
              :key="cat"
              class="flex items-center gap-2"
            >
              <div class="flex-1 bg-slate-100 rounded-full h-1.5 overflow-hidden">
                <div
                  class="h-full bg-blue-400 rounded-full"
                  :style="{ width: `${(count / metrics.memory.total_memories) * 100}%` }"
                />
              </div>
              <span class="text-xs text-slate-500 w-24 truncate">{{ cat }}</span>
              <span class="text-xs font-medium text-slate-700 w-6 text-right">{{ count }}</span>
            </div>
          </div>
          <p v-else class="text-xs text-slate-400">No memory entries in this window.</p>
        </div>

        <!-- Self-evaluation retry rate -->
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
          <h2 class="text-sm font-medium text-slate-500 mb-3">Self-Eval Retry Rate</h2>
          <div class="flex items-end gap-3 mb-4">
            <span
              class="text-4xl font-bold"
              :class="metrics.self_eval.retry_rate > 20 ? 'text-amber-500' : 'text-emerald-600'"
            >
              {{ fmt(metrics.self_eval.retry_rate) }}%
            </span>
            <span class="text-sm text-slate-400 mb-1">of evaluated tasks triggered a retry</span>
          </div>
          <div class="grid grid-cols-3 gap-3 text-center">
            <div class="bg-emerald-50 rounded-lg p-3">
              <div class="text-xl font-bold text-emerald-600">{{ metrics.self_eval.pass_count }}</div>
              <div class="text-xs text-emerald-700 mt-0.5">Pass</div>
            </div>
            <div class="bg-red-50 rounded-lg p-3">
              <div class="text-xl font-bold text-red-600">{{ metrics.self_eval.fail_count }}</div>
              <div class="text-xs text-red-700 mt-0.5">Fail</div>
            </div>
            <div class="bg-slate-50 rounded-lg p-3">
              <div class="text-xl font-bold text-slate-600">{{ metrics.self_eval.auto_pass_count }}</div>
              <div class="text-xs text-slate-500 mt-0.5">Auto-pass</div>
            </div>
          </div>
          <p class="text-xs text-slate-400 mt-3">
            {{ metrics.self_eval.total_tasks_evaluated }} tasks evaluated total
          </p>
        </div>
      </div>

      <!-- Row 2: Replan + Context budget -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">

        <!-- Replan trigger & success -->
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
          <h2 class="text-sm font-medium text-slate-500 mb-3">Replan Trigger Rate</h2>
          <div class="flex items-end gap-3 mb-4">
            <span class="text-4xl font-bold text-purple-600">
              {{ fmt(metrics.replan.trigger_rate_pct) }}%
            </span>
            <span class="text-sm text-slate-400 mb-1">of terminal tasks triggered a replan</span>
          </div>
          <div class="grid grid-cols-3 gap-3 text-center">
            <div class="bg-purple-50 rounded-lg p-3">
              <div class="text-xl font-bold text-purple-600">{{ metrics.replan.total_tasks_with_replan }}</div>
              <div class="text-xs text-purple-700 mt-0.5">Replanned</div>
            </div>
            <div class="bg-emerald-50 rounded-lg p-3">
              <div class="text-xl font-bold text-emerald-600">{{ metrics.replan.success_after_replan }}</div>
              <div class="text-xs text-emerald-700 mt-0.5">Succeeded</div>
            </div>
            <div class="bg-slate-50 rounded-lg p-3">
              <div class="text-xl font-bold text-slate-600">{{ metrics.replan.total_replan_events }}</div>
              <div class="text-xs text-slate-500 mt-0.5">Total events</div>
            </div>
          </div>
        </div>

        <!-- Context budget utilization -->
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
          <h2 class="text-sm font-medium text-slate-500 mb-3">Context Budget Utilization</h2>
          <div class="flex items-end gap-3 mb-4">
            <span
              class="text-4xl font-bold"
              :class="metrics.context_budget.critical_budget_events > 0 ? 'text-red-600' : 'text-slate-700'"
            >
              {{ metrics.context_budget.critical_budget_events }}
            </span>
            <span class="text-sm text-slate-400 mb-1">critical budget warnings</span>
          </div>
          <div class="grid grid-cols-2 gap-3 text-center">
            <div class="bg-slate-50 rounded-lg p-3">
              <div class="text-xl font-bold text-slate-700">
                {{ metrics.context_budget.total_token_budget_warnings }}
              </div>
              <div class="text-xs text-slate-500 mt-0.5">Total warnings</div>
            </div>
            <div class="bg-blue-50 rounded-lg p-3">
              <div class="text-xl font-bold text-blue-600">
                {{ fmt(metrics.context_budget.avg_output_token_ratio_pct) }}%
              </div>
              <div class="text-xs text-blue-700 mt-0.5">Avg output ratio</div>
            </div>
          </div>
          <p class="text-xs text-slate-400 mt-3">
            Output token ratio = tokens_out / (tokens_in + tokens_out) per LLM call.
          </p>
        </div>
      </div>

      <!-- Row 3: Debate usage + Specialization profile distribution -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">

        <!-- Debate usage and confidence -->
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
          <h2 class="text-sm font-medium text-slate-500 mb-3">Debate Usage &amp; Confidence</h2>
          <div v-if="!metrics.debate.data_available" class="py-4">
            <p class="text-sm text-slate-400">
              No debate data recorded yet. Debate outcomes will appear here once the
              MCP debate system writes results to session logs.
            </p>
          </div>
          <template v-else>
            <div class="flex items-end gap-3 mb-4">
              <span class="text-4xl font-bold text-violet-600">
                {{ fmt(metrics.debate.debate_usage_rate) }}%
              </span>
              <span class="text-sm text-slate-400 mb-1">of tasks used debate</span>
            </div>
            <div class="grid grid-cols-2 gap-3 text-center">
              <div class="bg-violet-50 rounded-lg p-3">
                <div class="text-xl font-bold text-violet-600">{{ metrics.debate.total_debates }}</div>
                <div class="text-xs text-violet-700 mt-0.5">Total debates</div>
              </div>
              <div class="bg-slate-50 rounded-lg p-3">
                <div class="text-xl font-bold text-slate-700">{{ fmt(metrics.debate.avg_confidence, 2) }}</div>
                <div class="text-xs text-slate-500 mt-0.5">Avg confidence</div>
              </div>
            </div>
          </template>
        </div>

        <!-- Specialization profile distribution -->
        <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
          <h2 class="text-sm font-medium text-slate-500 mb-3">Specialization Profile Distribution</h2>
          <div v-if="sortedProfiles.length" class="space-y-3">
            <div
              v-for="[profile, count] in sortedProfiles"
              :key="profile"
              class="flex items-center gap-3"
            >
              <span class="text-sm font-medium text-slate-700 w-32 truncate">{{ profile }}</span>
              <div class="flex-1 bg-slate-100 rounded-full h-2 overflow-hidden">
                <div
                  class="h-full bg-indigo-500 rounded-full transition-all"
                  :style="{
                    width: metrics.specialization.total_specializations > 0
                      ? `${(count / metrics.specialization.total_specializations) * 100}%`
                      : '0%'
                  }"
                />
              </div>
              <span class="text-sm text-slate-600 w-20 text-right">
                {{ count }} match{{ count !== 1 ? 'es' : '' }}
              </span>
            </div>
            <p class="text-xs text-slate-400 mt-1">
              {{ metrics.specialization.total_specializations }} total specialization matches recorded
            </p>
          </div>
          <p v-else class="text-sm text-slate-400 py-4">
            No specialization profiles recorded in this window. Profiles are applied when the engineer
            agent matches repository file patterns to a cached specialization profile.
          </p>
        </div>
      </div>
    </template>
  </div>
</template>
