<script setup lang="ts">
import { ref } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const collapsed = ref(false)

const navItems = [
  { to: '/', name: 'dashboard', label: 'Dashboard', icon: 'pi-th-large' },
  { to: '/agents', name: 'agents', label: 'Agents', icon: 'pi-server' },
  { to: '/tasks', name: 'tasks', label: 'Tasks', icon: 'pi-list-check' },
  { to: '/logs', name: 'logs', label: 'Logs', icon: 'pi-file' },
  { to: '/insights', name: 'insights', label: 'Insights', icon: 'pi-chart-line' },
  { to: '/settings', name: 'settings', label: 'Settings', icon: 'pi-cog' },
]

const shortcuts = [
  { key: 's', label: 'start' },
  { key: 'x', label: 'stop' },
  { key: 'p', label: 'pause' },
  { key: 'w', label: 'work' },
  { key: 'a', label: 'analyze' },
  { key: 't', label: 'ticket' },
  { key: 'r', label: 'retry' },
  { key: 'c', label: 'approve' },
]

function isActive(name: string): boolean {
  return route.name === name
}
</script>

<template>
  <aside
    class="bg-white border-r border-slate-200 flex flex-col shrink-0 transition-all duration-200 h-full"
    :class="collapsed ? 'w-16' : 'w-64'"
  >
    <!-- App title -->
    <div class="px-5 py-4 border-b border-slate-200 flex items-center gap-3">
      <span class="pi pi-box text-blue-600 text-lg"></span>
      <span v-if="!collapsed" class="text-base font-semibold text-slate-800">Agent Dashboard</span>
    </div>

    <!-- Navigation -->
    <nav class="flex-1 py-2 px-3 space-y-1">
      <router-link
        v-for="item in navItems"
        :key="item.name"
        :to="item.to"
        class="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors"
        :class="isActive(item.name)
          ? 'bg-blue-50 text-blue-700 border-r-2 border-blue-600'
          : 'text-slate-600 hover:bg-slate-100'"
      >
        <span class="pi text-base" :class="item.icon"></span>
        <span v-if="!collapsed">{{ item.label }}</span>
      </router-link>
    </nav>

    <!-- Keyboard shortcuts -->
    <div v-if="!collapsed" class="px-4 py-3 border-t border-slate-200">
      <p class="text-xs font-medium text-slate-400 mb-2">Keyboard Shortcuts</p>
      <div class="flex flex-wrap gap-x-3 gap-y-1">
        <span v-for="s in shortcuts" :key="s.key" class="text-xs text-slate-400">
          <kbd class="px-1 py-0.5 bg-slate-100 rounded text-slate-500 font-mono text-[10px]">{{ s.key }}</kbd>
          {{ s.label }}
        </span>
      </div>
    </div>

    <!-- Collapse toggle -->
    <button
      @click="collapsed = !collapsed"
      class="px-4 py-3 border-t border-slate-200 text-slate-400 hover:text-slate-600 hover:bg-slate-50 transition-colors text-sm flex items-center gap-2"
    >
      <span class="pi" :class="collapsed ? 'pi-angle-right' : 'pi-angle-left'"></span>
      <span v-if="!collapsed">Collapse</span>
    </button>
  </aside>
</template>
