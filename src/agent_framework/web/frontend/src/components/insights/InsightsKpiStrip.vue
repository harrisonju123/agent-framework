<script setup lang="ts">
defineProps<{
  cards: Array<{ label: string; value: string; subtitle: string; color?: string }>
}>()
</script>

<template>
  <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
    <div
      v-for="card in cards"
      :key="card.label"
      class="bg-white border border-slate-200 rounded-xl px-4 py-4"
    >
      <p class="text-xs font-medium text-slate-400 uppercase tracking-wide">{{ card.label }}</p>
      <p class="text-2xl font-semibold mt-1" :class="card.color || 'text-slate-800'">{{ card.value }}</p>
      <p class="text-xs text-slate-400 mt-1">{{ card.subtitle }}</p>
    </div>
  </div>
</template>
