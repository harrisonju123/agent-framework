<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted, computed } from 'vue'

const props = defineProps<{
  open: boolean
  title: string
}>()

const emit = defineEmits<{
  close: []
}>()

// Generate unique ID for ARIA
const titleId = computed(() => `modal-title-${Math.random().toString(36).slice(2, 9)}`)

// Handle escape key
function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape') {
    emit('close')
  }
}

onMounted(() => {
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
})

// Handle backdrop click
function handleBackdropClick(e: MouseEvent) {
  if (e.target === e.currentTarget) {
    emit('close')
  }
}
</script>

<template>
  <Teleport to="body">
    <Transition name="modal">
      <div
        v-if="open"
        @click="handleBackdropClick"
        class="fixed inset-0 bg-black/70 flex items-center justify-center z-50"
        role="dialog"
        aria-modal="true"
        :aria-labelledby="titleId"
      >
        <div class="bg-gray-900 border border-gray-700 w-full max-w-lg font-mono">
          <!-- Header -->
          <div class="flex items-center justify-between px-4 py-2 border-b border-gray-800">
            <h2 :id="titleId" class="text-sm text-gray-300">{{ title }}</h2>
            <button
              @click="emit('close')"
              class="text-gray-500 hover:text-gray-300 text-xs"
              aria-label="Close dialog"
            >
              [ESC]
            </button>
          </div>

          <!-- Content -->
          <div class="p-4">
            <slot></slot>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.modal-enter-active,
.modal-leave-active {
  transition: opacity 0.15s ease;
}

.modal-enter-from,
.modal-leave-to {
  opacity: 0;
}

.modal-enter-active .bg-gray-900,
.modal-leave-active .bg-gray-900 {
  transition: transform 0.15s ease;
}

.modal-enter-from .bg-gray-900,
.modal-leave-to .bg-gray-900 {
  transform: scale(0.95);
}
</style>
