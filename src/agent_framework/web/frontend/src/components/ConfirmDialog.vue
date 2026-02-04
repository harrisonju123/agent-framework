<script setup lang="ts">
import { ref, watch, onUnmounted } from 'vue'

const props = defineProps<{
  open: boolean
  title: string
  message: string
  confirmLabel?: string
  cancelLabel?: string
  destructive?: boolean
}>()

const emit = defineEmits<{
  confirm: []
  cancel: []
}>()

const confirmButtonRef = ref<HTMLButtonElement | null>(null)

function handleKeydown(e: KeyboardEvent) {
  if (!props.open) return
  if (e.key === 'Escape') {
    emit('cancel')
  } else if (e.key === 'Enter') {
    emit('confirm')
  }
}

function handleBackdropClick(e: MouseEvent) {
  if (e.target === e.currentTarget) {
    emit('cancel')
  }
}

// Only attach keyboard listener when dialog is open
watch(() => props.open, (isOpen) => {
  if (isOpen) {
    document.addEventListener('keydown', handleKeydown)
    // Focus confirm button when dialog opens
    setTimeout(() => confirmButtonRef.value?.focus(), 50)
  } else {
    document.removeEventListener('keydown', handleKeydown)
  }
}, { immediate: true })

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
})
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
        :aria-labelledby="'confirm-title'"
        :aria-describedby="'confirm-message'"
      >
        <div class="bg-gray-900 border border-gray-700 w-full max-w-sm font-mono">
          <!-- Header -->
          <div class="px-4 py-3 border-b border-gray-800">
            <h2 id="confirm-title" class="text-sm text-gray-200 font-medium">{{ title }}</h2>
          </div>

          <!-- Content -->
          <div class="px-4 py-4">
            <p id="confirm-message" class="text-sm text-gray-400">{{ message }}</p>
          </div>

          <!-- Actions -->
          <div class="flex justify-end gap-2 px-4 py-3 border-t border-gray-800 bg-gray-900/50">
            <button
              @click="emit('cancel')"
              class="px-4 py-1.5 text-sm text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded transition-colors"
            >
              {{ cancelLabel || 'Cancel' }}
            </button>
            <button
              ref="confirmButtonRef"
              @click="emit('confirm')"
              class="px-4 py-1.5 text-sm rounded transition-colors"
              :class="destructive
                ? 'bg-red-600 hover:bg-red-500 text-white'
                : 'bg-cyan-600 hover:bg-cyan-500 text-white'"
            >
              {{ confirmLabel || 'Confirm' }}
            </button>
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

.modal-enter-active > div,
.modal-leave-active > div {
  transition: transform 0.15s ease;
}

.modal-enter-from > div,
.modal-leave-to > div {
  transform: scale(0.95);
}
</style>
