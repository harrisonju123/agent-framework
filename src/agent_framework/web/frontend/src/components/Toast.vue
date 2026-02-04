<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue'

const props = defineProps<{
  message: string
  type: 'success' | 'error' | 'info'
  duration?: number
}>()

const emit = defineEmits<{
  dismiss: []
}>()

const visible = ref(true)
let timeoutId: ReturnType<typeof setTimeout> | null = null

function dismiss() {
  visible.value = false
  emit('dismiss')
}

function startTimer() {
  if (props.duration !== 0) {
    timeoutId = setTimeout(dismiss, props.duration ?? 3000)
  }
}

onMounted(() => {
  startTimer()
})

onUnmounted(() => {
  if (timeoutId) {
    clearTimeout(timeoutId)
  }
})

watch(() => props.message, () => {
  visible.value = true
  if (timeoutId) {
    clearTimeout(timeoutId)
  }
  startTimer()
})

const typeClasses = {
  success: 'bg-green-900/90 border-green-500 text-green-300',
  error: 'bg-red-900/90 border-red-500 text-red-300',
  info: 'bg-cyan-900/90 border-cyan-500 text-cyan-300',
}
</script>

<template>
  <Teleport to="body">
    <Transition name="toast">
      <div
        v-if="visible"
        role="alert"
        aria-live="polite"
        class="fixed top-4 right-4 z-50 flex items-center gap-3 px-4 py-3 border rounded-lg shadow-lg font-mono text-sm max-w-md"
        :class="typeClasses[type]"
      >
        <span aria-hidden="true">
          <template v-if="type === 'success'">+</template>
          <template v-else-if="type === 'error'">!</template>
          <template v-else>i</template>
        </span>
        <span class="flex-1">{{ message }}</span>
        <button
          @click="dismiss"
          class="text-current opacity-60 hover:opacity-100 ml-2"
          aria-label="Dismiss notification"
        >
          x
        </button>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.toast-enter-active,
.toast-leave-active {
  transition: all 0.3s ease;
}

.toast-enter-from {
  opacity: 0;
  transform: translateX(100%);
}

.toast-leave-to {
  opacity: 0;
  transform: translateY(-20px);
}
</style>
