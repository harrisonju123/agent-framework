import { onMounted, onUnmounted, ref } from 'vue'
import type { ModalType } from '../types'

export interface KeyboardActions {
  onStart: () => void
  onStop: () => void
  onPause: () => void
  onWork: () => void
  onAnalyze: () => void
  onTicket: () => void
  onRetry: () => void
  onEscape: () => void
}

export function useKeyboard(actions: KeyboardActions) {
  const isInputFocused = ref(false)

  function handleKeyDown(event: KeyboardEvent) {
    // Skip if user is typing in an input/textarea
    const target = event.target as HTMLElement
    if (
      target.tagName === 'INPUT' ||
      target.tagName === 'TEXTAREA' ||
      target.isContentEditable
    ) {
      // Only handle Escape in inputs
      if (event.key === 'Escape') {
        actions.onEscape()
        ;(target as HTMLInputElement).blur()
      }
      return
    }

    // Don't handle if modifier keys are pressed (except for shortcuts)
    if (event.ctrlKey || event.metaKey || event.altKey) {
      return
    }

    switch (event.key.toLowerCase()) {
      case 's':
        event.preventDefault()
        actions.onStart()
        break
      case 'x':
        event.preventDefault()
        actions.onStop()
        break
      case 'p':
        event.preventDefault()
        actions.onPause()
        break
      case 'w':
        event.preventDefault()
        actions.onWork()
        break
      case 'a':
        event.preventDefault()
        actions.onAnalyze()
        break
      case 't':
        event.preventDefault()
        actions.onTicket()
        break
      case 'r':
        event.preventDefault()
        actions.onRetry()
        break
      case 'escape':
        event.preventDefault()
        actions.onEscape()
        break
    }
  }

  onMounted(() => {
    document.addEventListener('keydown', handleKeyDown)
  })

  onUnmounted(() => {
    document.removeEventListener('keydown', handleKeyDown)
  })

  return {
    isInputFocused,
  }
}
