import { onMounted, onUnmounted, ref } from 'vue'

export interface KeyboardActions {
  onStart: () => void
  onStop: () => void
  onPause: () => void
  onWork: () => void
  onAnalyze: () => void
  onTicket: () => void
  onRetry: () => void
  onApprove: () => void
  onEscape: () => void
  onNavigate?: (page: number) => void
}

export function useKeyboard(actions: KeyboardActions) {
  const isInputFocused = ref(false)

  function handleKeyDown(event: KeyboardEvent) {
    const target = event.target as HTMLElement
    if (
      target.tagName === 'INPUT' ||
      target.tagName === 'TEXTAREA' ||
      target.isContentEditable
    ) {
      if (event.key === 'Escape') {
        actions.onEscape()
        ;(target as HTMLInputElement).blur()
      }
      return
    }

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
      case 'c':
        event.preventDefault()
        actions.onApprove()
        break
      case 'escape':
        event.preventDefault()
        actions.onEscape()
        break
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
        if (actions.onNavigate) {
          event.preventDefault()
          actions.onNavigate(parseInt(event.key))
        }
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
