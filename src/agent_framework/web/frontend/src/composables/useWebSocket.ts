import { ref, onMounted, onUnmounted } from 'vue'
import type { DashboardState } from '../types'

export function useWebSocket() {
  const state = ref<DashboardState | null>(null)
  const connected = ref(false)
  const error = ref<string | null>(null)
  const reconnecting = ref(false)
  const reconnectAttempt = ref(0)
  let ws: WebSocket | null = null
  let reconnectTimeout: ReturnType<typeof setTimeout> | null = null
  let reconnectAttempts = 0
  const maxReconnectAttempts = 10

  function connect() {
    // Determine WebSocket URL based on current host
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws`

    try {
      ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        connected.value = true
        error.value = null
        reconnecting.value = false
        reconnectAttempt.value = 0
        reconnectAttempts = 0
        console.log('WebSocket connected')
      }

      ws.onmessage = (event) => {
        try {
          state.value = JSON.parse(event.data) as DashboardState
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e)
        }
      }

      ws.onclose = () => {
        connected.value = false
        console.log('WebSocket disconnected')

        // Auto-reconnect with exponential backoff
        if (reconnectAttempts < maxReconnectAttempts) {
          reconnecting.value = true
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000)
          reconnectAttempts++
          reconnectAttempt.value = reconnectAttempts
          console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts})...`)
          reconnectTimeout = setTimeout(connect, delay)
        } else {
          reconnecting.value = false
          error.value = 'Connection lost. Please refresh the page.'
        }
      }

      ws.onerror = (e) => {
        console.error('WebSocket error:', e)
        error.value = 'WebSocket connection error'
      }
    } catch (e) {
      console.error('Failed to create WebSocket:', e)
      error.value = 'Failed to connect to server'
    }
  }

  function disconnect() {
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout)
      reconnectTimeout = null
    }
    if (ws) {
      ws.close()
      ws = null
    }
  }

  onMounted(() => {
    connect()
  })

  onUnmounted(() => {
    disconnect()
  })

  return {
    state,
    connected,
    error,
    reconnecting,
    reconnectAttempt,
    reconnect: () => {
      disconnect()
      reconnectAttempts = 0
      reconnectAttempt.value = 0
      reconnecting.value = false
      connect()
    },
  }
}
