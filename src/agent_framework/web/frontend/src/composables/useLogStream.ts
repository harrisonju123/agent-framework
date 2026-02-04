import { ref, onMounted, onUnmounted } from 'vue'
import type { LogEntry } from '../types'

export interface LogStreamOptions {
  maxLines?: number
  autoScroll?: boolean
}

export function useLogStream(options: LogStreamOptions = {}) {
  const { maxLines = 500 } = options

  const logs = ref<LogEntry[]>([])
  const connected = ref(false)
  const error = ref<string | null>(null)
  let ws: WebSocket | null = null
  let reconnectTimeout: ReturnType<typeof setTimeout> | null = null
  let reconnectAttempts = 0
  const maxReconnectAttempts = 10

  // Generate unique IDs for log entries
  let logIdCounter = 0

  function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/logs`

    try {
      ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        connected.value = true
        error.value = null
        reconnectAttempts = 0
        console.log('Log stream WebSocket connected')
      }

      ws.onmessage = (event) => {
        try {
          const entry = JSON.parse(event.data) as LogEntry
          // Add unique ID for Vue key binding
          const entryWithId = {
            ...entry,
            id: logIdCounter++,
          }
          logs.value.push(entryWithId as LogEntry)

          // Trim logs to maxLines
          if (logs.value.length > maxLines) {
            logs.value = logs.value.slice(-maxLines)
          }
        } catch (e) {
          console.error('Failed to parse log message:', e)
        }
      }

      ws.onclose = () => {
        connected.value = false
        console.log('Log stream WebSocket disconnected')

        // Auto-reconnect
        if (reconnectAttempts < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000)
          reconnectAttempts++
          console.log(`Reconnecting log stream in ${delay}ms...`)
          reconnectTimeout = setTimeout(connect, delay)
        } else {
          error.value = 'Log stream connection lost'
        }
      }

      ws.onerror = (e) => {
        console.error('Log stream WebSocket error:', e)
        error.value = 'Log stream connection error'
      }
    } catch (e) {
      console.error('Failed to create log stream WebSocket:', e)
      error.value = 'Failed to connect to log stream'
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

  function clear() {
    logs.value = []
  }

  onMounted(() => {
    connect()
  })

  onUnmounted(() => {
    disconnect()
  })

  return {
    logs,
    connected,
    error,
    clear,
    reconnect: () => {
      disconnect()
      reconnectAttempts = 0
      connect()
    },
  }
}
