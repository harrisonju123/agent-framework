import { ref, onMounted, onUnmounted } from 'vue'
import type { LogEntry } from '../types'

export interface LogStreamOptions {
  maxLines?: number
  flushInterval?: number
}

export function useLogStream(options: LogStreamOptions = {}) {
  const { maxLines = 200, flushInterval = 200 } = options

  const logs = ref<LogEntry[]>([])
  const connected = ref(false)
  const error = ref<string | null>(null)
  let ws: WebSocket | null = null
  let reconnectTimeout: ReturnType<typeof setTimeout> | null = null
  let reconnectAttempts = 0
  const maxReconnectAttempts = 10

  let logIdCounter = 0

  // Non-reactive buffer collects messages between flushes
  let pendingBuffer: LogEntry[] = []
  let flushTimer: ReturnType<typeof setInterval> | null = null

  function flush() {
    if (pendingBuffer.length === 0) return

    const incoming = pendingBuffer
    pendingBuffer = []

    const merged = logs.value.concat(incoming)
    logs.value = merged.length > maxLines
      ? merged.slice(-maxLines)
      : merged
  }

  function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/logs`

    try {
      ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        connected.value = true
        error.value = null
        reconnectAttempts = 0
        flushTimer = setInterval(flush, flushInterval)
        console.log('Log stream WebSocket connected')
      }

      ws.onmessage = (event) => {
        try {
          const entry = JSON.parse(event.data) as LogEntry
          entry.id = logIdCounter++
          pendingBuffer.push(entry)
        } catch (e) {
          console.error('Failed to parse log message:', e)
        }
      }

      ws.onclose = () => {
        connected.value = false
        stopFlushTimer()
        // Flush any remaining buffered messages so they aren't lost
        flush()
        console.log('Log stream WebSocket disconnected')

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

  function stopFlushTimer() {
    if (flushTimer) {
      clearInterval(flushTimer)
      flushTimer = null
    }
  }

  function disconnect() {
    stopFlushTimer()
    flush()
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
    pendingBuffer = []
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
