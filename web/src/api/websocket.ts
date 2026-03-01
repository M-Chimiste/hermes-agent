/* WebSocket manager for real-time chat. */

import { getToken } from './client'
import type { ClientMessage, ServerEvent } from './types'

export type WSState = 'connecting' | 'connected' | 'disconnected' | 'error'

export interface WSOptions {
  onEvent: (event: ServerEvent) => void
  onStateChange: (state: WSState) => void
}

export class ChatWebSocket {
  private ws: WebSocket | null = null
  private options: WSOptions
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null

  constructor(options: WSOptions) {
    this.options = options
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return

    this.options.onStateChange('connecting')
    const token = getToken()
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    this.ws = new WebSocket(`${protocol}//${host}/api/chat/ws?token=${token}`)

    this.ws.onopen = () => {
      this.options.onStateChange('connected')
    }

    this.ws.onmessage = (e) => {
      try {
        const event: ServerEvent = JSON.parse(e.data)
        this.options.onEvent(event)
      } catch {
        // ignore malformed messages
      }
    }

    this.ws.onclose = () => {
      this.options.onStateChange('disconnected')
      this.ws = null
    }

    this.ws.onerror = () => {
      this.options.onStateChange('error')
    }
  }

  disconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    this.ws?.close()
    this.ws = null
  }

  send(message: ClientMessage) {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected')
      return
    }
    this.ws.send(JSON.stringify(message))
  }

  sendMessage(content: string, sessionId?: string) {
    this.send({ type: 'message', content, session_id: sessionId })
  }

  sendInterrupt() {
    this.send({ type: 'interrupt' })
  }

  sendClarifyResponse(answer: string) {
    this.send({ type: 'clarify_response', answer })
  }

  get isConnected() {
    return this.ws?.readyState === WebSocket.OPEN
  }
}
