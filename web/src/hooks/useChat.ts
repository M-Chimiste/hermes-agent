/* Chat state machine hook — manages WebSocket connection and message state. */

import { useCallback, useEffect, useRef, useState } from 'react'
import { ChatWebSocket, type WSState } from '@/api/websocket'
import type { ChatMessage, ServerEvent } from '@/api/types'

let msgIdCounter = 0
function nextId() {
  return `msg_${++msgIdCounter}_${Date.now()}`
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [wsState, setWsState] = useState<WSState>('disconnected')
  const [isProcessing, setIsProcessing] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [clarifyPrompt, setClarifyPrompt] = useState<{
    question: string
    choices?: string[]
  } | null>(null)

  const wsRef = useRef<ChatWebSocket | null>(null)

  const handleEvent = useCallback((event: ServerEvent) => {
    switch (event.type) {
      case 'session_info':
        if (event.session_id) setSessionId(event.session_id)
        break

      case 'status':
        if (event.status === 'processing') {
          setIsProcessing(true)
        } else if (event.status === 'done') {
          setIsProcessing(false)
        }
        break

      case 'tool_progress':
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: 'tool_progress',
            content: `Using **${event.tool_name}**`,
            tool_name: event.tool_name,
            tool_preview: event.preview,
            timestamp: Date.now(),
          },
        ])
        break

      case 'response':
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: 'assistant',
            content: event.content ?? '',
            timestamp: Date.now(),
          },
        ])
        setIsProcessing(false)
        break

      case 'clarify':
        setClarifyPrompt({
          question: event.question ?? '',
          choices: event.choices,
        })
        break

      case 'error':
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: 'system',
            content: `Error: ${event.message}`,
            timestamp: Date.now(),
          },
        ])
        setIsProcessing(false)
        break
    }
  }, [])

  // Connect on mount
  useEffect(() => {
    const ws = new ChatWebSocket({
      onEvent: handleEvent,
      onStateChange: setWsState,
    })
    wsRef.current = ws
    ws.connect()

    return () => {
      ws.disconnect()
    }
  }, [handleEvent])

  const sendMessage = useCallback(
    (content: string) => {
      if (!content.trim()) return

      // Optimistic UI: add user message immediately
      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          role: 'user',
          content,
          timestamp: Date.now(),
        },
      ])

      wsRef.current?.sendMessage(content, sessionId ?? undefined)
    },
    [sessionId],
  )

  const interrupt = useCallback(() => {
    wsRef.current?.sendInterrupt()
  }, [])

  const respondToClarify = useCallback(
    (answer: string) => {
      setClarifyPrompt(null)
      wsRef.current?.sendClarifyResponse(answer)
    },
    [],
  )

  const clearMessages = useCallback(() => {
    setMessages([])
    setSessionId(null)
  }, [])

  return {
    messages,
    wsState,
    isProcessing,
    sessionId,
    clarifyPrompt,
    sendMessage,
    interrupt,
    respondToClarify,
    clearMessages,
  }
}
