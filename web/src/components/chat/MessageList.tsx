import { useEffect, useRef } from 'react'
import type { ChatMessage } from '@/api/types'
import { MessageBubble } from './MessageBubble'

interface MessageListProps {
  messages: ChatMessage[]
  isProcessing: boolean
}

export function MessageList({ messages, isProcessing }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isProcessing])

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center space-y-3">
          <p className="text-4xl">&#x2728;</p>
          <h2 className="text-xl font-semibold text-foreground">
            Hermes Agent
          </h2>
          <p className="text-sm text-muted-foreground max-w-md">
            Start a conversation with your AI agent. It can use tools, search
            the web, execute code, and more.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-4xl mx-auto divide-y divide-border/30">
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        {isProcessing && (
          <div className="flex items-center gap-3 px-6 py-4">
            <div className="w-7 h-7 rounded-full bg-accent flex items-center justify-center flex-shrink-0">
              <span className="animate-pulse text-xs">&#x2728;</span>
            </div>
            <div className="flex gap-1">
              <span className="w-2 h-2 bg-muted-foreground/50 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-2 h-2 bg-muted-foreground/50 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-2 h-2 bg-muted-foreground/50 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          </div>
        )}
      </div>
      <div ref={bottomRef} />
    </div>
  )
}
