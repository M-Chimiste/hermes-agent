import { cn } from '@/lib/utils'
import type { ChatMessage } from '@/api/types'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { User, Bot, Wrench, AlertCircle } from 'lucide-react'

interface MessageBubbleProps {
  message: ChatMessage
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const { role, content, tool_name, tool_preview } = message

  if (role === 'tool_progress') {
    return (
      <div className="flex items-start gap-3 px-6 py-2">
        <div className="w-7 h-7 rounded-full bg-warning/10 flex items-center justify-center flex-shrink-0 mt-0.5">
          <Wrench size={14} className="text-warning" />
        </div>
        <div className="min-w-0">
          <p className="text-sm text-muted-foreground">
            Using <span className="font-medium text-warning">{tool_name}</span>
          </p>
          {tool_preview && (
            <pre className="mt-1 text-xs text-muted-foreground/70 bg-muted/50 rounded px-2 py-1 overflow-x-auto max-w-2xl">
              <code>{tool_preview}</code>
            </pre>
          )}
        </div>
      </div>
    )
  }

  if (role === 'system') {
    return (
      <div className="flex items-start gap-3 px-6 py-2">
        <div className="w-7 h-7 rounded-full bg-destructive/10 flex items-center justify-center flex-shrink-0 mt-0.5">
          <AlertCircle size={14} className="text-destructive" />
        </div>
        <p className="text-sm text-destructive/80">{content}</p>
      </div>
    )
  }

  const isUser = role === 'user'

  return (
    <div
      className={cn(
        'flex items-start gap-3 px-6 py-4',
        isUser ? 'bg-transparent' : 'bg-card/50',
      )}
    >
      <div
        className={cn(
          'w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5',
          isUser ? 'bg-primary/10' : 'bg-accent',
        )}
      >
        {isUser ? (
          <User size={14} className="text-primary" />
        ) : (
          <Bot size={14} className="text-accent-foreground" />
        )}
      </div>
      <div className="min-w-0 flex-1 prose prose-invert prose-sm max-w-none">
        {isUser ? (
          <p className="whitespace-pre-wrap">{content}</p>
        ) : (
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              pre: ({ children }) => (
                <pre className="bg-muted rounded-lg p-4 overflow-x-auto">
                  {children}
                </pre>
              ),
              code: ({ children, className }) => {
                const isInline = !className
                return isInline ? (
                  <code className="bg-muted px-1.5 py-0.5 rounded text-sm">
                    {children}
                  </code>
                ) : (
                  <code className={className}>{children}</code>
                )
              },
            }}
          >
            {content}
          </ReactMarkdown>
        )}
      </div>
    </div>
  )
}
