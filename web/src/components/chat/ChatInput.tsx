import { useState, useRef, useCallback } from 'react'
import { Send, Square } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ChatInputProps {
  onSend: (content: string) => void
  onInterrupt: () => void
  isProcessing: boolean
  disabled: boolean
}

export function ChatInput({
  onSend,
  onInterrupt,
  isProcessing,
  disabled,
}: ChatInputProps) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = useCallback(() => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }, [value, disabled, onSend])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        if (isProcessing) {
          onInterrupt()
        } else {
          handleSubmit()
        }
      }
    },
    [handleSubmit, isProcessing, onInterrupt],
  )

  const handleInput = useCallback(() => {
    const el = textareaRef.current
    if (el) {
      el.style.height = 'auto'
      el.style.height = Math.min(el.scrollHeight, 200) + 'px'
    }
  }, [])

  return (
    <div className="border-t border-border bg-card px-4 py-3">
      <div className="max-w-4xl mx-auto flex items-end gap-2">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          placeholder={
            isProcessing
              ? 'Press Enter to interrupt...'
              : 'Message Hermes...'
          }
          rows={1}
          className={cn(
            'flex-1 resize-none bg-secondary rounded-lg px-4 py-2.5 text-sm',
            'placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring',
            'min-h-[40px] max-h-[200px]',
          )}
          disabled={disabled}
        />
        <button
          onClick={isProcessing ? onInterrupt : handleSubmit}
          disabled={disabled || (!isProcessing && !value.trim())}
          className={cn(
            'p-2.5 rounded-lg transition-colors flex-shrink-0',
            isProcessing
              ? 'bg-destructive/10 text-destructive hover:bg-destructive/20'
              : 'bg-primary text-primary-foreground hover:bg-primary/80',
            'disabled:opacity-50 disabled:cursor-not-allowed',
          )}
        >
          {isProcessing ? <Square size={18} /> : <Send size={18} />}
        </button>
      </div>
    </div>
  )
}
