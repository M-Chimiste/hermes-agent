import { useState } from 'react'

interface ClarifyDialogProps {
  question: string
  choices?: string[]
  onRespond: (answer: string) => void
}

export function ClarifyDialog({ question, choices, onRespond }: ClarifyDialogProps) {
  const [customAnswer, setCustomAnswer] = useState('')

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-card border border-border rounded-lg p-6 max-w-md w-full mx-4 space-y-4">
        <h3 className="font-semibold text-foreground">Agent needs your input</h3>
        <p className="text-sm text-muted-foreground">{question}</p>

        {choices && choices.length > 0 ? (
          <div className="space-y-2">
            {choices.map((choice, i) => (
              <button
                key={i}
                onClick={() => onRespond(choice)}
                className="w-full text-left px-4 py-2 rounded-md bg-secondary hover:bg-accent text-sm transition-colors"
              >
                {choice}
              </button>
            ))}
          </div>
        ) : null}

        <div className="flex gap-2">
          <input
            type="text"
            value={customAnswer}
            onChange={(e) => setCustomAnswer(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && customAnswer.trim()) {
                onRespond(customAnswer.trim())
              }
            }}
            placeholder="Type your answer..."
            className="flex-1 bg-secondary rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
          />
          <button
            onClick={() => onRespond(customAnswer.trim())}
            disabled={!customAnswer.trim()}
            className="px-4 py-2 rounded-md bg-primary text-primary-foreground text-sm hover:bg-primary/80 disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  )
}
