import { useState } from 'react'
import { setToken } from '@/api/client'

interface TokenSetupProps {
  onTokenSet: () => void
}

export function TokenSetup({ onTokenSet }: TokenSetupProps) {
  const [input, setInput] = useState('')
  const [error, setError] = useState('')

  const handleSubmit = async () => {
    const trimmed = input.trim()
    if (!trimmed) return

    // Test the token against the health endpoint
    try {
      const res = await fetch('/api/health', {
        headers: { Authorization: `Bearer ${trimmed}` },
      })
      if (res.ok) {
        setToken(trimmed)
        onTokenSet()
      } else {
        setError('Invalid token — check the token printed by `hermes serve`')
      }
    } catch {
      setError('Cannot reach API server — is `hermes serve` running?')
    }
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="max-w-md w-full mx-4 space-y-6">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-foreground">Hermes Agent</h1>
          <p className="text-sm text-muted-foreground mt-2">
            Enter the API token shown when you ran{' '}
            <code className="bg-muted px-1.5 py-0.5 rounded text-xs">
              hermes serve
            </code>
          </p>
        </div>
        <div className="space-y-3">
          <input
            type="text"
            value={input}
            onChange={(e) => {
              setInput(e.target.value)
              setError('')
            }}
            onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
            placeholder="Paste API token here..."
            className="w-full bg-card border border-border rounded-lg px-4 py-3 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-ring"
            autoFocus
          />
          {error && <p className="text-sm text-destructive">{error}</p>}
          <button
            onClick={handleSubmit}
            disabled={!input.trim()}
            className="w-full px-4 py-2.5 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/80 disabled:opacity-50 transition-colors"
          >
            Connect
          </button>
        </div>
      </div>
    </div>
  )
}
