import { cn } from '@/lib/utils'
import type { WSState } from '@/api/websocket'

const stateColors: Record<WSState, string> = {
  connected: 'bg-success',
  connecting: 'bg-warning',
  disconnected: 'bg-muted-foreground',
  error: 'bg-destructive',
}

const stateLabels: Record<WSState, string> = {
  connected: 'Connected',
  connecting: 'Connecting...',
  disconnected: 'Disconnected',
  error: 'Error',
}

interface HeaderProps {
  wsState: WSState
  sessionId: string | null
}

export function Header({ wsState, sessionId }: HeaderProps) {
  return (
    <header className="flex items-center justify-between px-6 py-3 border-b border-border bg-card">
      <div className="flex items-center gap-3">
        {sessionId && (
          <span className="text-xs text-muted-foreground font-mono">
            {sessionId}
          </span>
        )}
      </div>
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span
          className={cn('w-2 h-2 rounded-full', stateColors[wsState])}
        />
        {stateLabels[wsState]}
      </div>
    </header>
  )
}
