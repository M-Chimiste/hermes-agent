import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { SessionSummary } from '@/api/types'
import { History, Search, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { cn } from '@/lib/utils'

export function SessionsPage() {
  const [search, setSearch] = useState('')

  const { data: sessions, isLoading, refetch } = useQuery({
    queryKey: ['sessions'],
    queryFn: () => api.get<SessionSummary[]>('/api/sessions?limit=50'),
  })

  const searchResults = useQuery({
    queryKey: ['sessions-search', search],
    queryFn: () =>
      api.post<unknown[]>('/api/sessions/search', { query: search, limit: 20 }),
    enabled: search.length > 2,
  })

  const handleDelete = async (id: string) => {
    await api.delete(`/api/sessions/${id}`)
    refetch()
  }

  const formatDate = (ts: number) =>
    new Date(ts * 1000).toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })

  return (
    <div className="flex flex-col h-full">
      <div className="border-b border-border bg-card px-6 py-4">
        <h1 className="text-lg font-semibold flex items-center gap-2">
          <History size={20} /> Sessions
        </h1>
        <div className="mt-3 relative">
          <Search
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground"
          />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search sessions..."
            className="w-full pl-9 pr-4 py-2 bg-secondary rounded-lg text-sm focus:outline-none focus:ring-1 focus:ring-ring"
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        {isLoading ? (
          <p className="text-muted-foreground text-sm">Loading sessions...</p>
        ) : search.length > 2 && searchResults.data ? (
          <div className="space-y-2">
            <p className="text-xs text-muted-foreground mb-3">
              Search results for &ldquo;{search}&rdquo;
            </p>
            {(searchResults.data as Array<Record<string, unknown>>).map((r, i) => (
              <div key={i} className="bg-card border border-border rounded-lg p-4">
                <p className="text-sm">{String(r.snippet ?? '')}</p>
                <p className="text-xs text-muted-foreground mt-1">
                  Session: {String(r.session_id ?? '')} &middot; {String(r.source ?? '')}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-2">
            {sessions?.map((s) => (
              <div
                key={s.id}
                className="bg-card border border-border rounded-lg p-4 flex items-center justify-between hover:border-muted-foreground/30 transition-colors"
              >
                <div className="min-w-0">
                  <p className="text-sm font-mono truncate">{s.id}</p>
                  <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                    <span
                      className={cn(
                        'px-1.5 py-0.5 rounded text-xs',
                        s.source === 'cli'
                          ? 'bg-primary/10 text-primary'
                          : 'bg-accent text-accent-foreground',
                      )}
                    >
                      {s.source}
                    </span>
                    <span>{s.message_count} msgs</span>
                    <span>{s.tool_call_count} tools</span>
                    <span>{formatDate(s.started_at)}</span>
                    {s.model && <span className="truncate max-w-[200px]">{s.model}</span>}
                  </div>
                </div>
                <button
                  onClick={() => handleDelete(s.id)}
                  className="p-2 rounded hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
            {sessions?.length === 0 && (
              <p className="text-muted-foreground text-sm text-center py-8">
                No sessions yet
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
