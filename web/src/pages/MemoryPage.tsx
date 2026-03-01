import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { MemoryData } from '@/api/types'
import { Brain, Save } from 'lucide-react'
import { useState, useEffect } from 'react'

export function MemoryPage() {
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState<'memory' | 'user'>('memory')
  const [editContent, setEditContent] = useState('')
  const [isDirty, setIsDirty] = useState(false)

  const { data, isLoading } = useQuery({
    queryKey: ['memory', activeTab],
    queryFn: () => api.get<MemoryData>(`/api/memory/${activeTab}`),
  })

  useEffect(() => {
    if (data) {
      setEditContent(data.content)
      setIsDirty(false)
    }
  }, [data])

  const saveMutation = useMutation({
    mutationFn: (content: string) =>
      api.put(`/api/memory/${activeTab}`, { content }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memory', activeTab] })
      setIsDirty(false)
    },
  })

  return (
    <div className="flex flex-col h-full">
      <div className="border-b border-border bg-card px-6 py-4">
        <h1 className="text-lg font-semibold flex items-center gap-2">
          <Brain size={20} /> Memory
        </h1>
        <div className="flex items-center gap-4 mt-3">
          <button
            onClick={() => setActiveTab('memory')}
            className={`text-sm pb-1 border-b-2 transition-colors ${
              activeTab === 'memory'
                ? 'border-primary text-foreground'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            }`}
          >
            MEMORY.md
          </button>
          <button
            onClick={() => setActiveTab('user')}
            className={`text-sm pb-1 border-b-2 transition-colors ${
              activeTab === 'user'
                ? 'border-primary text-foreground'
                : 'border-transparent text-muted-foreground hover:text-foreground'
            }`}
          >
            USER.md
          </button>
          {data && (
            <span className="text-xs text-muted-foreground ml-auto">
              {data.usage} &middot; {data.entry_count} entries
            </span>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        {isLoading ? (
          <p className="text-muted-foreground text-sm">Loading...</p>
        ) : (
          <div className="space-y-4">
            <textarea
              value={editContent}
              onChange={(e) => {
                setEditContent(e.target.value)
                setIsDirty(true)
              }}
              className="w-full h-96 bg-card border border-border rounded-lg p-4 text-sm font-mono resize-y focus:outline-none focus:ring-1 focus:ring-ring"
              placeholder={`No ${activeTab === 'memory' ? 'memory' : 'user profile'} entries yet`}
            />
            {isDirty && (
              <button
                onClick={() => saveMutation.mutate(editContent)}
                disabled={saveMutation.isPending}
                className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm hover:bg-primary/80 disabled:opacity-50"
              >
                <Save size={14} />
                {saveMutation.isPending ? 'Saving...' : 'Save'}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
