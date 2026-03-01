import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { ConfigData } from '@/api/types'
import { Settings, Save } from 'lucide-react'
import { useState, useEffect } from 'react'

export function ConfigPage() {
  const queryClient = useQueryClient()
  const [editJson, setEditJson] = useState('')
  const [isDirty, setIsDirty] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const { data, isLoading } = useQuery({
    queryKey: ['config'],
    queryFn: () => api.get<ConfigData>('/api/config'),
  })

  useEffect(() => {
    if (data) {
      setEditJson(JSON.stringify(data.config, null, 2))
      setIsDirty(false)
    }
  }, [data])

  const saveMutation = useMutation({
    mutationFn: async (jsonStr: string) => {
      const parsed = JSON.parse(jsonStr)
      // Flatten into dotted key updates
      const updates: Record<string, unknown> = {}
      function flatten(obj: Record<string, unknown>, prefix = '') {
        for (const [key, val] of Object.entries(obj)) {
          const path = prefix ? `${prefix}.${key}` : key
          if (val && typeof val === 'object' && !Array.isArray(val)) {
            flatten(val as Record<string, unknown>, path)
          } else {
            updates[path] = val
          }
        }
      }
      flatten(parsed)
      return api.patch('/api/config', { updates })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['config'] })
      setIsDirty(false)
      setError(null)
    },
    onError: (err: Error) => setError(err.message),
  })

  const handleSave = () => {
    try {
      JSON.parse(editJson)
      setError(null)
      saveMutation.mutate(editJson)
    } catch {
      setError('Invalid JSON')
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="border-b border-border bg-card px-6 py-4">
        <h1 className="text-lg font-semibold flex items-center gap-2">
          <Settings size={20} /> Configuration
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Edit ~/.hermes/config.yaml (shown as JSON)
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        {isLoading ? (
          <p className="text-muted-foreground text-sm">Loading config...</p>
        ) : (
          <div className="space-y-4">
            <textarea
              value={editJson}
              onChange={(e) => {
                setEditJson(e.target.value)
                setIsDirty(true)
              }}
              className="w-full h-[600px] bg-card border border-border rounded-lg p-4 text-sm font-mono resize-y focus:outline-none focus:ring-1 focus:ring-ring"
              spellCheck={false}
            />
            {error && (
              <p className="text-sm text-destructive">{error}</p>
            )}
            {isDirty && (
              <button
                onClick={handleSave}
                disabled={saveMutation.isPending}
                className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm hover:bg-primary/80 disabled:opacity-50"
              >
                <Save size={14} />
                {saveMutation.isPending ? 'Saving...' : 'Save Configuration'}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
