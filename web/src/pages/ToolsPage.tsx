import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { ToolInfo, ToolsetInfo } from '@/api/types'
import { Wrench, CheckCircle, XCircle } from 'lucide-react'
import { useState } from 'react'
import { cn } from '@/lib/utils'

export function ToolsPage() {
  const [selectedToolset, setSelectedToolset] = useState<string | null>(null)

  const { data: tools, isLoading } = useQuery({
    queryKey: ['tools'],
    queryFn: () => api.get<ToolInfo[]>('/api/tools'),
  })

  const { data: toolsets } = useQuery({
    queryKey: ['toolsets'],
    queryFn: () => api.get<ToolsetInfo[]>('/api/tools/toolsets'),
  })

  const filteredTools = selectedToolset
    ? tools?.filter((t) => t.toolset === selectedToolset)
    : tools

  return (
    <div className="flex flex-col h-full">
      <div className="border-b border-border bg-card px-6 py-4">
        <h1 className="text-lg font-semibold flex items-center gap-2">
          <Wrench size={20} /> Tools
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          {tools?.length ?? 0} tools registered
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        {/* Toolset filter chips */}
        {toolsets && (
          <div className="flex flex-wrap gap-2 mb-4">
            <button
              onClick={() => setSelectedToolset(null)}
              className={cn(
                'px-3 py-1 rounded-full text-xs transition-colors',
                !selectedToolset
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-secondary text-muted-foreground hover:bg-muted',
              )}
            >
              All
            </button>
            {toolsets.map((ts) => (
              <button
                key={ts.name}
                onClick={() =>
                  setSelectedToolset(ts.name === selectedToolset ? null : ts.name)
                }
                className={cn(
                  'px-3 py-1 rounded-full text-xs transition-colors',
                  ts.name === selectedToolset
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-secondary text-muted-foreground hover:bg-muted',
                )}
              >
                {ts.name} ({ts.tools.length})
              </button>
            ))}
          </div>
        )}

        {isLoading ? (
          <p className="text-muted-foreground text-sm">Loading tools...</p>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {filteredTools?.map((tool) => (
              <div
                key={tool.name}
                className="bg-card border border-border rounded-lg p-4"
              >
                <div className="flex items-center justify-between">
                  <h3 className="font-mono text-sm font-medium">
                    {tool.name}
                  </h3>
                  {tool.available ? (
                    <CheckCircle size={14} className="text-success" />
                  ) : (
                    <XCircle size={14} className="text-destructive" />
                  )}
                </div>
                {tool.description && (
                  <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                    {tool.description}
                  </p>
                )}
                <div className="flex items-center gap-2 mt-2">
                  {tool.toolset && (
                    <span className="text-xs bg-secondary px-2 py-0.5 rounded">
                      {tool.toolset}
                    </span>
                  )}
                  {tool.missing_env?.map((env) => (
                    <span
                      key={env}
                      className="text-xs bg-destructive/10 text-destructive px-2 py-0.5 rounded"
                    >
                      {env}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
