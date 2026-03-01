import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import { Server, RefreshCw, CheckCircle, XCircle } from 'lucide-react'

interface CatalogResponse {
  configured: boolean
  models: Array<{
    id: string
    name: string
    server_url: string
    model_id: string
    healthy: boolean
    tags: string[]
    description: string
  }>
}

export function ModelsPage() {
  const queryClient = useQueryClient()

  const { data, isLoading } = useQuery({
    queryKey: ['model-catalog'],
    queryFn: () => api.get<CatalogResponse>('/api/models/catalog'),
  })

  const healthCheck = useMutation({
    mutationFn: () => api.post('/api/models/catalog/health'),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ['model-catalog'] }),
  })

  return (
    <div className="flex flex-col h-full">
      <div className="border-b border-border bg-card px-6 py-4">
        <div className="flex items-center justify-between">
          <h1 className="text-lg font-semibold flex items-center gap-2">
            <Server size={20} /> Model Catalog
          </h1>
          {data?.configured && (
            <button
              onClick={() => healthCheck.mutate()}
              disabled={healthCheck.isPending}
              className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-secondary text-sm hover:bg-muted transition-colors disabled:opacity-50"
            >
              <RefreshCw
                size={14}
                className={healthCheck.isPending ? 'animate-spin' : ''}
              />
              Check Health
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        {isLoading ? (
          <p className="text-muted-foreground text-sm">Loading...</p>
        ) : !data?.configured ? (
          <div className="text-center py-12">
            <Server size={48} className="mx-auto text-muted-foreground/30 mb-4" />
            <h2 className="text-lg font-medium">Model Catalog Not Configured</h2>
            <p className="text-sm text-muted-foreground mt-2 max-w-md mx-auto">
              Add model servers to ~/.hermes/model_catalog.yaml to enable
              multi-server delegation.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {data.models.map((model) => (
              <div
                key={model.id}
                className="bg-card border border-border rounded-lg p-4"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-medium text-sm">{model.name || model.id}</h3>
                    <p className="text-xs text-muted-foreground font-mono mt-0.5">
                      {model.server_url}
                    </p>
                  </div>
                  {model.healthy ? (
                    <CheckCircle size={16} className="text-success" />
                  ) : (
                    <XCircle size={16} className="text-destructive" />
                  )}
                </div>
                {model.description && (
                  <p className="text-xs text-muted-foreground mt-2">
                    {model.description}
                  </p>
                )}
                {model.tags.length > 0 && (
                  <div className="flex gap-1 mt-2">
                    {model.tags.map((tag) => (
                      <span
                        key={tag}
                        className="text-xs bg-secondary px-2 py-0.5 rounded"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
