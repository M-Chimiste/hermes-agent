import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'
import { Sparkles } from 'lucide-react'
import { useState } from 'react'
import { cn } from '@/lib/utils'

export function SkillsPage() {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [selectedSkill, setSelectedSkill] = useState<string | null>(null)

  const { data: categories, isLoading: catLoading } = useQuery({
    queryKey: ['skills-categories'],
    queryFn: () => api.get<Record<string, unknown>>('/api/skills/categories'),
  })

  const { data: skills } = useQuery({
    queryKey: ['skills', selectedCategory],
    queryFn: () =>
      api.get<Record<string, unknown>>(
        selectedCategory
          ? `/api/skills?category=${selectedCategory}`
          : '/api/skills',
      ),
    enabled: !!selectedCategory,
  })

  const { data: skillDetail } = useQuery({
    queryKey: ['skill-detail', selectedSkill],
    queryFn: () =>
      api.get<Record<string, unknown>>(`/api/skills/${selectedSkill}`),
    enabled: !!selectedSkill,
  })

  const categoryList = categories
    ? Object.entries(categories).filter(([k]) => k !== 'total')
    : []

  return (
    <div className="flex flex-col h-full">
      <div className="border-b border-border bg-card px-6 py-4">
        <h1 className="text-lg font-semibold flex items-center gap-2">
          <Sparkles size={20} /> Skills
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Browse agent skill categories and capabilities
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        {catLoading ? (
          <p className="text-muted-foreground text-sm">Loading...</p>
        ) : (
          <div className="grid gap-6 lg:grid-cols-3">
            {/* Categories */}
            <div className="space-y-2">
              <h2 className="text-sm font-medium text-muted-foreground mb-3">
                Categories
              </h2>
              {categoryList.map(([name, count]) => (
                <button
                  key={name}
                  onClick={() => {
                    setSelectedCategory(name)
                    setSelectedSkill(null)
                  }}
                  className={cn(
                    'w-full text-left px-3 py-2 rounded-md text-sm transition-colors',
                    name === selectedCategory
                      ? 'bg-accent text-accent-foreground'
                      : 'hover:bg-muted text-muted-foreground',
                  )}
                >
                  {name}{' '}
                  <span className="text-xs opacity-60">({String(count)})</span>
                </button>
              ))}
            </div>

            {/* Skills list */}
            <div className="space-y-2">
              {selectedCategory && (
                <>
                  <h2 className="text-sm font-medium text-muted-foreground mb-3">
                    {selectedCategory}
                  </h2>
                  {skills &&
                    (Array.isArray(skills)
                      ? skills
                      : Object.entries(skills).map(([k, v]) => ({
                          name: k,
                          ...(typeof v === 'object' ? v : {}),
                        }))
                    ).map((skill: Record<string, unknown>, i: number) => (
                      <button
                        key={i}
                        onClick={() =>
                          setSelectedSkill(String(skill.name ?? skill.id ?? i))
                        }
                        className={cn(
                          'w-full text-left px-3 py-2 rounded-md text-sm transition-colors',
                          String(skill.name) === selectedSkill
                            ? 'bg-accent text-accent-foreground'
                            : 'hover:bg-muted text-muted-foreground',
                        )}
                      >
                        {String(skill.name ?? skill.id ?? `Skill ${i}`)}
                      </button>
                    ))}
                </>
              )}
            </div>

            {/* Skill detail */}
            <div>
              {selectedSkill && skillDetail && (
                <div className="bg-card border border-border rounded-lg p-4">
                  <h3 className="font-medium mb-2">{selectedSkill}</h3>
                  <pre className="text-xs text-muted-foreground whitespace-pre-wrap overflow-auto max-h-96">
                    {JSON.stringify(skillDetail, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
