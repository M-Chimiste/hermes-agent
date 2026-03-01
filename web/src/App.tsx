import { Routes, Route, Navigate } from 'react-router-dom'
import { Sidebar } from '@/components/layout/Sidebar'
import { ChatPage } from '@/pages/ChatPage'
import { SessionsPage } from '@/pages/SessionsPage'
import { ToolsPage } from '@/pages/ToolsPage'
import { MemoryPage } from '@/pages/MemoryPage'
import { ConfigPage } from '@/pages/ConfigPage'
import { SkillsPage } from '@/pages/SkillsPage'
import { ModelsPage } from '@/pages/ModelsPage'
import { TokenSetup } from '@/components/TokenSetup'
import { getToken } from '@/api/client'
import { useState } from 'react'

export function App() {
  const [hasToken, setHasToken] = useState(() => !!getToken())

  if (!hasToken) {
    return <TokenSetup onTokenSet={() => setHasToken(true)} />
  }

  return (
    <div className="flex h-screen bg-background">
      <Sidebar />
      <main className="flex-1 min-w-0">
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/sessions" element={<SessionsPage />} />
          <Route path="/tools" element={<ToolsPage />} />
          <Route path="/memory" element={<MemoryPage />} />
          <Route path="/config" element={<ConfigPage />} />
          <Route path="/skills" element={<SkillsPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  )
}
