import { NavLink } from 'react-router-dom'
import {
  MessageSquare,
  History,
  Wrench,
  Brain,
  Settings,
  Sparkles,
  Server,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useState } from 'react'

const navItems = [
  { to: '/', icon: MessageSquare, label: 'Chat' },
  { to: '/sessions', icon: History, label: 'Sessions' },
  { to: '/tools', icon: Wrench, label: 'Tools' },
  { to: '/memory', icon: Brain, label: 'Memory' },
  { to: '/config', icon: Settings, label: 'Config' },
  { to: '/skills', icon: Sparkles, label: 'Skills' },
  { to: '/models', icon: Server, label: 'Models' },
]

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <aside
      className={cn(
        'flex flex-col border-r border-border bg-card transition-all duration-200',
        collapsed ? 'w-16' : 'w-56',
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-4 border-b border-border">
        {!collapsed && (
          <span className="text-lg font-bold text-primary tracking-tight">
            Hermes
          </span>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-1 rounded hover:bg-muted text-muted-foreground"
        >
          {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-2 space-y-1 px-2">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors',
                isActive
                  ? 'bg-accent text-accent-foreground font-medium'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground',
              )
            }
          >
            <Icon size={18} />
            {!collapsed && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      {!collapsed && (
        <div className="px-4 py-3 border-t border-border text-xs text-muted-foreground">
          Hermes Agent
        </div>
      )}
    </aside>
  )
}
