'use client'
import type { Tab, Theme } from '@/app/page'

const NAV_ITEMS: { tab: Tab; label: string; icon: React.ReactNode }[] = [
  {
    tab: 'overview', label: 'Overview',
    icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/></svg>,
  },
  {
    tab: 'tryit', label: 'Try It',
    icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 19l7-7 3 3-7 7-3-3z"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/><path d="M2 2l7.586 7.586"/><circle cx="11" cy="11" r="2"/></svg>,
  },
  {
    tab: 'performance', label: 'Performance',
    icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>,
  },
  {
    tab: 'architecture', label: 'Architecture',
    icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>,
  },
  {
    tab: 'quantization', label: 'Quantization',
    icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>,
  },
  {
    tab: 'hls4ml', label: 'hls4ml',
    icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>,
  },
  {
    tab: 'insights', label: 'AI Insights',
    icon: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>,
  },
]

const SECTIONS = [
  { label: 'Dashboard', tabs: ['overview', 'tryit', 'performance'] as Tab[] },
  { label: 'Hardware', tabs: ['architecture', 'quantization'] as Tab[] },
  { label: 'Tools', tabs: ['hls4ml', 'insights'] as Tab[] },
]

interface SidebarProps {
  activeTab: Tab
  onTabChange: (tab: Tab) => void
  theme: Theme
  onToggleTheme: () => void
  isOpen: boolean
}

export default function Sidebar({ activeTab, onTabChange, theme, onToggleTheme, isOpen }: SidebarProps) {
  return (
    <aside className={`sidebar${isOpen ? ' open' : ''}`}>
      <div className="sidebar-brand">
        <svg className="brand-icon" viewBox="0 0 28 28" fill="none">
          <rect x="2" y="2" width="10" height="10" rx="2.5" stroke="currentColor" strokeWidth="1.6"/>
          <rect x="16" y="2" width="10" height="10" rx="2.5" stroke="currentColor" strokeWidth="1.6"/>
          <rect x="2" y="16" width="10" height="10" rx="2.5" stroke="currentColor" strokeWidth="1.6"/>
          <rect x="16" y="16" width="10" height="10" rx="2.5" fill="currentColor" opacity="0.35"/>
          <circle cx="21" cy="21" r="2.5" fill="currentColor"/>
        </svg>
        <span className="brand-name">NeuralForge</span>
      </div>

      <nav className="sidebar-nav">
        {SECTIONS.map(section => (
          <div className="nav-section" key={section.label}>
            <div className="nav-section-label">{section.label}</div>
            {section.tabs.map(tabId => {
              const item = NAV_ITEMS.find(n => n.tab === tabId)!
              return (
                <button
                  key={tabId}
                  className={`nav-link${activeTab === tabId ? ' active' : ''}`}
                  onClick={() => onTabChange(tabId)}
                >
                  <div className="nav-icon-wrap">{item.icon}</div>
                  <span className="nav-label">{item.label}</span>
                </button>
              )
            })}
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="status-chip">
          <span className="status-dot" />
          <span>OpenAI Live</span>
        </div>
        <button className="theme-toggle" onClick={onToggleTheme} aria-label="Toggle theme">
          {theme === 'dark'
            ? <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
            : <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
          }
        </button>
      </div>
    </aside>
  )
}
