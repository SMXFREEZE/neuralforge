'use client'
import { useEffect, useState } from 'react'
import dynamic from 'next/dynamic'
import Sidebar from '@/components/Sidebar'
import Overview from '@/components/Overview'
import TryIt from '@/components/TryIt'
import Performance from '@/components/Performance'
import Architecture from '@/components/Architecture'
import Quantization from '@/components/Quantization'
import Hls4ml from '@/components/Hls4ml'
import Insights from '@/components/Insights'

const VantaBg = dynamic(() => import('@/components/VantaBg'), { ssr: false })

export type Tab = 'overview' | 'tryit' | 'performance' | 'architecture' | 'quantization' | 'hls4ml' | 'insights'
export type Theme = 'dark' | 'light'

const TABS: Tab[] = ['overview', 'tryit', 'performance', 'architecture', 'quantization', 'hls4ml', 'insights']

export default function Home() {
  const [tab, setTab] = useState<Tab>('overview')
  const [theme, setTheme] = useState<Theme>('dark')
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // Deep-link support: open a tab directly via URL hash, e.g. /#tryit.
  // Deferred so hydration completes before the tab switch (also satisfies
  // react-hooks/set-state-in-effect).
  useEffect(() => {
    const hash = window.location.hash.replace('#', '') as Tab
    if (!TABS.includes(hash)) return
    const id = window.setTimeout(() => setTab(hash), 0)
    return () => window.clearTimeout(id)
  }, [])

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark')

  return (
    <div className="app-shell" data-theme={theme === 'light' ? 'light' : undefined}>
      <VantaBg theme={theme} />

      <div
        className={`sidebar-overlay${sidebarOpen ? ' open' : ''}`}
        onClick={() => setSidebarOpen(false)}
      />

      <Sidebar
        activeTab={tab}
        onTabChange={(t) => { setTab(t); setSidebarOpen(false) }}
        theme={theme}
        onToggleTheme={toggleTheme}
        isOpen={sidebarOpen}
      />

      <div className="main-content">
        <header className="mobile-header">
          <div className="mobile-brand">
            <BrandIcon />
            <span className="mobile-brand-name">NeuralForge</span>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button type="button" className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
              {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
            </button>
            <button type="button" className="hamburger" onClick={() => setSidebarOpen(o => !o)} aria-label="Open menu">
              <HamburgerIcon />
            </button>
          </div>
        </header>

        <div className="main-scroll">
          <div className="main-inner">
            <div className={`view${tab === 'overview' ? ' active' : ''}`}><Overview /></div>
            <div className={`view${tab === 'tryit' ? ' active' : ''}`}><TryIt /></div>
            <div className={`view${tab === 'performance' ? ' active' : ''}`}>
              <Performance active={tab === 'performance'} />
            </div>
            <div className={`view${tab === 'architecture' ? ' active' : ''}`}><Architecture /></div>
            <div className={`view${tab === 'quantization' ? ' active' : ''}`}>
              <Quantization active={tab === 'quantization'} />
            </div>
            <div className={`view${tab === 'hls4ml' ? ' active' : ''}`}><Hls4ml /></div>
            <div className={`view${tab === 'insights' ? ' active' : ''}`}><Insights /></div>
          </div>
        </div>
      </div>
    </div>
  )
}

function BrandIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 28 28" fill="none" style={{ color: 'var(--accent)' }}>
      <rect x="2" y="2" width="10" height="10" rx="2.5" stroke="currentColor" strokeWidth="1.6"/>
      <rect x="16" y="2" width="10" height="10" rx="2.5" stroke="currentColor" strokeWidth="1.6"/>
      <rect x="2" y="16" width="10" height="10" rx="2.5" stroke="currentColor" strokeWidth="1.6"/>
      <rect x="16" y="16" width="10" height="10" rx="2.5" fill="currentColor" opacity="0.35"/>
      <circle cx="21" cy="21" r="2.5" fill="currentColor"/>
    </svg>
  )
}
function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
      <circle cx="12" cy="12" r="5"/>
      <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
      <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
    </svg>
  )
}
function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="14" height="14">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
    </svg>
  )
}
function HamburgerIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="16" height="16">
      <line x1="3" y1="6" x2="21" y2="6"/>
      <line x1="3" y1="12" x2="21" y2="12"/>
      <line x1="3" y1="18" x2="21" y2="18"/>
    </svg>
  )
}
