'use client'
import { useState, useRef, useEffect } from 'react'

type Mode = 'performance' | 'quantization' | 'architecture' | 'question'

const MODE_LABELS: Record<Mode, string> = {
  performance: 'Performance Analysis',
  quantization: 'Quantization Review',
  architecture: 'Architecture Analysis',
  question: 'Custom Question',
}

export default function Insights() {
  const [mode, setMode] = useState<Mode>('performance')
  const [question, setQuestion] = useState('')
  const [output, setOutput] = useState('')
  const [loading, setLoading] = useState(false)
  const outputRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight
    }
  }, [output])

  async function runAnalysis() {
    setLoading(true)
    setOutput('')
    try {
      const res = await fetch('/api/insights', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode, question }),
      })
      if (!res.ok) throw new Error(await res.text())
      if (!res.body) throw new Error('No stream')

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6)
          if (data === '[DONE]') continue
          try {
            const { text } = JSON.parse(data)
            if (text) setOutput(prev => prev + text)
          } catch { /* ignore parse errors */ }
        }
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Analysis failed'
      setOutput(`**Error:** ${msg}\n\nMake sure OPENAI_API_KEY is configured in your environment.`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">AI Insights</h1>
        <p className="page-subtitle">GPT-4o powered analysis of FPGA performance, quantization, and architecture</p>
      </div>

      <div className="grid-2">
        {/* Controls */}
        <div className="card">
          <div className="card-title">Analysis Configuration</div>
          <div className="insights-form">
            <label style={{ fontSize: 11, color: 'var(--text-3)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', display: 'block', marginBottom: 6 }}>
              Analysis Mode
            </label>
            <select
              className="insights-select"
              value={mode}
              onChange={e => setMode(e.target.value as Mode)}
            >
              {(Object.entries(MODE_LABELS) as [Mode, string][]).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>

            {mode === 'question' && (
              <>
                <label style={{ fontSize: 11, color: 'var(--text-3)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', display: 'block', marginBottom: 6 }}>
                  Your Question
                </label>
                <textarea
                  className="insights-textarea"
                  value={question}
                  onChange={e => setQuestion(e.target.value)}
                  placeholder="e.g. How would this design scale to an 8×8 systolic array?"
                />
              </>
            )}

            <button
              className="btn btn-primary btn-full"
              onClick={runAnalysis}
              disabled={loading || (mode === 'question' && !question.trim())}
            >
              {loading ? (
                <><span className="loading-dots"><span/><span/><span/></span> Analyzing…</>
              ) : (
                <><AnalyzeIcon /> Run Analysis</>
              )}
            </button>
          </div>

          <div className="divider" />

          <div className="card-title" style={{ marginBottom: 12 }}>Quick Prompts</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {[
              { q: 'What would an 8×8 version of this design look like?', label: '8×8 array scaling' },
              { q: 'Compare this to Google TPU v1 architecture.', label: 'TPU comparison' },
              { q: 'How does the ping-pong buffer eliminate pipeline bubbles?', label: 'Ping-pong buffers' },
              { q: 'What is the maximum clock frequency achievable without timing violations?', label: 'Timing analysis' },
            ].map(item => (
              <button
                key={item.label}
                className="btn btn-ghost"
                style={{ justifyContent: 'flex-start', fontSize: 12, textAlign: 'left' }}
                onClick={() => { setMode('question'); setQuestion(item.q) }}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        {/* Output */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="section-header">
            <span className="section-title">Analysis Output</span>
            {output && (
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => navigator.clipboard?.writeText(output)}
                style={{ fontSize: 11 }}
              >
                Copy
              </button>
            )}
          </div>
          <div
            ref={outputRef}
            className="insights-output"
            style={{ flex: 1 }}
          >
            {loading && !output ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: 8, color: 'var(--text-2)' }}>
                <span className="loading-dots"><span/><span/><span/></span>
                <span style={{ fontSize: 13 }}>GPT-4o is analyzing the NeuralForge system…</span>
              </div>
            ) : output ? (
              <MarkdownOutput text={output} />
            ) : (
              <p className="empty-state">Select a mode and click &quot;Run Analysis&quot; to get AI-powered insights</p>
            )}
          </div>
        </div>
      </div>
    </>
  )
}

function MarkdownOutput({ text }: { text: string }) {
  const html = text
    .replace(/^### (.*$)/gm, '<h3>$1</h3>')
    .replace(/^## (.*$)/gm, '<h2>$1</h2>')
    .replace(/^# (.*$)/gm, '<h1>$1</h1>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^\| (.*)/gm, (_, row) => {
      if (row.replace(/[-|: ]/g, '') === '') return '<tr class="sep"/>'
      const cells = row.split('|').map((c: string) => c.trim()).filter(Boolean)
      return '<tr>' + cells.map((c: string) => `<td>${c}</td>`).join('') + '</tr>'
    })
    .replace(/(<tr.*<\/tr>\n?)+/gs, m => `<table>${m}</table>`)
    .replace(/^- (.*)/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/gs, m => `<ul>${m}</ul>`)
    .replace(/\n\n/g, '<br/><br/>')
    .replace(/\n/g, '<br/>')
  return <div dangerouslySetInnerHTML={{ __html: html }} />
}

function AnalyzeIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
    </svg>
  )
}
