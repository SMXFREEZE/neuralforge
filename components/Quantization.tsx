'use client'
import { useEffect, useRef } from 'react'

const QUANT_LAYERS = [
  { name: 'conv1_weight', shape: '[6,1,5,5]', params: '150', scale: '0.01234', mse: '1.2e-5', bits: 8 },
  { name: 'conv1_bias', shape: '[6]', params: '6', scale: '0.00089', mse: '4.1e-7', bits: 8 },
  { name: 'conv2_weight', shape: '[16,6,5,5]', params: '2,400', scale: '0.00891', mse: '2.7e-5', bits: 8 },
  { name: 'conv2_bias', shape: '[16]', params: '16', scale: '0.00112', mse: '6.3e-7', bits: 8 },
  { name: 'fc1_weight', shape: '[120,400]', params: '48,000', scale: '0.00762', mse: '3.1e-5', bits: 8 },
  { name: 'fc1_bias', shape: '[120]', params: '120', scale: '0.00234', mse: '8.2e-7', bits: 8 },
  { name: 'fc2_weight', shape: '[84,120]', params: '10,080', scale: '0.00645', mse: '2.8e-5', bits: 8 },
  { name: 'fc3_weight', shape: '[10,84]', params: '840', scale: '0.00923', mse: '9.3e-6', bits: 8 },
]

const PER_CLASS = [99.1, 98.8, 98.3, 98.7, 98.5, 98.9, 99.0, 98.2, 98.1, 98.4]
const PER_CLASS_FP32 = [99.5, 99.3, 99.1, 99.2, 99.0, 99.4, 99.5, 98.9, 98.8, 99.1]

export default function Quantization({ active }: { active: boolean }) {
  const accRef = useRef<HTMLCanvasElement>(null)
  const classRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!active) return
    const draw = () => {
      if (accRef.current) drawAccuracyBars(accRef.current)
      if (classRef.current) drawPerClassChart(classRef.current)
    }
    const raf = requestAnimationFrame(draw)
    window.addEventListener('resize', draw)
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', draw) }
  }, [active])

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">INT8 Quantization</h1>
        <p className="page-subtitle">Symmetric post-training quantization: accuracy, compression, per-layer MSE</p>
      </div>

      <div className="grid-3">
        {[
          { label: 'FP32 Accuracy', value: '99.2%', color: 'var(--text-2)', desc: 'Baseline float32' },
          { label: 'INT8 Accuracy', value: '98.5%', color: 'var(--green)', desc: 'Post-training quant' },
          { label: 'Accuracy Drop', value: '-0.7%', color: 'var(--amber)', desc: 'Under 1% loss' },
        ].map(m => (
          <div className="metric-card" key={m.label}>
            <span className="metric-label">{m.label}</span>
            <span style={{ fontSize: 28, fontWeight: 700, letterSpacing: '-0.04em', color: m.color }}>{m.value}</span>
            <span style={{ fontSize: 11, color: 'var(--text-3)' }}>{m.desc}</span>
          </div>
        ))}
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-title">FP32 vs INT8 Accuracy</div>
          <canvas ref={accRef} className="chart-canvas" />
        </div>
        <div className="card">
          <div className="card-title">Per-Class Accuracy (INT8 vs FP32)</div>
          <canvas ref={classRef} className="chart-canvas" />
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-title">Quantization Scheme</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[
              ['Scheme', 'Symmetric INT8 (zero_point = 0)'],
              ['Granularity', 'Per-tensor scale factors'],
              ['Scale type', 'FP32 stored'],
              ['Compression', '4x (32-bit to 8-bit)'],
              ['Total params', '61,612'],
              ['FP32 size', '240.7 KB'],
              ['INT8 size', '60.2 KB'],
              ['Calibration', 'Post-training, full MNIST test set'],
            ].map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', gap: 8, fontSize: 12 }}>
                <span style={{ color: 'var(--text-3)' }}>{k}</span>
                <span style={{ color: 'var(--text-1)', fontWeight: 500, textAlign: 'right' }}>{v}</span>
              </div>
            ))}
          </div>
          <div className="divider" />
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <span className="badge badge-green">4x Compression</span>
            <span className="badge badge-accent">INT8 Symmetric</span>
            <span className="badge badge-amber">Under 1% Accuracy Loss</span>
          </div>
        </div>

        <div className="card">
          <div className="card-title">Recommendations</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {[
              {
                title: 'Per-channel quantization',
                desc: 'Apply separate scale per conv filter. Expected +0.2% accuracy recovery at minimal hardware cost.',
                badge: 'High impact', badgeClass: 'badge-green',
              },
              {
                title: 'Quantization-Aware Training (QAT)',
                desc: 'Simulate quantization during training. Typically recovers 80-90% of the accuracy gap.',
                badge: 'Recommended', badgeClass: 'badge-accent',
              },
              {
                title: 'Mixed precision for FC3',
                desc: 'Keep the output layer in FP16 for better class discrimination. Only 840 params, minimal cost.',
                badge: 'Quick win', badgeClass: 'badge-amber',
              },
            ].map(item => (
              <div key={item.title}>
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)', marginBottom: 4 }}>
                  {item.title}{' '}
                  <span className={`badge ${item.badgeClass}`} style={{ marginLeft: 4 }}>{item.badge}</span>
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.6 }}>{item.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">Per-Layer Quantization Details</div>
        <div style={{ overflowX: 'auto' }}>
          <table className="quant-table">
            <thead>
              <tr>
                {['Tensor', 'Shape', 'Params', 'Scale Factor', 'MSE Error', 'Bits'].map(h => (
                  <th key={h}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {QUANT_LAYERS.map(l => (
                <tr key={l.name}>
                  <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>{l.name}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{l.shape}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{l.params}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>{l.scale}</td>
                  <td className="quant-mse">{l.mse}</td>
                  <td><span className="badge badge-accent">{l.bits}b</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  )
}

function drawAccuracyBars(canvas: HTMLCanvasElement) {
  const dpr = window.devicePixelRatio || 1
  const rect = canvas.parentElement!.getBoundingClientRect()
  const width = Math.max(rect.width - 40, 100)
  const height = 160
  canvas.width = width * dpr
  canvas.height = height * dpr
  canvas.style.width = width + 'px'
  canvas.style.height = height + 'px'
  const ctx = canvas.getContext('2d')!
  ctx.scale(dpr, dpr)
  ctx.clearRect(0, 0, width, height)

  const bars = [
    { label: 'FP32', value: 99.2, color: '#6c8cff' },
    { label: 'INT8', value: 98.5, color: '#34d399' },
  ]
  const minVal = 97
  const maxVal = 100
  const range = maxVal - minVal
  const pad = { left: 52, right: 20, top: 18, bottom: 30 }
  const cw = width - pad.left - pad.right
  const ch = height - pad.top - pad.bottom
  const groupW = cw / bars.length
  const barW = groupW * 0.52

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.04)'
  ctx.lineWidth = 1
  for (let i = 0; i <= 3; i++) {
    const y = pad.top + (ch / 3) * i
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cw, y); ctx.stroke()
    ctx.fillStyle = 'rgba(255,255,255,0.28)'
    ctx.font = '10px JetBrains Mono, monospace'
    ctx.textAlign = 'right'
    const val = maxVal - (range / 3) * i
    ctx.fillText(val.toFixed(1) + '%', pad.left - 6, y + 4)
  }

  bars.forEach((bar, i) => {
    const barH = ((bar.value - minVal) / range) * ch
    const bx = pad.left + i * groupW + (groupW - barW) / 2
    const by = pad.top + ch - barH
    const grad = ctx.createLinearGradient(0, by, 0, by + barH)
    grad.addColorStop(0, bar.color)
    grad.addColorStop(1, bar.color + '55')
    ctx.fillStyle = grad
    ctx.beginPath()
    ctx.roundRect(bx, by, barW, barH, [4, 4, 0, 0])
    ctx.fill()

    ctx.fillStyle = 'rgba(255,255,255,0.8)'
    ctx.font = '11px JetBrains Mono, monospace'
    ctx.textAlign = 'center'
    ctx.fillText(bar.value.toFixed(1) + '%', bx + barW / 2, by - 6)

    ctx.fillStyle = 'rgba(255,255,255,0.38)'
    ctx.font = '11px Inter, sans-serif'
    ctx.fillText(bar.label, bx + barW / 2, pad.top + ch + 18)
  })
}

function drawPerClassChart(canvas: HTMLCanvasElement) {
  const dpr = window.devicePixelRatio || 1
  const rect = canvas.parentElement!.getBoundingClientRect()
  const width = Math.max(rect.width - 40, 100)
  const height = 160
  canvas.width = width * dpr
  canvas.height = height * dpr
  canvas.style.width = width + 'px'
  canvas.style.height = height + 'px'
  const ctx = canvas.getContext('2d')!
  ctx.scale(dpr, dpr)
  ctx.clearRect(0, 0, width, height)

  const pad = { left: 48, right: 12, top: 16, bottom: 28 }
  const cw = width - pad.left - pad.right
  const ch = height - pad.top - pad.bottom
  const minVal = 97.5
  const maxVal = 100
  const range = maxVal - minVal
  const stepX = cw / (PER_CLASS.length - 1)
  const toY = (v: number) => pad.top + ch - ((v - minVal) / range) * ch

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.04)'
  ctx.lineWidth = 1
  for (let i = 0; i <= 3; i++) {
    const y = pad.top + (ch / 3) * i
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cw, y); ctx.stroke()
    ctx.fillStyle = 'rgba(255,255,255,0.28)'
    ctx.font = '9px JetBrains Mono, monospace'
    ctx.textAlign = 'right'
    const val = maxVal - (range / 3) * i
    ctx.fillText(val.toFixed(1), pad.left - 4, y + 3)
  }

  // FP32 line (dim reference)
  ctx.beginPath()
  PER_CLASS_FP32.forEach((v, i) => {
    const x = pad.left + i * stepX
    const y = toY(v)
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
  })
  ctx.strokeStyle = 'rgba(108,140,255,0.3)'
  ctx.lineWidth = 1.5
  ctx.setLineDash([3, 3])
  ctx.stroke()
  ctx.setLineDash([])

  // INT8 fill
  const pts = PER_CLASS.map((v, i) => ({ x: pad.left + i * stepX, y: toY(v) }))
  const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ch)
  grad.addColorStop(0, '#34d39930')
  grad.addColorStop(1, '#34d39908')
  ctx.beginPath()
  ctx.moveTo(pts[0].x, pad.top + ch)
  pts.forEach(p => ctx.lineTo(p.x, p.y))
  ctx.lineTo(pts[pts.length - 1].x, pad.top + ch)
  ctx.closePath()
  ctx.fillStyle = grad
  ctx.fill()

  // INT8 line
  ctx.beginPath()
  pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y))
  ctx.strokeStyle = '#34d399'
  ctx.lineWidth = 2
  ctx.stroke()

  // Dots + digit labels
  pts.forEach((p, i) => {
    ctx.fillStyle = '#34d399'
    ctx.beginPath()
    ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = 'rgba(255,255,255,0.32)'
    ctx.font = '9px Inter, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(String(i), p.x, pad.top + ch + 17)
  })

  // Legend
  ctx.fillStyle = 'rgba(255,255,255,0.32)'
  ctx.font = '9px Inter, sans-serif'
  ctx.textAlign = 'left'
  ctx.fillText('Digit class', pad.left, pad.top + ch + 27)
}
