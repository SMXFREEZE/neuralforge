'use client'
import { useEffect, useRef } from 'react'

const LATENCY_DATA = {
  labels: ['CPU i7-8700', 'CPU PyTorch', 'GPU RTX 3080', 'FPGA NeuralForge'],
  datasets: [{ label: 'Latency (ms)', data: [2.13, 1.87, 0.38, 0.48], color: '#6c8cff' }],
}

const THROUGHPUT_DATA = {
  labels: ['CPU i7', 'CPU PyTorch', 'GPU RTX3080', 'FPGA'],
  datasets: [{ label: 'img/s', data: [470, 535, 2631, 2095], color: '#34d399' }],
}

const ENERGY_DATA = {
  labels: ['FPGA 0.5W', 'CPU 65W'],
  values: [4190, 7.3],
  colors: ['#34d399', '#f87171'],
}

export default function Performance({ active }: { active: boolean }) {
  const latencyRef = useRef<HTMLCanvasElement>(null)
  const throughputRef = useRef<HTMLCanvasElement>(null)
  const energyRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!active) return
    const draw = () => {
      if (latencyRef.current) drawBarChart(latencyRef.current, LATENCY_DATA, 220)
      if (throughputRef.current) drawBarChart(throughputRef.current, THROUGHPUT_DATA, 220)
      if (energyRef.current) drawHBar(energyRef.current, ENERGY_DATA, 120)
    }
    // rAF ensures layout is painted before measuring
    const raf = requestAnimationFrame(draw)
    window.addEventListener('resize', draw)
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', draw) }
  }, [active])

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Performance Benchmarks</h1>
        <p className="page-subtitle">NeuralForge vs CPU and GPU baselines — modeled/estimated figures (cycle-accurate simulator + datasheet power), not hardware measurements</p>
      </div>

      <div className="grid-4" style={{ marginBottom: 24 }}>
        {[
          { label: 'FPGA Latency', value: '0.48 ms', note: '100 MHz, INT8, modeled', color: 'var(--green)' },
          { label: 'CPU Latency', value: '2.13 ms', note: 'i7-8700, FP32', color: 'var(--red)' },
          { label: 'Speedup', value: '4.4x', note: 'vs CPU baseline, est.', color: 'var(--accent)' },
          { label: 'Energy Saving', value: '574x', note: '0.5 W vs 65 W, est.', color: 'var(--purple)' },
        ].map(m => (
          <div className="metric-card" key={m.label}>
            <span className="metric-label">{m.label}</span>
            <span style={{ fontSize: 26, fontWeight: 700, letterSpacing: '-0.04em', color: m.color }}>{m.value}</span>
            <span style={{ fontSize: 11, color: 'var(--text-3)' }}>{m.note}</span>
          </div>
        ))}
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-title">Inference Latency (ms), lower is better</div>
          <canvas ref={latencyRef} className="chart-canvas" />
        </div>
        <div className="card">
          <div className="card-title">Throughput (img/s), higher is better</div>
          <canvas ref={throughputRef} className="chart-canvas" />
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-title">Energy Efficiency (inferences / joule)</div>
          <canvas ref={energyRef} className="chart-canvas" />
          <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 10 }}>
            FPGA @ 0.5 W: 4,190 inf/J (modeled), CPU @ 65 W: 7.3 inf/J.{' '}
            <strong style={{ color: 'var(--green)' }}>574x improvement (estimated).</strong>
          </p>
        </div>
        <div className="card">
          <div className="card-title">Benchmark Summary</div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr>
                {['Platform', 'Latency', 'Throughput', 'Power', 'Inf/J'].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: '6px 8px', borderBottom: '1px solid var(--border-default)', color: 'var(--text-3)', fontWeight: 600, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ['CPU i7-8700', '2.13 ms', '470 img/s', '65 W', '7.3'],
                ['CPU PyTorch', '1.87 ms', '535 img/s', '65 W', '8.2'],
                ['GPU RTX 3080', '0.38 ms', '2,631 img/s', '320 W', '8.2'],
                ['FPGA (ours)', '0.48 ms', '2,095 img/s', '0.5 W', '4,190'],
              ].map((row, i) => (
                <tr key={i} style={{ background: i === 3 ? 'var(--green-dim)' : undefined }}>
                  {row.map((cell, j) => (
                    <td key={j} style={{ padding: '7px 8px', borderBottom: '1px solid var(--border-subtle)', color: i === 3 && j === 0 ? 'var(--green)' : 'var(--text-2)', fontWeight: i === 3 ? 600 : 400, fontFamily: j > 0 ? 'var(--font-mono)' : undefined, fontSize: 12 }}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  )
}

function drawBarChart(
  canvas: HTMLCanvasElement,
  data: { labels: string[]; datasets: { data: number[]; color: string; label: string }[] },
  height: number,
) {
  const dpr = window.devicePixelRatio || 1
  const rect = canvas.parentElement!.getBoundingClientRect()
  const width = Math.max(rect.width - 40, 100)
  canvas.width = width * dpr
  canvas.height = height * dpr
  canvas.style.width = width + 'px'
  canvas.style.height = height + 'px'
  const ctx = canvas.getContext('2d')!
  ctx.scale(dpr, dpr)
  ctx.clearRect(0, 0, width, height)

  const pad = { top: 20, bottom: 44, left: 52, right: 16 }
  const cw = width - pad.left - pad.right
  const ch = height - pad.top - pad.bottom
  const allValues = data.datasets.flatMap(d => d.data)
  const maxVal = Math.max(...allValues) * 1.25
  const { labels, datasets } = data
  const groupW = cw / labels.length
  const barW = groupW * 0.52

  ctx.strokeStyle = 'rgba(255,255,255,0.04)'
  ctx.lineWidth = 1
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (ch / 4) * i
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cw, y); ctx.stroke()
    ctx.fillStyle = 'rgba(255,255,255,0.28)'
    ctx.font = `10px JetBrains Mono, monospace`
    ctx.textAlign = 'right'
    const val = maxVal - (maxVal / 4) * i
    ctx.fillText(val >= 1000 ? (val / 1000).toFixed(1) + 'k' : val.toFixed(2), pad.left - 6, y + 4)
  }

  labels.forEach((label, li) => {
    const x = pad.left + li * groupW + (groupW - barW * datasets.length) / 2
    datasets.forEach((ds, di) => {
      const barH = (ds.data[li] / maxVal) * ch
      const bx = x + di * barW
      const by = pad.top + ch - barH
      const grad = ctx.createLinearGradient(0, by, 0, by + barH)
      grad.addColorStop(0, ds.color)
      grad.addColorStop(1, ds.color + '60')
      ctx.fillStyle = grad
      ctx.beginPath()
      ctx.roundRect(bx, by, barW - 3, barH, [4, 4, 0, 0])
      ctx.fill()
    })
    ctx.fillStyle = 'rgba(255,255,255,0.38)'
    ctx.font = '10px Inter, sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(label, pad.left + li * groupW + groupW / 2, pad.top + ch + 15)
  })
}

function drawHBar(
  canvas: HTMLCanvasElement,
  data: { labels: string[]; values: number[]; colors: string[] },
  height: number,
) {
  const dpr = window.devicePixelRatio || 1
  const rect = canvas.parentElement!.getBoundingClientRect()
  const width = Math.max(rect.width - 40, 100)
  canvas.width = width * dpr
  canvas.height = height * dpr
  canvas.style.width = width + 'px'
  canvas.style.height = height + 'px'
  const ctx = canvas.getContext('2d')!
  ctx.scale(dpr, dpr)
  ctx.clearRect(0, 0, width, height)

  const maxVal = Math.max(...data.values)
  const barH = 26
  const gap = 22
  const labelW = 80
  const pad = { top: 10, left: labelW + 12, right: 72 }

  data.labels.forEach((label, i) => {
    const y = pad.top + i * (barH + gap)
    const barW = (data.values[i] / maxVal) * (width - pad.left - pad.right)

    ctx.fillStyle = 'rgba(255,255,255,0.45)'
    ctx.font = '11px Inter, sans-serif'
    ctx.textAlign = 'right'
    ctx.fillText(label, labelW, y + barH / 2 + 4)

    const grad = ctx.createLinearGradient(pad.left, 0, pad.left + barW, 0)
    grad.addColorStop(0, data.colors[i])
    grad.addColorStop(1, data.colors[i] + '80')
    ctx.fillStyle = grad
    ctx.beginPath()
    ctx.roundRect(pad.left, y, barW, barH, 4)
    ctx.fill()

    ctx.fillStyle = 'rgba(255,255,255,0.65)'
    ctx.font = '10px JetBrains Mono, monospace'
    ctx.textAlign = 'left'
    const formatted = data.values[i] >= 1000 ? data.values[i].toLocaleString() : data.values[i].toString()
    ctx.fillText(formatted, pad.left + barW + 6, y + barH / 2 + 4)
  })
}
