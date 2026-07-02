'use client'
import { useEffect, useRef } from 'react'

const METRICS = [
  { label: 'Inference Latency', value: '0.41', suffix: 'ms', cls: 'green', delta: '5.1x vs CPU', deltaColor: 'green' },
  { label: 'Throughput', value: '2,439', suffix: 'img/s', cls: 'accent', delta: '5.1x vs CPU', deltaColor: 'green' },
  { label: 'Energy Efficiency', value: '668', suffix: 'x gain', cls: 'purple', delta: 'vs i7 baseline', deltaColor: 'amber' },
  { label: 'INT8 Accuracy', value: '98.5', suffix: '%', cls: 'amber', delta: '-0.7% vs FP32', deltaColor: 'amber' },
]

const PIPELINE = [
  ['UART Interface', '115200 baud, 8N1'],
  ['Input Buffer', 'Ping-pong, 784 B x 2'],
  ['Conv Engine (3x3)', '4-stage pipeline'],
  ['ReLU Activation', 'Sign-bit masking'],
  ['MaxPool (2x2)', 'Comparator tree'],
  ['Systolic Array (4x4)', '16 MACs / cycle'],
  ['Argmax + Output', 'Class 0-9'],
]

const SPARKLINE_DATA = {
  latency: [0.48, 0.43, 0.45, 0.41, 0.42, 0.39, 0.41, 0.40, 0.41, 0.43, 0.41],
  throughput: [2100, 2200, 2150, 2300, 2250, 2400, 2380, 2439, 2420, 2439, 2439],
}

export default function Overview() {
  const sparkRef1 = useRef<HTMLCanvasElement>(null)
  const sparkRef2 = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const draw = () => {
      if (sparkRef1.current) drawSparkline(sparkRef1.current, SPARKLINE_DATA.latency, '#34d399')
      if (sparkRef2.current) drawSparkline(sparkRef2.current, SPARKLINE_DATA.throughput, '#6c8cff')
    }
    draw()
    window.addEventListener('resize', draw)
    return () => window.removeEventListener('resize', draw)
  }, [])

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">System Overview</h1>
        <p className="page-subtitle">NeuralForge FPGA: Xilinx XC7A35T Artix-7, 100 MHz, INT8 LeNet-5</p>
      </div>

      <div className="grid-4">
        {METRICS.map(m => (
          <div className="metric-card" key={m.label}>
            <span className="metric-label">{m.label}</span>
            <div>
              <span className={`metric-value ${m.cls}`}>{m.value}</span>
              <span className="metric-suffix">{m.suffix}</span>
            </div>
            <span className={`metric-delta ${m.deltaColor}`}>{m.delta}</span>
          </div>
        ))}
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="section-header">
            <span className="section-title">Inference Latency Trend</span>
            <span className="section-meta">100 MHz</span>
          </div>
          <div style={{ marginBottom: 4 }}>
            <span style={{ fontSize: 22, fontWeight: 700, letterSpacing: '-0.04em', color: 'var(--green)' }}>0.41 ms</span>
            <span style={{ fontSize: 12, color: 'var(--text-3)', marginLeft: 6 }}>mean</span>
          </div>
          <div className="sparkline-wrap">
            <canvas ref={sparkRef1} style={{ width: '100%', height: 32 }} />
          </div>
        </div>

        <div className="card">
          <div className="section-header">
            <span className="section-title">Throughput</span>
            <span className="section-meta">img/s</span>
          </div>
          <div style={{ marginBottom: 4 }}>
            <span style={{ fontSize: 22, fontWeight: 700, letterSpacing: '-0.04em', color: 'var(--accent)' }}>2,439</span>
            <span style={{ fontSize: 12, color: 'var(--text-3)', marginLeft: 6 }}>img/s</span>
          </div>
          <div className="sparkline-wrap">
            <canvas ref={sparkRef2} style={{ width: '100%', height: 32 }} />
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-title">Inference Pipeline</div>
          <div className="pipe-list">
            {PIPELINE.map(([name, meta], i) => (
              <div className="pipe-row" key={name}>
                <span className="pipe-num">{i + 1}</span>
                <span className="pipe-name">{name}</span>
                <span className="pipe-meta">{meta}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-title">System Specs</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[
              ['FPGA Target', 'Xilinx XC7A35T Artix-7'],
              ['Clock Frequency', '100 MHz'],
              ['Systolic Array', '4x4 weight-stationary'],
              ['Peak Throughput', '1.6 GMAC/s (16 MACs/cycle)'],
              ['Network', 'LeNet-5 (MNIST, 10 classes)'],
              ['Quantization', 'INT8 symmetric, per-tensor'],
              ['Interface', 'UART 115200 baud / AXI4-Stream'],
              ['DSP48E1 Used', '16 of 90 (17.8%)'],
              ['Block RAM', '4 of 50 (8%)'],
            ].map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8 }}>
                <span style={{ fontSize: 12, color: 'var(--text-3)', whiteSpace: 'nowrap' }}>{k}</span>
                <span style={{ fontSize: 12, color: 'var(--text-1)', fontWeight: 500, textAlign: 'right' }}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  )
}

function drawSparkline(canvas: HTMLCanvasElement, data: number[], color: string) {
  const dpr = window.devicePixelRatio || 1
  const parent = canvas.parentElement
  if (!parent) return
  const width = parent.offsetWidth
  const height = 32
  canvas.width = width * dpr
  canvas.height = height * dpr
  canvas.style.width = width + 'px'
  canvas.style.height = height + 'px'
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.scale(dpr, dpr)
  ctx.clearRect(0, 0, width, height)

  const max = Math.max(...data) * 1.05
  const min = Math.min(...data) * 0.95
  const range = max - min || 1
  const stepX = width / (data.length - 1)
  const pts = data.map((v, i) => ({
    x: i * stepX,
    y: height - ((v - min) / range) * height * 0.75 - height * 0.12,
  }))

  const grad = ctx.createLinearGradient(0, 0, 0, height)
  grad.addColorStop(0, color + '28')
  grad.addColorStop(1, color + '00')

  ctx.beginPath()
  ctx.moveTo(0, height)
  pts.forEach(p => ctx.lineTo(p.x, p.y))
  ctx.lineTo(width, height)
  ctx.closePath()
  ctx.fillStyle = grad
  ctx.fill()

  ctx.beginPath()
  pts.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y))
  ctx.strokeStyle = color
  ctx.lineWidth = 1.5
  ctx.stroke()
}
