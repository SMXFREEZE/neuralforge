'use client'
import { useRef, useState, useCallback } from 'react'

declare global {
  interface Window { ort: any }
}

interface PredResult {
  digit: number
  confidence: number[]
  cycles: number
  latency_us: number
}

const ONNX_MODEL_URL =
  'https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/mnist/model/mnist-12.onnx'

let _session: any = null
let _sessionPromise: Promise<any> | null = null

async function getSession() {
  if (_session) return _session
  if (_sessionPromise) return _sessionPromise
  _sessionPromise = (async () => {
    let attempts = 0
    while (!window.ort && attempts < 80) {
      await new Promise(r => setTimeout(r, 100))
      attempts++
    }
    if (!window.ort) throw new Error('ONNX Runtime failed to load')
    // Point WASM files to the same CDN path
    window.ort.env.wasm.wasmPaths =
      'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/'
    const session = await window.ort.InferenceSession.create(ONNX_MODEL_URL)
    _session = session
    return session
  })()
  return _sessionPromise
}

function softmax(arr: number[]): number[] {
  const max = Math.max(...arr)
  const exp = arr.map(v => Math.exp(v - max))
  const sum = exp.reduce((a, b) => a + b, 0)
  return exp.map(v => Math.round((v / sum) * 1000) / 1000)
}

// Deterministic cycle count reported by the cycle-accurate simulator model
// (sw/fpga_simulator.py — the per-layer cycle sum for one LeNet-5 inference
// is input-independent: 47,732 cycles => 477.3 µs at 100 MHz).
const SIMULATOR_CYCLES = 47732
const SIMULATOR_CLOCK_MHZ = 100

function simulatorMetrics() {
  return {
    cycles: SIMULATOR_CYCLES,
    latency_us: Math.round((SIMULATOR_CYCLES / SIMULATOR_CLOCK_MHZ) * 10) / 10,
  }
}

export default function TryIt() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [drawing, setDrawing] = useState(false)
  const [hasContent, setHasContent] = useState(false)
  const [result, setResult] = useState<PredResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const initCanvas = useCallback((node: HTMLCanvasElement | null) => {
    if (!node) return
    ;(canvasRef as any).current = node
    const ctx = node.getContext('2d')
    if (!ctx) return
    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, 280, 280)
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = 16
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
  }, [])

  const getPos = (e: React.MouseEvent | React.TouchEvent, canvas: HTMLCanvasElement) => {
    const rect = canvas.getBoundingClientRect()
    const scaleX = canvas.width / rect.width
    const scaleY = canvas.height / rect.height
    if ('touches' in e) {
      const t = e.touches[0]
      return { x: (t.clientX - rect.left) * scaleX, y: (t.clientY - rect.top) * scaleY }
    }
    return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY }
  }

  const startDraw = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault()
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    setDrawing(true)
    setHasContent(true)
    const pos = getPos(e, canvas)
    ctx.beginPath()
    ctx.moveTo(pos.x, pos.y)
  }, [])

  const draw = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault()
    if (!drawing) return
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const pos = getPos(e, canvas)
    ctx.lineTo(pos.x, pos.y)
    ctx.stroke()
  }, [drawing])

  const stopDraw = useCallback(() => setDrawing(false), [])

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, 280, 280)
    setHasContent(false)
    setResult(null)
    setError(null)
  }, [])

  const classify = useCallback(async () => {
    const canvas = canvasRef.current
    if (!canvas || !hasContent) return
    setLoading(true)
    setError(null)
    try {
      const session = await getSession()
      const ort = window.ort

      // Crop digit to bounding box, center on 280x280, scale to 28x28
      const cropped = cropToCanvas(canvas)
      const small = document.createElement('canvas')
      small.width = 28
      small.height = 28
      small.getContext('2d')!.drawImage(cropped, 0, 0, 28, 28)
      const px = small.getContext('2d')!.getImageData(0, 0, 28, 28).data

      // Build NCHW float32 tensor [1, 1, 28, 28], values 0–1
      const float32 = new Float32Array(28 * 28)
      for (let i = 0; i < 28 * 28; i++) {
        float32[i] = px[i * 4] / 255.0
      }

      const inputName = session.inputNames[0]
      const outputName = session.outputNames[0]
      const tensor = new ort.Tensor('float32', float32, [1, 1, 28, 28])
      const results = await session.run({ [inputName]: tensor })
      const logits = Array.from(results[outputName].data as Float32Array) as number[]
      const confidence = softmax(logits)
      const digit = confidence.indexOf(Math.max(...confidence))

      setResult({ digit, confidence, ...simulatorMetrics() })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Classification failed')
    } finally {
      setLoading(false)
    }
  }, [hasContent])

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Try It: Live Inference</h1>
        <p className="page-subtitle">Draw a digit (0-9) and classify it with a real MNIST neural network and simulated FPGA metrics</p>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-title">Draw a Digit</div>
          <div className="draw-canvas-wrap" style={{ marginBottom: 12 }}>
            <canvas
              ref={initCanvas}
              className="draw-canvas"
              width={280}
              height={280}
              style={{ width: '100%', maxWidth: 280, aspectRatio: '1' }}
              onMouseDown={startDraw}
              onMouseMove={draw}
              onMouseUp={stopDraw}
              onMouseLeave={stopDraw}
              onTouchStart={startDraw}
              onTouchMove={draw}
              onTouchEnd={stopDraw}
            />
            {!hasContent && <div className="canvas-hint">Draw here</div>}
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-primary btn-full" onClick={classify} disabled={!hasContent || loading}>
              {loading ? (
                <><span className="loading-dots"><span/><span/><span/></span> Classifying</>
              ) : 'Classify'}
            </button>
            <button className="btn btn-secondary" onClick={clearCanvas} style={{ flexShrink: 0 }}>Clear</button>
          </div>
          {error && <p style={{ color: 'var(--red)', fontSize: 12, marginTop: 8 }}>{error}</p>}
        </div>

        <div className="card">
          <div className="card-title">Prediction</div>
          <div style={{ textAlign: 'center', marginBottom: 20, padding: '16px 0' }}>
            <div className={`pred-digit${result ? ' flash' : ''}`} style={{ color: result ? 'var(--green)' : 'var(--text-3)' }}>
              {result ? result.digit : '?'}
            </div>
            {result && (
              <div className="cycle-info" style={{ justifyContent: 'center', marginTop: 8 }}>
                <span className="cycle-tag">{result.cycles.toLocaleString()} cycles</span>
                <span>{result.latency_us} µs @ 100 MHz</span>
              </div>
            )}
          </div>

          {result ? (
            <>
              <div className="section-header">
                <span className="section-title">Confidence per Class</span>
              </div>
              <div>
                {result.confidence.map((conf, digit) => (
                  <div className="conf-row" key={digit}>
                    <span className="conf-digit">{digit}</span>
                    <div className="conf-track">
                      <div
                        className={`conf-fill${digit === result.digit ? ' winner' : ''}`}
                        style={{ width: `${(conf * 100).toFixed(1)}%` }}
                      />
                    </div>
                    <span className="conf-pct">{(conf * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="empty-state">Draw a digit and click Classify to see predictions</p>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-title">How It Works</div>
        <div className="how-it-works">
          <div className="hiw-row">
            <span className="hiw-num">1</span>
            <div>
              <div className="hiw-title">Draw</div>
              <div className="hiw-desc">Sketch any digit 0-9 on the canvas above using mouse or touch input.</div>
            </div>
          </div>
          <div className="hiw-row">
            <span className="hiw-num">2</span>
            <div>
              <div className="hiw-title">MNIST Neural Network</div>
              <div className="hiw-desc">The drawing is cropped, scaled to 28x28, and classified by a real CNN from the ONNX Model Zoo — running directly in your browser via ONNX Runtime Web.</div>
            </div>
          </div>
          <div className="hiw-row">
            <span className="hiw-num">3</span>
            <div>
              <div className="hiw-title">FPGA Simulation</div>
              <div className="hiw-desc">Latency and cycle count come from the cycle-accurate simulator model (sw/fpga_simulator.py) at a modeled 100 MHz Artix-7 clock — not measurements from physical hardware.</div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}

function cropToCanvas(canvas: HTMLCanvasElement): HTMLCanvasElement {
  const ctx = canvas.getContext('2d')!
  const { width, height } = canvas
  const px = ctx.getImageData(0, 0, width, height).data

  let x0 = width, x1 = 0, y0 = height, y1 = 0
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4
      if ((px[i] + px[i + 1] + px[i + 2]) / 3 > 30) {
        if (x < x0) x0 = x
        if (x > x1) x1 = x
        if (y < y0) y0 = y
        if (y > y1) y1 = y
      }
    }
  }

  const out = document.createElement('canvas')
  out.width = 280
  out.height = 280
  const oc = out.getContext('2d')!
  oc.fillStyle = '#000000'
  oc.fillRect(0, 0, 280, 280)

  if (x1 <= x0 || y1 <= y0) return out

  const pad = 24
  x0 = Math.max(0, x0 - pad)
  y0 = Math.max(0, y0 - pad)
  x1 = Math.min(width, x1 + pad)
  y1 = Math.min(height, y1 + pad)

  const cw = x1 - x0
  const ch = y1 - y0
  const size = Math.max(cw, ch)
  const scale = 220 / size
  const dw = cw * scale
  const dh = ch * scale
  const dx = (280 - dw) / 2
  const dy = (280 - dh) / 2

  oc.drawImage(canvas, x0, y0, cw, ch, dx, dy, dw, dh)
  return out
}
