'use client'
import { useEffect, useRef, useState } from 'react'

const RESOURCES = [
  { name: 'LUTs', used: 3370, total: 20800, color: '#6c8cff' },
  { name: 'Flip-Flops', used: 2753, total: 41600, color: '#34d399' },
  { name: 'DSP48E1', used: 16, total: 90, color: '#a78bfa' },
  { name: 'Block RAM', used: 4, total: 50, color: '#fbbf24' },
]

type MacCell = { active: boolean; weight: number; activation: number }

export default function Architecture() {
  const [macCells, setMacCells] = useState<MacCell[][]>(
    Array.from({ length: 4 }, () => Array.from({ length: 4 }, () => ({ active: false, weight: 0, activation: 0 })))
  )
  const resRefs = useRef<(HTMLDivElement | null)[]>([])
  const tickRef = useRef(0)

  useEffect(() => {
    const id = setInterval(() => {
      const tick = tickRef.current
      tickRef.current = (tick + 1) % 8
      setMacCells(prev => prev.map((row, r) =>
        row.map((_, c) => {
          const active = (r + c) === tick
          return {
            active,
            weight: active ? (Math.random() * 127 | 0) : 0,
            activation: active ? ((Math.random() * 127 | 0) - 64) : 0,
          }
        })
      ))
    }, 350)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    const timeout = setTimeout(() => {
      resRefs.current.forEach((el, i) => {
        if (el) el.style.width = (RESOURCES[i].used / RESOURCES[i].total * 100).toFixed(1) + '%'
      })
    }, 200)
    return () => clearTimeout(timeout)
  }, [])

  return (
    <>
      <div className="page-header">
        <h1 className="page-title">Hardware Architecture</h1>
        <p className="page-subtitle">4x4 weight-stationary systolic array, RTL modules, resource utilization</p>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="section-header">
            <span className="section-title">Systolic Array: Live Simulation</span>
            <span className="badge badge-green">4x4, 16 MACs</span>
          </div>
          <div className="systolic-grid">
            {macCells.flat().map((cell, i) => (
              <div key={i} className={`mac-cell${cell.active ? ' active' : ''}`}>
                <span className="mac-id">[{Math.floor(i / 4)},{i % 4}]</span>
                <span className="mac-val">
                  {cell.active ? `${cell.weight}*${cell.activation}` : '--'}
                </span>
              </div>
            ))}
          </div>
          <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 10, lineHeight: 1.5 }}>
            Diagonal wavefront propagation, weight-stationary dataflow modeled after Google TPU v1.
            Each cell computes INT8 MAC: W*A accumulated to 32-bit.
          </p>
        </div>

        <div className="card">
          <div className="card-title">FPGA Resource Utilization: XC7A35T</div>
          <div>
            {RESOURCES.map((res, i) => {
              const pct = (res.used / res.total * 100).toFixed(1)
              return (
                <div className="resource-item" key={res.name}>
                  <div className="resource-top">
                    <span className="resource-label">{res.name}</span>
                    <span className="resource-pct">{pct}%</span>
                  </div>
                  <div className="resource-track">
                    <div
                      className="resource-fill"
                      ref={el => { resRefs.current[i] = el }}
                      style={{ background: res.color, width: 0 }}
                    />
                  </div>
                  <div className="resource-info">{res.used.toLocaleString()} / {res.total.toLocaleString()}</div>
                </div>
              )
            })}
          </div>
          <div className="divider" />
          <p style={{ fontSize: 11, color: 'var(--text-2)', lineHeight: 1.6 }}>
            74% DSP headroom remaining. Can scale to 8x8 (64 MACs) on the same device.
            LUT and FF usage is minimal, leaving room for future features.
          </p>
        </div>
      </div>

      <div className="card">
        <div className="card-title">RTL Modules</div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr>
                {['Module', 'File', 'Description', 'Status'].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: '7px 12px', borderBottom: '1px solid var(--border-default)', color: 'var(--text-3)', fontWeight: 600, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ['mac_unit', 'mac_unit.v', 'INT8 x INT8 to INT32 multiply-accumulate', 'Verified'],
                ['systolic_array', 'systolic_array.v', '4x4 weight-stationary array, 16 MACs', 'Verified'],
                ['conv_engine', 'conv_engine.v', '3x3 spatially-unrolled convolution, 4-stage pipeline', 'Verified'],
                ['activation', 'activation.v', 'ReLU / LeakyReLU with mode select', 'Verified'],
                ['pooling', 'pooling.v', '2x2 max pool, 2-stage pipeline', 'Verified'],
                ['layer_controller', 'layer_controller.v', '19-state FSM: Conv1, ReLU, Pool, FC, ArgMax', 'Verified'],
                ['uart_interface', 'uart_interface.v', '8N1 UART transceiver, 115200 baud @ 100 MHz', 'Verified'],
                ['axi_stream_wrapper', 'axi_stream_wrapper.v', 'AXI4-Stream alternative interface', 'Beta'],
                ['perf_counters', 'perf_counters.v', '32-bit cycle-accurate profiling, 6 counters', 'Verified'],
                ['input_buffer', 'input_buffer.v', 'Double-buffered ping-pong dual-port RAM', 'Verified'],
                ['weight_buffer', 'weight_buffer.v', 'BRAM INT8 weight storage, dual-port', 'Verified'],
                ['register_file', 'register_file.v', 'Memory-mapped control and status registers', 'Verified'],
                ['fifo', 'fifo.v', 'Configurable synchronous FIFO, depth 64', 'Verified'],
                ['top', 'top.v', 'Top-level CNN inference controller', 'Verified'],
              ].map(([mod, file, desc, status]) => (
                <tr key={mod}>
                  <td style={{ padding: '7px 12px', borderBottom: '1px solid var(--border-subtle)', fontFamily: 'var(--font-mono)', color: 'var(--accent)', fontSize: 11, fontWeight: 500 }}>{mod}</td>
                  <td style={{ padding: '7px 12px', borderBottom: '1px solid var(--border-subtle)', fontFamily: 'var(--font-mono)', color: 'var(--text-3)', fontSize: 11, whiteSpace: 'nowrap' }}>{file}</td>
                  <td style={{ padding: '7px 12px', borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-2)', fontSize: 11 }}>{desc}</td>
                  <td style={{ padding: '7px 12px', borderBottom: '1px solid var(--border-subtle)' }}>
                    <span className={`badge ${status === 'Verified' ? 'badge-green' : 'badge-amber'}`}>{status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  )
}
