'use client'

const LAYERS = [
  { name: 'Input', dims: '1x28x28', params: '-', flops: '-', backend: 'IO', badgeClass: 'badge-cyan' },
  { name: 'Conv2D_1', dims: '6x24x24', params: '156', flops: '299K', backend: 'Vivado HLS', badgeClass: 'badge-accent' },
  { name: 'ReLU_1', dims: '6x24x24', params: '-', flops: '3.5K', backend: 'LUT', badgeClass: 'badge-green' },
  { name: 'MaxPool2D_1', dims: '6x12x12', params: '-', flops: '3.5K', backend: 'DSP', badgeClass: 'badge-purple' },
  { name: 'Conv2D_2', dims: '16x8x8', params: '2,416', flops: '615K', backend: 'Vivado HLS', badgeClass: 'badge-accent' },
  { name: 'ReLU_2', dims: '16x8x8', params: '-', flops: '1.0K', backend: 'LUT', badgeClass: 'badge-green' },
  { name: 'MaxPool2D_2', dims: '16x4x4', params: '-', flops: '512', backend: 'DSP', badgeClass: 'badge-purple' },
  { name: 'Flatten', dims: '256', params: '-', flops: '-', backend: 'Wire', badgeClass: 'badge-cyan' },
  { name: 'Dense_1', dims: '120', params: '30,840', flops: '61K', backend: 'Vivado HLS', badgeClass: 'badge-accent' },
  { name: 'ReLU_3', dims: '120', params: '-', flops: '120', backend: 'LUT', badgeClass: 'badge-green' },
  { name: 'Dense_2', dims: '84', params: '10,164', flops: '20K', backend: 'Vivado HLS', badgeClass: 'badge-accent' },
  { name: 'ReLU_4', dims: '84', params: '-', flops: '84', backend: 'LUT', badgeClass: 'badge-green' },
  { name: 'Dense_3 (out)', dims: '10', params: '850', flops: '1.7K', backend: 'Vivado HLS', badgeClass: 'badge-accent' },
  { name: 'Softmax', dims: '10', params: '-', flops: '50', backend: 'LUT', badgeClass: 'badge-green' },
]

const STRATEGIES = [
  { label: 'Reuse Factor', value: '4', desc: 'Balance DSP reuse vs latency. Factor 1 = fully unrolled.' },
  { label: 'Strategy', value: 'Resource', desc: 'Minimizes LUT/DSP usage for constrained FPGAs.' },
  { label: 'Precision', value: 'ap_fixed<8,4>', desc: '8-bit fixed-point: 4 integer bits + 4 fractional bits.' },
  { label: 'Clock Period', value: '10 ns', desc: '100 MHz target. Vivado HLS schedules at this constraint.' },
  { label: 'IOType', value: 'io_parallel', desc: 'All inputs consumed in one clock cycle. Best for small networks.' },
]

export default function Hls4ml() {
  return (
    <>
      <div className="page-header">
        <h1 className="page-title">hls4ml Integration</h1>
        <p className="page-subtitle">HLS-synthesized LeNet-5, layer-by-layer mapping to Vivado HLS and Xilinx IP blocks</p>
      </div>

      <div className="grid-3">
        {[
          { label: 'Total Parameters', value: '44,426', color: 'var(--accent)' },
          { label: 'Total FLOPs', value: '1.0M', color: 'var(--purple)' },
          { label: 'HLS Clock', value: '10 ns', color: 'var(--green)' },
        ].map(m => (
          <div className="metric-card" key={m.label}>
            <span className="metric-label">{m.label}</span>
            <span style={{ fontSize: 26, fontWeight: 700, letterSpacing: '-0.04em', color: m.color }}>{m.value}</span>
          </div>
        ))}
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-title">Layer Mapping: hls4ml Synthesis Plan</div>
        <div className="layer-grid">
          {LAYERS.map(layer => (
            <div className="layer-row" key={layer.name}>
              <span className="layer-name">{layer.name}</span>
              <span className="layer-dims">{layer.dims}</span>
              <span className="layer-params">{layer.params !== '-' ? `${layer.params} params` : ''}</span>
              <span className={`layer-backend badge ${layer.badgeClass}`}>{layer.backend}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-title">HLS Configuration</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {STRATEGIES.map(s => (
              <div key={s.label}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                  <span style={{ fontSize: 12, color: 'var(--text-3)', fontWeight: 500 }}>{s.label}</span>
                  <code style={{ fontSize: 12, color: 'var(--accent)', fontFamily: 'var(--font-mono)', background: 'var(--accent-dim)', padding: '1px 6px', borderRadius: 4 }}>{s.value}</code>
                </div>
                <p style={{ fontSize: 11, color: 'var(--text-3)', lineHeight: 1.5 }}>{s.desc}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-title">hls4ml vs Manual RTL</div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, marginBottom: 16 }}>
            <thead>
              <tr>
                {['Metric', 'hls4ml', 'Manual RTL'].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: '5px 8px', borderBottom: '1px solid var(--border-default)', color: 'var(--text-3)', fontWeight: 600, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                { label: 'Dev Time', hls: '< 1 hour', manual: '2-3 weeks', winner: 'hls' },
                { label: 'Latency', hls: '1.2 us', manual: '0.41 us', winner: 'manual' },
                { label: 'LUT Usage', hls: '~4800', manual: '3370', winner: 'manual' },
                { label: 'Portability', hls: 'Multi-FPGA', manual: 'XC7A35T only', winner: 'hls' },
                { label: 'Maintainability', hls: 'PyTorch model', manual: 'Hand-coded RTL', winner: 'hls' },
              ].map(row => (
                <tr key={row.label}>
                  <td style={{ padding: '6px 8px', borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-3)' }}>{row.label}</td>
                  <td style={{ padding: '6px 8px', borderBottom: '1px solid var(--border-subtle)', color: row.winner === 'hls' ? 'var(--green)' : 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>{row.hls}</td>
                  <td style={{ padding: '6px 8px', borderBottom: '1px solid var(--border-subtle)', color: row.winner === 'manual' ? 'var(--green)' : 'var(--text-2)', fontFamily: 'var(--font-mono)' }}>{row.manual}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ fontSize: 11, color: 'var(--text-2)', lineHeight: 1.6 }}>
            Manual RTL has lower latency (3x). hls4ml wins on iteration speed, making it better for rapid prototyping and design-space exploration.
          </p>
        </div>
      </div>
    </>
  )
}
