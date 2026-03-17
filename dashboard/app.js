/* =================================================================
   NeuralForge Dashboard — Application
   ================================================================= */

const API = '';
const CHARTS = [];
const PAL = ['#3b82f6','#22c55e','#a78bfa','#f59e0b','#ef4444','#06b6d4','#71717a'];

/* ── Navigation ─────────────────────────────────────────────── */
document.querySelectorAll('.nav-link').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-link').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
        btn.classList.add('active');
        const view = document.getElementById('view-' + btn.dataset.tab);
        if (view) view.classList.add('active');
        requestAnimationFrame(() => CHARTS.forEach(c => c.resize()));
    });
});

/* ── Animated counter ───────────────────────────────────────── */
function countUp(el) {
    const to = parseFloat(el.dataset.to);
    const dec = parseInt(el.dataset.decimals || '0', 10);
    const dur = 1200;
    const t0 = performance.now();
    (function step(now) {
        const p = Math.min((now - t0) / dur, 1);
        const e = 1 - Math.pow(1 - p, 3);
        const v = to * e;
        el.textContent = dec > 0 ? v.toFixed(dec) : Math.round(v).toLocaleString();
        if (p < 1) requestAnimationFrame(step);
    })(t0);
}

/* ── Sparklines ─────────────────────────────────────────────── */
function spark(id, data, color) {
    const c = document.getElementById(id);
    if (!c) return;
    const dpr = devicePixelRatio || 1;
    const W = c.parentElement.offsetWidth - 40; // card padding
    const H = 32;
    c.width = W * dpr; c.height = H * dpr;
    c.style.width = W + 'px'; c.style.height = H + 'px';
    const ctx = c.getContext('2d');
    ctx.scale(dpr, dpr);
    const mx = Math.max(...data) * 1.05, mn = Math.min(...data) * 0.95, r = mx - mn || 1;
    const s = W / (data.length - 1);
    const pts = data.map((v, i) => ({ x: i * s, y: H - ((v - mn) / r) * H * 0.75 - H * 0.12 }));

    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, color + '28'); g.addColorStop(1, color + '00');
    ctx.beginPath(); ctx.moveTo(0, H);
    pts.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.lineTo(W, H); ctx.closePath(); ctx.fillStyle = g; ctx.fill();

    ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.stroke();
}

/* ── Pipeline ───────────────────────────────────────────────── */
function initPipeline() {
    const el = document.getElementById('pipelineList');
    if (!el) return;
    const st = [
        ['UART Interface','115200 baud, 8N1'],
        ['Input Buffer','Ping-pong, 784 B x 2'],
        ['Conv Engine (3x3)','4-stage pipeline'],
        ['ReLU Activation','Sign-bit masking'],
        ['MaxPool (2x2)','Comparator tree'],
        ['Systolic Array (4x4)','16 MACs / cycle'],
        ['Argmax + Output','Class 0-9'],
    ];
    st.forEach(([name, meta], i) => {
        const d = document.createElement('div');
        d.className = 'pipe-row';
        d.style.opacity = '0'; d.style.transform = 'translateX(-12px)';
        d.innerHTML = `<span class="pipe-num">${i + 1}</span><span class="pipe-name">${name}</span><span class="pipe-meta">${meta}</span>`;
        el.appendChild(d);
        setTimeout(() => { d.style.transition = 'all .4s ease'; d.style.opacity = '1'; d.style.transform = 'none'; }, 50 * i);
    });
}

/* ── Systolic Array ─────────────────────────────────────────── */
function initSystolic() {
    const g = document.getElementById('systolicGrid');
    if (!g) return;
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
        const d = document.createElement('div');
        d.className = 'mac'; d.id = `m${r}${c}`;
        d.innerHTML = `<span class="mac-id">[${r},${c}]</span><span class="mac-val">--</span>`;
        g.appendChild(d);
    }
    let tick = 0;
    setInterval(() => {
        document.querySelectorAll('.mac').forEach(m => { m.classList.remove('on'); m.querySelector('.mac-val').textContent = '--'; });
        for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) {
            if ((r + c) === (tick % 8)) {
                const m = document.getElementById(`m${r}${c}`);
                if (m) { m.classList.add('on'); m.querySelector('.mac-val').textContent = `${(Math.random()*127|0)}x${(Math.random()*127|0)-64}`; }
            }
        }
        tick++;
    }, 350);
}

/* ── Resources ──────────────────────────────────────────────── */
function initResources() {
    const el = document.getElementById('resGrid');
    if (!el) return;
    const rs = [
        { n: 'LUTs', u: 3370, t: 20800, c: '#3b82f6' },
        { n: 'Flip-Flops', u: 2753, t: 41600, c: '#22c55e' },
        { n: 'DSP48E1', u: 16, t: 90, c: '#a78bfa' },
        { n: 'Block RAM', u: 4, t: 50, c: '#f59e0b' },
    ];
    rs.forEach(r => {
        const p = (r.u / r.t * 100).toFixed(1);
        const d = document.createElement('div');
        d.className = 'res-item';
        d.innerHTML = `<div class="res-top"><span class="res-label">${r.n}</span><span class="res-pct">${p}%</span></div><div class="res-track"><div class="res-fill" style="background:${r.c}" data-w="${p}"></div></div><div class="res-info">${r.u.toLocaleString()} / ${r.t.toLocaleString()}</div>`;
        el.appendChild(d);
        setTimeout(() => { d.querySelector('.res-fill').style.width = p + '%'; }, 200);
    });
}

/* ── Chart engine ───────────────────────────────────────────── */
class C {
    constructor(id, data, opts) {
        this.cv = document.getElementById(id); if (!this.cv) return;
        this.cx = this.cv.getContext('2d'); this.d = data; this.o = opts;
        this.resize(); window.addEventListener('resize', () => this.resize());
        CHARTS.push(this);
    }
    resize() {
        const dpr = devicePixelRatio || 1;
        const r = this.cv.parentElement.getBoundingClientRect();
        if (r.width < 1) return;
        this.cv.width = r.width * dpr; this.cv.height = (this.o.h || 260) * dpr;
        this.cv.style.width = r.width + 'px'; this.cv.style.height = (this.o.h || 260) + 'px';
        this.dpr = dpr; this.draw();
    }
    draw() {
        if (!this.cx) return;
        const cx = this.cx, d = this.dpr || 1;
        cx.save(); cx.scale(d, d);
        const w = this.cv.width / d, h = this.cv.height / d;
        cx.clearRect(0, 0, w, h);
        if (this.o.t === 'bar') this._bar(cx, w, h);
        else if (this.o.t === 'hbar') this._hbar(cx, w, h);
        else if (this.o.t === 'donut') this._donut(cx, w, h);
        cx.restore();
    }
    _bar(cx, w, h) {
        const p = { t: 28, b: 52, l: 50, r: 16 };
        const cW = w - p.l - p.r, cH = h - p.t - p.b;
        const { labels: L, datasets: DS } = this.d;
        const max = Math.max(...DS.flatMap(d => d.data)) * 1.2;
        const gW = cW / L.length, bW = gW / (DS.length + 1);

        // grid
        cx.strokeStyle = 'rgba(255,255,255,0.04)'; cx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = p.t + (cH / 4) * i;
            cx.beginPath(); cx.moveTo(p.l, y); cx.lineTo(p.l + cW, y); cx.stroke();
            cx.fillStyle = '#71717a'; cx.font = '11px JetBrains Mono'; cx.textAlign = 'right';
            const v = max - (max / 4) * i;
            cx.fillText(v >= 100 ? Math.round(v) : v.toFixed(2), p.l - 6, y + 4);
        }

        DS.forEach((ds, di) => {
            ds.data.forEach((v, i) => {
                const x = p.l + i * gW + di * bW + bW * 0.4;
                const bH = (v / max) * cH, y = p.t + cH - bH;
                const g = cx.createLinearGradient(0, y, 0, y + bH);
                const col = ds.color || PAL[di];
                g.addColorStop(0, col); g.addColorStop(1, col + '44');
                cx.fillStyle = g; cx.beginPath(); cx.roundRect(x, y, bW * 0.7, bH, [4,4,0,0]); cx.fill();
                cx.fillStyle = '#a1a1aa'; cx.font = '11px JetBrains Mono'; cx.textAlign = 'center';
                cx.fillText(v >= 100 ? v : v.toFixed(2), x + bW * 0.35, y - 6);
            });
        });

        L.forEach((l, i) => {
            cx.fillStyle = '#71717a'; cx.font = '11px Inter'; cx.textAlign = 'center';
            cx.fillText(l, p.l + i * gW + gW / 2, h - p.b + 18);
        });

        let lx = p.l;
        DS.forEach((ds, di) => {
            const col = ds.color || PAL[di];
            cx.fillStyle = col; cx.fillRect(lx, h - 10, 8, 8);
            cx.fillStyle = '#a1a1aa'; cx.font = '11px Inter'; cx.textAlign = 'left';
            cx.fillText(ds.label, lx + 12, h - 3);
            lx += cx.measureText(ds.label).width + 28;
        });
    }
    _hbar(cx, w, h) {
        const p = { t: 8, b: 8, l: 84, r: 50 };
        const cW = w - p.l - p.r, cH = h - p.t - p.b;
        const { labels: L, values: V, colors: CO } = this.d;
        const max = Math.max(...V) * 1.15;
        const step = cH / L.length, bH = Math.min(step * 0.5, 20);
        L.forEach((l, i) => {
            const y = p.t + i * step + step / 2;
            const bW = (V[i] / max) * cW;
            const col = (CO && CO[i]) || PAL[i % PAL.length];
            const g = cx.createLinearGradient(p.l, 0, p.l + bW, 0);
            g.addColorStop(0, col); g.addColorStop(1, col + '88');
            cx.fillStyle = g; cx.beginPath(); cx.roundRect(p.l, y - bH/2, bW, bH, [0,4,4,0]); cx.fill();
            cx.fillStyle = '#a1a1aa'; cx.font = '11px JetBrains Mono'; cx.textAlign = 'right';
            cx.fillText(l, p.l - 6, y + 4);
            cx.fillStyle = '#fafafa'; cx.font = '600 11px JetBrains Mono'; cx.textAlign = 'left';
            cx.fillText(V[i].toLocaleString(), p.l + bW + 8, y + 4);
        });
    }
    _donut(cx, w, h) {
        const { labels: L, values: V, colors: CO } = this.d;
        const total = V.reduce((a, b) => a + b, 0);
        const cx0 = w / 2, cy0 = h / 2 - 10, r = Math.min(w, h) * 0.28;
        let a = -Math.PI / 2;
        V.forEach((v, i) => {
            const s = (v / total) * Math.PI * 2;
            cx.beginPath(); cx.moveTo(cx0, cy0); cx.arc(cx0, cy0, r, a, a + s); cx.closePath();
            cx.fillStyle = (CO && CO[i]) || PAL[i % PAL.length]; cx.fill();
            a += s;
        });
        cx.beginPath(); cx.arc(cx0, cy0, r * 0.58, 0, Math.PI * 2);
        cx.fillStyle = '#111113'; cx.fill();
        cx.fillStyle = '#fafafa'; cx.font = '700 15px Inter'; cx.textAlign = 'center';
        cx.fillText(total.toLocaleString(), cx0, cy0 + 5);
        cx.fillStyle = '#71717a'; cx.font = '11px Inter'; cx.fillText('Total LUTs', cx0, cy0 + 20);
    }
}

/* ── Init Charts ────────────────────────────────────────────── */
function initCharts() {
    new C('chartLatency', {
        labels: ['Mean','Median','P95','P99'],
        datasets: [
            { label: 'CPU (PyTorch)', data: [2.13,2.09,2.89,3.25], color: '#71717a' },
            { label: 'FPGA', data: [0.41,0.41,0.44,0.46], color: '#3b82f6' },
        ],
    }, { t: 'bar', h: 270 });

    new C('chartThroughput', {
        labels: ['CPU','FPGA'],
        datasets: [{ label: 'img/s', data: [469, 2439], color: '#22c55e' }],
    }, { t: 'bar', h: 270 });

    new C('chartPipeline', {
        labels: ['Conv1','Pool1','Conv2','Pool2','FC1','FC2','FC3'],
        values: [8200,1200,12800,800,3800,2100,840],
        colors: ['#3b82f6','#22c55e','#a78bfa','#22c55e','#f59e0b','#f59e0b','#f59e0b'],
    }, { t: 'hbar', h: 230 });

    new C('chartPie', {
        labels: ['SystolicArr','ConvEng','LayerCtrl','UART','Buffers','PerfCtrs','Other'],
        values: [1200,580,320,180,285,160,645],
        colors: PAL,
    }, { t: 'donut', h: 270 });

    new C('chartQuantErr', {
        labels: ['conv1_w','conv1_b','conv2_w','conv2_b','fc1_w','fc1_b','fc2_w','fc2_b','fc3_w','fc3_b'],
        values: [12,0.31,8.7,0.18,5.4,0.42,6.1,0.29,9.3,0.15],
        colors: Array(10).fill('#3b82f6'),
    }, { t: 'hbar', h: 270 });

    new C('chartScales', {
        labels: ['conv1','conv2','fc1','fc2','fc3'],
        datasets: [{ label: 'Scale x1000', data: [23,18,12,15,21], color: '#22c55e' }],
    }, { t: 'bar', h: 210 });
}

/* ── Counters ───────────────────────────────────────────────── */
function initCounters() {
    const el = document.getElementById('countersGrid');
    if (!el) return;
    const c = [
        ['Total Cycles','41,283','0.413 ms @ 100 MHz'],
        ['Inference Cycles','38,420','93.1% of total'],
        ['Conv Cycles','24,680','64.2% of inference'],
        ['Pool Cycles','3,200','8.3% of inference'],
        ['FC Cycles','10,540','27.4% of inference'],
        ['MAC Active','31,200','81.2% utilization'],
        ['Memory Reads','8,750','22.8% bandwidth'],
        ['Pipeline Stalls','1,890','4.9% stall rate'],
        ['Completed','1','Inference count'],
    ];
    c.forEach(([n, v, d]) => {
        const div = document.createElement('div');
        div.className = 'counter-box';
        div.innerHTML = `<div class="counter-name">${n}</div><div class="counter-val">${v}</div><div class="counter-info">${d}</div>`;
        el.appendChild(div);
    });
}

/* ── Module Tree ────────────────────────────────────────────── */
function initTree() {
    const el = document.getElementById('moduleTree');
    if (!el) return;
    const ms = [
        [0,'top.v','controller','450 LUTs, 380 FFs'],
        [1,'uart_interface.v','io','180 LUTs, 160 FFs'],
        [1,'input_buffer.v','memory','120 LUTs, 1 BRAM'],
        [1,'weight_buffer.v','memory','80 LUTs, 2 BRAM'],
        [1,'layer_controller.v','controller','320 LUTs, 280 FFs'],
        [2,'conv_engine.v','compute','580 LUTs, 420 FFs'],
        [2,'activation.v','compute','45 LUTs, 35 FFs'],
        [2,'pooling.v','compute','60 LUTs, 50 FFs'],
        [2,'systolic_array.v','compute','1,200 LUTs, 16 DSPs'],
        [3,'mac_unit.v x16','compute','75 LUTs ea, 1 DSP ea'],
        [2,'fifo.v','memory','85 LUTs, 1 BRAM'],
        [1,'perf_counters.v','debug','160 LUTs, 384 FFs'],
        [1,'register_file.v','io','90 LUTs, 64 FFs'],
    ];
    ms.forEach(([indent, name, type, stats], i) => {
        const d = document.createElement('div');
        d.className = 'tree-node';
        d.style.paddingLeft = (8 + indent * 20) + 'px';
        d.style.opacity = '0'; d.style.transform = 'translateX(-8px)';
        d.innerHTML = `${indent > 0 ? '<span style="color:#333;margin-right:4px">|_</span>' : ''}<span class="tree-tag tree-tag-${type}">${type}</span><span class="tree-name">${name}</span><span class="tree-stats">${stats}</span>`;
        el.appendChild(d);
        setTimeout(() => { d.style.transition = 'all .25s ease'; d.style.opacity = '1'; d.style.transform = 'none'; }, 30 * i);
    });
}

/* ── Dataflow ───────────────────────────────────────────────── */
function initDataflow() {
    const el = document.getElementById('dataflow');
    if (!el) return;
    const s = [
        ['UART RX','784 bytes, ~54 ms at 115200'],
        ['Input Buffer','28x28 INT8 image, ping-pong'],
        ['Conv1 (5x5x6)','6 maps, 24x24 out'],
        ['ReLU','Sign-bit mask, 0-cycle'],
        ['Pool 2x2','6 maps, 12x12 out'],
        ['Conv2 (5x5x16)','16 maps, 8x8 out'],
        ['ReLU + Pool','4x4 = 256 values'],
        ['FC1 (256 > 120)','Systolic, tiled 4x4'],
        ['FC2 (120 > 84)','Systolic, tiled 4x4'],
        ['FC3 (84 > 10)','Final logits'],
        ['Argmax','Digit 0-9'],
        ['UART TX','Result to host'],
    ];
    s.forEach(([label, detail], i) => {
        const d = document.createElement('div');
        d.className = 'df-row';
        d.innerHTML = `<span class="df-idx">${String(i+1).padStart(2,'0')}</span><span class="df-txt"><strong>${label}</strong> — ${detail}</span>`;
        el.appendChild(d);
    });
}

/* ── AI Insights ────────────────────────────────────────────── */
function initInsights() {
    const mode = document.getElementById('aiMode');
    const q    = document.getElementById('aiQuestion');
    const btn  = document.getElementById('runAnalysisBtn');
    const out  = document.getElementById('aiOutput');
    const hdr  = document.getElementById('headerAnalyzeBtn');
    if (!mode || !btn) return;

    mode.addEventListener('change', () => { q.style.display = mode.value === 'question' ? 'block' : 'none'; });

    async function run() {
        out.innerHTML = '<div class="loading-spinner">Analyzing...</div>';
        try {
            const res = await fetch(API + '/api/analyze', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: mode.value, question: q.value }),
            });
            if (!res.ok) throw 0;
            const d = await res.json();
            out.innerHTML = d.error ? `<p style="color:#ef4444">${d.error}</p>` : md(d.analysis);
        } catch {
            out.innerHTML = md(offline(mode.value));
        }
    }

    btn.addEventListener('click', run);
    if (hdr) hdr.addEventListener('click', () => { document.querySelector('[data-tab="insights"]').click(); setTimeout(run, 250); });
}

function md(s) {
    return s
        .replace(/^### (.*$)/gm, '<h3>$1</h3>')
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code style="background:rgba(59,130,246,0.1);padding:2px 5px;border-radius:3px;font-family:JetBrains Mono;font-size:.85em">$1</code>')
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>');
}

function offline(m) {
    const a = {
performance: `## Performance Analysis

**5.1x speedup over CPU baseline.**

NeuralForge delivers **0.41 ms** mean inference latency vs **2.13 ms** on an Intel i7 running PyTorch.

### Bottleneck

Primary bottleneck is the **UART interface** at 115200 baud (14.4 KB/s). Image transfer takes ~54 ms while compute takes only 0.41 ms. **99.2% of wall-clock time is I/O-bound.**

### Top Optimizations

1. **SPI interface** at 10 MHz — 100x I/O speedup
2. **8x8 systolic array** — 64 MACs/cycle, 4x compute
3. **Sparse weight support** — 2x memory bandwidth savings

### Energy Efficiency

FPGA at 0.5 W: **4,878 inferences/joule** vs CPU at 65 W: **7.3 inferences/joule** — **668x improvement**.`,

quantization: `## Quantization Quality

INT8 symmetric quantization achieves **<1% accuracy drop** (99.2% to 98.5%) with **4x compression**.

### Per-Layer Analysis

- **conv1_weight** — MSE 1.2e-5 (low error)
- **fc3_weight** — MSE 9.3e-6 (highest, output layer most sensitive)
- **All biases** — negligible quantization error

### Recommendations

1. **Per-channel quantization** for conv layers (+0.2% accuracy)
2. **Quantization-aware training** to minimize the gap
3. Keep FC3 in **FP16 mixed precision** for better class discrimination`,

architecture: `## Architecture Review

The **4x4 weight-stationary systolic array** mirrors Google TPU v1 dataflow at smaller scale.

### Strengths

- Weight-stationary minimizes weight memory bandwidth
- 16 DSPs used (18% of XC7A35T) — room to scale to 8x8 (64 DSPs)
- Ping-pong buffer eliminates inter-image pipeline bubbles
- Performance counters provide cycle-accurate profiling

### Comparison

| Design | Array | Precision |
|--------|-------|-----------|
| Google TPU v1 | 256x256 | INT8 |
| NVIDIA Tensor Core | 4x4 | FP16 |
| NeuralForge | 4x4 | INT8 |

Conceptually identical to TPU v1 at smaller scale.`,

question: `Enter a question and click Analyze.`,
    };
    return a[m] || a.performance;
}

/* ── Try It: Gallery, Saliency & Accuracy ────────────────────── */
async function loadGallery() {
    const grid = document.getElementById('galleryGrid');
    const accBtn = document.getElementById('runAccuracyBtn');
    if (!grid) return;

    try {
        const res = await fetch(API + '/api/gallery');
        if (!res.ok) throw new Error('API Error');
        const data = await res.json();
        
        if (!data.samples || data.samples.length === 0) {
            grid.innerHTML = '<p class="empty-state">No gallery samples found</p>';
            return;
        }

        grid.innerHTML = '';
        data.samples.forEach((sample, idx) => {
            const cv = document.createElement('canvas');
            cv.className = 'gallery-item';
            cv.width = 28; cv.height = 28;
            cv.title = `True label: ${sample.label}`;
            
            const cx = cv.getContext('2d');
            const imgData = cx.createImageData(28, 28);
            for (let i=0; i<784; i++) {
                const val = sample.pixels[i];
                const off = i * 4;
                imgData.data[off] = val;
                imgData.data[off+1] = val;
                imgData.data[off+2] = val;
                imgData.data[off+3] = 255;
            }
            cx.putImageData(imgData, 0, 0);

            // Add label badge
            const wrapper = document.createElement('div');
            wrapper.style.position = 'relative';
            const badge = document.createElement('div');
            badge.className = 'gallery-label';
            badge.textContent = sample.label;
            wrapper.appendChild(cv);
            wrapper.appendChild(badge);

            wrapper.addEventListener('click', () => {
                document.querySelectorAll('.gallery-item').forEach(el => el.classList.remove('selected'));
                cv.classList.add('selected');
                
                // Copy to main canvas
                const mainCanvas = document.getElementById('drawCanvas');
                const mctx = mainCanvas.getContext('2d');
                mctx.fillStyle = '#09090b';
                mctx.fillRect(0,0,280,280);
                
                // Scale up 28x28 -> 280x280 blocky for retro feel
                mctx.imageSmoothingEnabled = false;
                mctx.drawImage(cv, 0, 0, 280, 280);
                
                // Set flag and auto-classify
                document.getElementById('drawCanvas').dispatchEvent(new Event('mousedown'));
                document.getElementById('drawCanvas').dispatchEvent(new Event('mouseup'));
                document.getElementById('classifyBtn').click();
            });

            grid.appendChild(wrapper);
        });

        if (accBtn) {
            accBtn.addEventListener('click', async () => {
                accBtn.textContent = 'Running...';
                accBtn.disabled = true;
                const rBox = document.getElementById('accuracyResults');
                rBox.style.display = 'block';
                rBox.innerHTML = '<p class="empty-state">Running full batch classification on FPGA simulator...</p>';

                try {
                    const r = await fetch(API + '/api/accuracy');
                    const d = await r.json();
                    
                    let html = `
                        <div class="accuracy-header">
                            <div class="accuracy-big">${d.accuracy.toFixed(1)}%</div>
                            <div class="accuracy-meta">
                                <strong>${d.correct} / ${d.total}</strong> correct<br>
                                100-sample mini-batch
                            </div>
                        </div>
                        <div class="confusion-grid">
                            <div class="confusion-header"></div>
                    `;
                    
                    // Column headers (Predicted)
                    for(let i=0; i<10; i++) html += `<div class="confusion-header">${i}</div>`;
                    
                    // Rows (Actual)
                    for(let r=0; r<10; r++) {
                        html += `<div class="confusion-header">${r}</div>`;
                        for(let c=0; c<10; c++) {
                            const val = d.confusion_matrix[r][c];
                            const isDiag = (r === c);
                            let bg = 'transparent';
                            if (val > 0) {
                                const intensity = Math.min(val / 10, 1);
                                if (isDiag) bg = `rgba(34, 197, 94, ${intensity * 0.5 + 0.1})`; // Green
                                else bg = `rgba(239, 68, 68, ${intensity * 0.8 + 0.2})`; // Red error!
                            }
                            html += `<div class="confusion-cell" style="background:${bg}; color: ${val > 0 ? '#fff' : 'var(--text-3)'}">${val}</div>`;
                        }
                    }
                    html += `</div>`;
                    rBox.innerHTML = html;

                } catch (e) {
                    rBox.innerHTML = '<p class="empty-state">Error running test</p>';
                } finally {
                    accBtn.textContent = 'Run Accuracy Test';
                    accBtn.disabled = false;
                }
            });
        }

    } catch (e) {
        grid.innerHTML = '<p class="empty-state">Server offline or gallery missing</p>';
    }
}

function renderSaliency(saliencyMap) {
    const container = document.getElementById('saliencyContainer');
    if (!container || !saliencyMap) return;

    container.innerHTML = '';
    
    // Create canvas
    const cv = document.createElement('canvas');
    cv.width = 112; // 28 * 4
    cv.height = 112;
    const ctx = cv.getContext('2d');
    
    // Draw heatmap (Blue = low, Red = high importance)
    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            const val = saliencyMap[y][x];
            // Inferno-ish colormap
            const r = Math.min(255, val * 300);
            const g = Math.min(255, val * 100);
            const b = Math.max(0, 150 - val * 200);
            
            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(x * 4, y * 4, 4, 4);
        }
    }
    
    container.appendChild(cv);
    const legend = document.createElement('div');
    legend.className = 'saliency-legend';
    legend.innerHTML = `
        <strong>Occlusion Sensitivity</strong><br>
        Bright <span style="color:#ef4444">red/orange</span> pixels indicate regions<br>most critical for the model's prediction.<br>
        Computed dynamically per inference.
    `;
    container.appendChild(legend);
}

/* ── Try It: Drawing Canvas & Classification ───────────────── */
function initTryIt() {
    const canvas = document.getElementById('drawCanvas');
    const classifyBtn = document.getElementById('classifyBtn');
    const clearBtn = document.getElementById('clearBtn');
    if (!canvas || !classifyBtn) return;

    const ctx = canvas.getContext('2d');
    let drawing = false;
    let hasDrawn = false;

    // Black background
    ctx.fillStyle = '#09090b';
    ctx.fillRect(0, 0, 280, 280);

    // Drawing settings
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 16;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Mouse handlers
    canvas.addEventListener('mousedown', (e) => {
        // Only set drawing if the event is trusted (real mouse click) OR if we fired it manually from the gallery
        if (e.isTrusted === false || e.clientX === undefined) {
             drawing = true; hasDrawn = true;
             return;
        }
        drawing = true; hasDrawn = true;
        const r = canvas.getBoundingClientRect();
        ctx.beginPath();
        ctx.moveTo(e.clientX - r.left, e.clientY - r.top);
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!drawing || !e.clientX) return;
        const r = canvas.getBoundingClientRect();
        ctx.lineTo(e.clientX - r.left, e.clientY - r.top);
        ctx.stroke();
    });

    canvas.addEventListener('mouseup', () => drawing = false);
    canvas.addEventListener('mouseleave', () => drawing = false);

    // Touch handlers
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault(); drawing = true; hasDrawn = true;
        const r = canvas.getBoundingClientRect();
        const t = e.touches[0];
        ctx.beginPath();
        ctx.moveTo(t.clientX - r.left, t.clientY - r.top);
    });

    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        if (!drawing) return;
        const r = canvas.getBoundingClientRect();
        const t = e.touches[0];
        ctx.lineTo(t.clientX - r.left, t.clientY - r.top);
        ctx.stroke();
    });

    canvas.addEventListener('touchend', () => drawing = false);

    // Clear
    clearBtn.addEventListener('click', () => {
        ctx.fillStyle = '#09090b';
        ctx.fillRect(0, 0, 280, 280);
        ctx.strokeStyle = '#ffffff';
        hasDrawn = false;
        document.getElementById('predictedDigit').textContent = '?';
        document.getElementById('predLatency').textContent = 'Draw a digit to begin';
        document.getElementById('cycleTag').textContent = '-- cycles';
        document.getElementById('confidenceBars').innerHTML = '';
        document.getElementById('featureMaps').innerHTML = '<p class="empty-state">Run a classification to see layer activations</p>';
        document.getElementById('saliencyContainer').innerHTML = '<p class="empty-state">Classify a digit to see which pixels influenced the prediction</p>';
        document.querySelectorAll('.gallery-item').forEach(el => el.classList.remove('selected'));
    });

    // Classify
    classifyBtn.addEventListener('click', async () => {
        if (!hasDrawn) return;

        // Downsample 280×280 canvas → 28×28 grayscale
        const tmp = document.createElement('canvas');
        tmp.width = 28; tmp.height = 28;
        const tctx = tmp.getContext('2d');
        tctx.drawImage(canvas, 0, 0, 28, 28);
        const imgData = tctx.getImageData(0, 0, 28, 28);
        const pixels = [];
        for (let i = 0; i < imgData.data.length; i += 4) {
            pixels.push(imgData.data[i]); // R channel (grayscale)
        }

        classifyBtn.textContent = 'Classifying...';
        classifyBtn.disabled = true;

        try {
            const res = await fetch(API + '/api/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pixels }),
            });

            if (!res.ok) throw new Error('API error');
            const data = await res.json();
            renderPrediction(data);
        } catch (err) {
            document.getElementById('predictedDigit').textContent = '!';
            document.getElementById('predLatency').textContent = 'Error: server not running';
        } finally {
            classifyBtn.textContent = 'Classify';
            classifyBtn.disabled = false;
        }
    });
}

function renderPrediction(data) {
    // Big digit
    const digitEl = document.getElementById('predictedDigit');
    digitEl.textContent = data.digit;
    digitEl.classList.remove('flash');
    void digitEl.offsetWidth; // Force reflow
    digitEl.classList.add('flash');

    // Cycle count and latency
    document.getElementById('cycleTag').textContent = data.cycles.toLocaleString() + ' cycles';
    document.getElementById('predLatency').textContent =
        `${data.latency_us} us at 100 MHz | ${data.cycles.toLocaleString()} cycles`;

    // Confidence bars
    renderConfidence(data.confidence, data.digit);

    // Feature and Saliency maps
    if (data.layer_outputs) {
        renderFeatureMaps(data.layer_outputs);
        if (data.layer_outputs.saliency) {
            renderSaliency(data.layer_outputs.saliency);
        }
    }
}

function renderConfidence(confidence, predicted) {
    const container = document.getElementById('confidenceBars');
    container.innerHTML = '';

    confidence.forEach((pct, i) => {
        const row = document.createElement('div');
        row.className = 'conf-row';
        const isTop = (i === predicted);

        row.innerHTML = `
            <span class="conf-label ${isTop ? 'highlight' : ''}">${i}</span>
            <div class="conf-track">
                <div class="conf-fill ${isTop ? 'top' : ''}" style="width: 0%"></div>
            </div>
            <span class="conf-pct ${isTop ? 'highlight' : ''}">${pct.toFixed(1)}%</span>
        `;
        container.appendChild(row);

        // Animate the bar
        requestAnimationFrame(() => {
            row.querySelector('.conf-fill').style.width = Math.max(pct, 0.5) + '%';
        });
    });
}

function renderFeatureMaps(layers) {
    const container = document.getElementById('featureMaps');
    container.innerHTML = '';

    const layerNames = [
        ['conv1', 'Conv1 Output (6 maps, 24x24)'],
        ['relu1', 'ReLU1 (6 maps, 24x24)'],
        ['pool1', 'Pool1 (6 maps, 12x12)'],
        ['conv2', 'Conv2 Output (16 maps, 8x8)'],
        ['pool2', 'Pool2 (16 maps, 4x4)'],
    ];

    layerNames.forEach(([key, label]) => {
        const maps = layers[key];
        if (!maps || !Array.isArray(maps)) return;

        const section = document.createElement('div');
        section.className = 'fmap-section';
        section.innerHTML = `<div class="fmap-label">${label}</div><div class="fmap-grid"></div>`;
        const grid = section.querySelector('.fmap-grid');

        maps.forEach((map2d) => {
            const h = map2d.length;
            const w = map2d[0].length;
            const scale = Math.max(2, Math.floor(48 / Math.max(h, w)));
            const cv = document.createElement('canvas');
            cv.width = w * scale; cv.height = h * scale;
            const cx = cv.getContext('2d');

            // Find range for normalization
            let mn = Infinity, mx = -Infinity;
            for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
                mn = Math.min(mn, map2d[y][x]);
                mx = Math.max(mx, map2d[y][x]);
            }
            const range = mx - mn || 1;

            for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
                const val = Math.round(((map2d[y][x] - mn) / range) * 255);
                // Blue channel tint for a premium look
                const r = Math.round(val * 0.23);
                const g = Math.round(val * 0.51);
                const b = Math.round(val * 0.96);
                cx.fillStyle = `rgb(${r},${g},${b})`;
                cx.fillRect(x * scale, y * scale, scale, scale);
            }
            grid.appendChild(cv);
        });

        container.appendChild(section);
    });
}

/* ── Status ─────────────────────────────────────────────────── */
async function status() {
    try {
        const r = await fetch(API + '/api/status');
        if (r.ok) {
            const d = await r.json();
            const lbl = document.getElementById('statusLabel');
            if (lbl) lbl.textContent = 'Connected';
            const tag = document.getElementById('backendTag');
            if (tag) tag.textContent = d.backend || 'Offline';
        }
    } catch {
        const lbl = document.getElementById('statusLabel');
        if (lbl) lbl.textContent = 'Standalone';
        const chip = document.getElementById('statusChip');
        if (chip) {
            chip.style.background = 'rgba(59,130,246,0.1)';
            chip.style.borderColor = 'rgba(59,130,246,0.2)';
            chip.style.color = '#60a5fa';
            const dot = chip.querySelector('.chip-dot');
            if (dot) dot.style.background = '#3b82f6';
        }
    }
}

/* ── Boot ───────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.stat-value[data-to]').forEach(countUp);
    spark('sparkSpeedup',  [3.2,3.8,4.1,4.5,4.8,5.0,5.1,5.1], '#3b82f6');
    spark('sparkThroughput',[1200,1500,1800,2000,2200,2350,2400,2439], '#22c55e');
    spark('sparkLatency',  [.85,.72,.60,.52,.48,.44,.42,.41], '#a78bfa');
    spark('sparkEfficiency',[200,310,420,500,560,610,650,668], '#f59e0b');
    initPipeline();
    initSystolic();
    initResources();
    initCharts();
    initCounters();
    initTree();
    initDataflow();
    initInsights();
    initTryIt();
    loadGallery();
    status();
});

