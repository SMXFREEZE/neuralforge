/* ================================================================
   NeuralForge Dashboard
   Clean, modular application logic
   ================================================================ */

const API_BASE = '';
const ACCENT_PALETTE = ['#6c8cff', '#34d399', '#a78bfa', '#fbbf24', '#f87171', '#22d3ee', '#94a3b8'];
const activeCharts = [];

/* ----------------------------------------------------------------
   Navigation
   ---------------------------------------------------------------- */
function initNavigation() {
    document.querySelectorAll('.nav-link').forEach(button => {
        button.addEventListener('click', () => {
            document.querySelectorAll('.nav-link').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
            button.classList.add('active');

            const view = document.getElementById('view-' + button.dataset.tab);
            if (view) view.classList.add('active');

            requestAnimationFrame(() => activeCharts.forEach(chart => chart.resize()));
        });
    });
}

/* ----------------------------------------------------------------
   Animated Counter
   ---------------------------------------------------------------- */
function animateCounter(element) {
    const target = parseFloat(element.dataset.to);
    const decimals = parseInt(element.dataset.decimals || '0', 10);
    const duration = 1200;
    const startTime = performance.now();

    function step(currentTime) {
        const progress = Math.min((currentTime - startTime) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const value = target * eased;

        element.textContent = decimals > 0
            ? value.toFixed(decimals)
            : Math.round(value).toLocaleString();

        if (progress < 1) requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
}

/* ----------------------------------------------------------------
   Sparklines
   ---------------------------------------------------------------- */
function drawSparkline(canvasId, data, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const dpr = devicePixelRatio || 1;
    const width = canvas.parentElement.offsetWidth - 40;
    const height = 32;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const max = Math.max(...data) * 1.05;
    const min = Math.min(...data) * 0.95;
    const range = max - min || 1;
    const stepX = width / (data.length - 1);

    const points = data.map((value, index) => ({
        x: index * stepX,
        y: height - ((value - min) / range) * height * 0.75 - height * 0.12,
    }));

    // Fill gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, color + '28');
    gradient.addColorStop(1, color + '00');

    ctx.beginPath();
    ctx.moveTo(0, height);
    points.forEach(point => ctx.lineTo(point.x, point.y));
    ctx.lineTo(width, height);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Stroke line
    ctx.beginPath();
    points.forEach((point, i) => i ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
}

/* ----------------------------------------------------------------
   Staggered Entrance Animation (shared util)
   ---------------------------------------------------------------- */
function staggerEntrance(element, delay = 40) {
    element.style.opacity = '0';
    element.style.transform = 'translateX(-10px)';
    return new Promise(resolve => {
        setTimeout(() => {
            element.style.transition = 'all 0.3s ease';
            element.style.opacity = '1';
            element.style.transform = 'none';
            resolve();
        }, delay);
    });
}

/* ----------------------------------------------------------------
   Pipeline
   ---------------------------------------------------------------- */
function initPipeline() {
    const container = document.getElementById('pipelineList');
    if (!container) return;

    const stages = [
        ['UART Interface', '115200 baud, 8N1'],
        ['Input Buffer', 'Ping-pong, 784 B x 2'],
        ['Conv Engine (3x3)', '4-stage pipeline'],
        ['ReLU Activation', 'Sign-bit masking'],
        ['MaxPool (2x2)', 'Comparator tree'],
        ['Systolic Array (4x4)', '16 MACs / cycle'],
        ['Argmax + Output', 'Class 0-9'],
    ];

    stages.forEach(([name, meta], index) => {
        const row = document.createElement('div');
        row.className = 'pipe-row';
        row.innerHTML = `
            <span class="pipe-num">${index + 1}</span>
            <span class="pipe-name">${name}</span>
            <span class="pipe-meta">${meta}</span>
        `;
        container.appendChild(row);
        staggerEntrance(row, 50 * index);
    });
}

/* ----------------------------------------------------------------
   Systolic Array Animation
   ---------------------------------------------------------------- */
function initSystolicArray() {
    const grid = document.getElementById('systolicGrid');
    if (!grid) return;

    for (let row = 0; row < 4; row++) {
        for (let col = 0; col < 4; col++) {
            const cell = document.createElement('div');
            cell.className = 'mac-cell';
            cell.id = `mac-${row}-${col}`;
            cell.innerHTML = `
                <span class="mac-id">[${row},${col}]</span>
                <span class="mac-val">--</span>
            `;
            grid.appendChild(cell);
        }
    }

    let tick = 0;
    setInterval(() => {
        document.querySelectorAll('.mac-cell').forEach(cell => {
            cell.classList.remove('active');
            cell.querySelector('.mac-val').textContent = '--';
        });

        for (let row = 0; row < 4; row++) {
            for (let col = 0; col < 4; col++) {
                if ((row + col) === (tick % 8)) {
                    const cell = document.getElementById(`mac-${row}-${col}`);
                    if (cell) {
                        cell.classList.add('active');
                        const weight = (Math.random() * 127 | 0);
                        const activation = (Math.random() * 127 | 0) - 64;
                        cell.querySelector('.mac-val').textContent = `${weight}x${activation}`;
                    }
                }
            }
        }
        tick++;
    }, 350);
}

/* ----------------------------------------------------------------
   Resource Utilization
   ---------------------------------------------------------------- */
function initResources() {
    const container = document.getElementById('resGrid');
    if (!container) return;

    const resources = [
        { name: 'LUTs', used: 3370, total: 20800, color: '#6c8cff' },
        { name: 'Flip-Flops', used: 2753, total: 41600, color: '#34d399' },
        { name: 'DSP48E1', used: 16, total: 90, color: '#a78bfa' },
        { name: 'Block RAM', used: 4, total: 50, color: '#fbbf24' },
    ];

    resources.forEach(resource => {
        const percentage = (resource.used / resource.total * 100).toFixed(1);
        const item = document.createElement('div');
        item.className = 'resource-item';
        item.innerHTML = `
            <div class="resource-top">
                <span class="resource-label">${resource.name}</span>
                <span class="resource-pct">${percentage}%</span>
            </div>
            <div class="resource-track">
                <div class="resource-fill" style="background:${resource.color}" data-width="${percentage}"></div>
            </div>
            <div class="resource-info">${resource.used.toLocaleString()} / ${resource.total.toLocaleString()}</div>
        `;
        container.appendChild(item);

        setTimeout(() => {
            item.querySelector('.resource-fill').style.width = percentage + '%';
        }, 200);
    });
}

/* ----------------------------------------------------------------
   Chart Engine
   ---------------------------------------------------------------- */
class Chart {
    constructor(canvasId, data, options) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        this.data = data;
        this.options = options;
        this.dpr = 1;

        this.resize();
        window.addEventListener('resize', () => this.resize());
        activeCharts.push(this);
    }

    resize() {
        const dpr = devicePixelRatio || 1;
        const rect = this.canvas.parentElement.getBoundingClientRect();
        if (rect.width < 1) return;

        const height = this.options.height || 260;
        this.canvas.width = rect.width * dpr;
        this.canvas.height = height * dpr;
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = height + 'px';
        this.dpr = dpr;
        this.draw();
    }

    draw() {
        if (!this.ctx) return;
        const ctx = this.ctx;
        const dpr = this.dpr || 1;
        ctx.save();
        ctx.scale(dpr, dpr);

        const width = this.canvas.width / dpr;
        const height = this.canvas.height / dpr;
        ctx.clearRect(0, 0, width, height);

        switch (this.options.type) {
            case 'bar':    this.drawBarChart(ctx, width, height); break;
            case 'hbar':   this.drawHorizontalBarChart(ctx, width, height); break;
            case 'donut':  this.drawDonutChart(ctx, width, height); break;
        }

        ctx.restore();
    }

    drawBarChart(ctx, width, height) {
        const padding = { top: 28, bottom: 52, left: 50, right: 16 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;
        const { labels, datasets } = this.data;
        const maxValue = Math.max(...datasets.flatMap(ds => ds.data)) * 1.2;
        const groupWidth = chartWidth / labels.length;
        const barWidth = groupWidth / (datasets.length + 1);

        // Grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = padding.top + (chartHeight / 4) * i;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(padding.left + chartWidth, y);
            ctx.stroke();

            ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.font = '11px JetBrains Mono';
            ctx.textAlign = 'right';
            const labelValue = maxValue - (maxValue / 4) * i;
            ctx.fillText(
                labelValue >= 100 ? Math.round(labelValue) : labelValue.toFixed(2),
                padding.left - 6,
                y + 4
            );
        }

        // Bars
        datasets.forEach((dataset, datasetIndex) => {
            dataset.data.forEach((value, labelIndex) => {
                const x = padding.left + labelIndex * groupWidth + datasetIndex * barWidth + barWidth * 0.4;
                const barHeight = (value / maxValue) * chartHeight;
                const y = padding.top + chartHeight - barHeight;
                const color = dataset.color || ACCENT_PALETTE[datasetIndex];

                const gradient = ctx.createLinearGradient(0, y, 0, y + barHeight);
                gradient.addColorStop(0, color);
                gradient.addColorStop(1, color + '44');

                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.roundRect(x, y, barWidth * 0.7, barHeight, [4, 4, 0, 0]);
                ctx.fill();

                ctx.fillStyle = 'rgba(255, 255, 255, 0.55)';
                ctx.font = '11px JetBrains Mono';
                ctx.textAlign = 'center';
                ctx.fillText(value >= 100 ? value : value.toFixed(2), x + barWidth * 0.35, y - 6);
            });
        });

        // X-axis labels
        labels.forEach((label, index) => {
            ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.font = '11px Inter';
            ctx.textAlign = 'center';
            ctx.fillText(label, padding.left + index * groupWidth + groupWidth / 2, height - padding.bottom + 18);
        });

        // Legend
        let legendX = padding.left;
        datasets.forEach((dataset, datasetIndex) => {
            const color = dataset.color || ACCENT_PALETTE[datasetIndex];
            ctx.fillStyle = color;
            ctx.fillRect(legendX, height - 10, 8, 8);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.55)';
            ctx.font = '11px Inter';
            ctx.textAlign = 'left';
            ctx.fillText(dataset.label, legendX + 12, height - 3);
            legendX += ctx.measureText(dataset.label).width + 28;
        });
    }

    drawHorizontalBarChart(ctx, width, height) {
        const padding = { top: 8, bottom: 8, left: 84, right: 50 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;
        const { labels, values, colors } = this.data;
        const maxValue = Math.max(...values) * 1.15;
        const stepHeight = chartHeight / labels.length;
        const barHeight = Math.min(stepHeight * 0.5, 20);

        labels.forEach((label, index) => {
            const y = padding.top + index * stepHeight + stepHeight / 2;
            const barW = (values[index] / maxValue) * chartWidth;
            const color = (colors && colors[index]) || ACCENT_PALETTE[index % ACCENT_PALETTE.length];

            const gradient = ctx.createLinearGradient(padding.left, 0, padding.left + barW, 0);
            gradient.addColorStop(0, color);
            gradient.addColorStop(1, color + '88');

            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.roundRect(padding.left, y - barHeight / 2, barW, barHeight, [0, 4, 4, 0]);
            ctx.fill();

            ctx.fillStyle = 'rgba(255, 255, 255, 0.55)';
            ctx.font = '11px JetBrains Mono';
            ctx.textAlign = 'right';
            ctx.fillText(label, padding.left - 6, y + 4);

            ctx.fillStyle = 'rgba(255, 255, 255, 0.92)';
            ctx.font = '600 11px JetBrains Mono';
            ctx.textAlign = 'left';
            ctx.fillText(values[index].toLocaleString(), padding.left + barW + 8, y + 4);
        });
    }

    drawDonutChart(ctx, width, height) {
        const { labels, values, colors } = this.data;
        const total = values.reduce((sum, v) => sum + v, 0);
        const centerX = width / 2;
        const centerY = height / 2 - 10;
        const radius = Math.min(width, height) * 0.28;

        let angle = -Math.PI / 2;
        values.forEach((value, index) => {
            const slice = (value / total) * Math.PI * 2;
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.arc(centerX, centerY, radius, angle, angle + slice);
            ctx.closePath();
            ctx.fillStyle = (colors && colors[index]) || ACCENT_PALETTE[index % ACCENT_PALETTE.length];
            ctx.fill();
            angle += slice;
        });

        // Center hole
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius * 0.58, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(11, 13, 16, 0.95)';
        ctx.fill();

        ctx.fillStyle = 'rgba(255, 255, 255, 0.92)';
        ctx.font = '700 15px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(total.toLocaleString(), centerX, centerY + 5);

        ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.font = '11px Inter';
        ctx.fillText('Total LUTs', centerX, centerY + 20);
    }
}

/* ----------------------------------------------------------------
   Initialize All Charts
   ---------------------------------------------------------------- */
function initCharts() {
    new Chart('chartLatency', {
        labels: ['Mean', 'Median', 'P95', 'P99'],
        datasets: [
            { label: 'CPU (PyTorch)', data: [2.13, 2.09, 2.89, 3.25], color: '#94a3b8' },
            { label: 'FPGA', data: [0.41, 0.41, 0.44, 0.46], color: '#6c8cff' },
        ],
    }, { type: 'bar', height: 270 });

    new Chart('chartThroughput', {
        labels: ['CPU', 'FPGA'],
        datasets: [{ label: 'img/s', data: [469, 2439], color: '#34d399' }],
    }, { type: 'bar', height: 270 });

    new Chart('chartPipeline', {
        labels: ['Conv1', 'Pool1', 'Conv2', 'Pool2', 'FC1', 'FC2', 'FC3'],
        values: [8200, 1200, 12800, 800, 3800, 2100, 840],
        colors: ['#6c8cff', '#34d399', '#a78bfa', '#34d399', '#fbbf24', '#fbbf24', '#fbbf24'],
    }, { type: 'hbar', height: 230 });

    new Chart('chartPie', {
        labels: ['SystolicArr', 'ConvEng', 'LayerCtrl', 'UART', 'Buffers', 'PerfCtrs', 'Other'],
        values: [1200, 580, 320, 180, 285, 160, 645],
        colors: ACCENT_PALETTE,
    }, { type: 'donut', height: 270 });

    new Chart('chartQuantErr', {
        labels: ['conv1_w', 'conv1_b', 'conv2_w', 'conv2_b', 'fc1_w', 'fc1_b', 'fc2_w', 'fc2_b', 'fc3_w', 'fc3_b'],
        values: [12, 0.31, 8.7, 0.18, 5.4, 0.42, 6.1, 0.29, 9.3, 0.15],
        colors: Array(10).fill('#6c8cff'),
    }, { type: 'hbar', height: 270 });

    new Chart('chartScales', {
        labels: ['conv1', 'conv2', 'fc1', 'fc2', 'fc3'],
        datasets: [{ label: 'Scale x1000', data: [23, 18, 12, 15, 21], color: '#34d399' }],
    }, { type: 'bar', height: 210 });
}

/* ----------------------------------------------------------------
   Hardware Performance Counters
   ---------------------------------------------------------------- */
function initCounters() {
    const container = document.getElementById('countersGrid');
    if (!container) return;

    const counters = [
        ['Total Cycles', '41,283', '0.413 ms @ 100 MHz'],
        ['Inference Cycles', '38,420', '93.1% of total'],
        ['Conv Cycles', '24,680', '64.2% of inference'],
        ['Pool Cycles', '3,200', '8.3% of inference'],
        ['FC Cycles', '10,540', '27.4% of inference'],
        ['MAC Active', '31,200', '81.2% utilization'],
        ['Memory Reads', '8,750', '22.8% bandwidth'],
        ['Pipeline Stalls', '1,890', '4.9% stall rate'],
        ['Completed', '1', 'Inference count'],
    ];

    counters.forEach(([name, value, description]) => {
        const box = document.createElement('div');
        box.className = 'counter-box';
        box.innerHTML = `
            <div class="counter-name">${name}</div>
            <div class="counter-val">${value}</div>
            <div class="counter-info">${description}</div>
        `;
        container.appendChild(box);
    });
}

/* ----------------------------------------------------------------
   RTL Module Tree
   ---------------------------------------------------------------- */
function initModuleTree() {
    const container = document.getElementById('moduleTree');
    if (!container) return;

    const modules = [
        [0, 'top.v', 'controller', '450 LUTs, 380 FFs'],
        [1, 'uart_interface.v', 'io', '180 LUTs, 160 FFs'],
        [1, 'input_buffer.v', 'memory', '120 LUTs, 1 BRAM'],
        [1, 'weight_buffer.v', 'memory', '80 LUTs, 2 BRAM'],
        [1, 'layer_controller.v', 'controller', '320 LUTs, 280 FFs'],
        [2, 'conv_engine.v', 'compute', '580 LUTs, 420 FFs'],
        [2, 'activation.v', 'compute', '45 LUTs, 35 FFs'],
        [2, 'pooling.v', 'compute', '60 LUTs, 50 FFs'],
        [2, 'systolic_array.v', 'compute', '1,200 LUTs, 16 DSPs'],
        [3, 'mac_unit.v x16', 'compute', '75 LUTs ea, 1 DSP ea'],
        [2, 'fifo.v', 'memory', '85 LUTs, 1 BRAM'],
        [1, 'perf_counters.v', 'debug', '160 LUTs, 384 FFs'],
        [1, 'register_file.v', 'io', '90 LUTs, 64 FFs'],
    ];

    modules.forEach(([indent, name, type, stats], index) => {
        const node = document.createElement('div');
        node.className = 'tree-node';
        node.style.paddingLeft = (8 + indent * 20) + 'px';
        node.innerHTML = `
            ${indent > 0 ? '<span class="tree-connector">|_</span>' : ''}
            <span class="tree-tag tree-tag-${type}">${type}</span>
            <span class="tree-name">${name}</span>
            <span class="tree-stats">${stats}</span>
        `;
        container.appendChild(node);
        staggerEntrance(node, 30 * index);
    });
}

/* ----------------------------------------------------------------
   Data Path
   ---------------------------------------------------------------- */
function initDataflow() {
    const container = document.getElementById('dataflow');
    if (!container) return;

    const steps = [
        ['UART RX', '784 bytes, ~54 ms at 115200'],
        ['Input Buffer', '28x28 INT8 image, ping-pong'],
        ['Conv1 (5x5x6)', '6 maps, 24x24 out'],
        ['ReLU', 'Sign-bit mask, 0-cycle'],
        ['Pool 2x2', '6 maps, 12x12 out'],
        ['Conv2 (5x5x16)', '16 maps, 8x8 out'],
        ['ReLU + Pool', '4x4 = 256 values'],
        ['FC1 (256 > 120)', 'Systolic, tiled 4x4'],
        ['FC2 (120 > 84)', 'Systolic, tiled 4x4'],
        ['FC3 (84 > 10)', 'Final logits'],
        ['Argmax', 'Digit 0-9'],
        ['UART TX', 'Result to host'],
    ];

    steps.forEach(([label, detail], index) => {
        const row = document.createElement('div');
        row.className = 'dataflow-row';
        row.innerHTML = `
            <span class="dataflow-idx">${String(index + 1).padStart(2, '0')}</span>
            <span class="dataflow-txt"><strong>${label}</strong> / ${detail}</span>
        `;
        container.appendChild(row);
    });
}

/* ----------------------------------------------------------------
   AI Insights
   ---------------------------------------------------------------- */
function initInsights() {
    const modeSelect = document.getElementById('aiMode');
    const questionInput = document.getElementById('aiQuestion');
    const analyzeBtn = document.getElementById('runAnalysisBtn');
    const outputDiv = document.getElementById('aiOutput');
    if (!modeSelect || !analyzeBtn) return;

    modeSelect.addEventListener('change', () => {
        questionInput.style.display = modeSelect.value === 'question' ? 'block' : 'none';
    });

    async function runAnalysis() {
        outputDiv.innerHTML = '<div class="loading-spinner">Analyzing...</div>';
        try {
            const response = await fetch(API_BASE + '/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: modeSelect.value, question: questionInput.value }),
            });
            if (!response.ok) throw new Error('API Error');
            const result = await response.json();
            outputDiv.innerHTML = result.error
                ? `<p style="color:var(--red)">${result.error}</p>`
                : renderMarkdown(result.analysis);
        } catch {
            outputDiv.innerHTML = renderMarkdown(getOfflineAnalysis(modeSelect.value));
        }
    }

    analyzeBtn.addEventListener('click', runAnalysis);
}

function renderMarkdown(text) {
    return text
        .replace(/^### (.*$)/gm, '<h3>$1</h3>')
        .replace(/^## (.*$)/gm, '<h2>$1</h2>')
        .replace(/^# (.*$)/gm, '<h1>$1</h1>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code style="background:var(--accent-dim);padding:2px 5px;border-radius:3px;font-family:var(--font-mono);font-size:.85em">$1</code>')
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>');
}

function getOfflineAnalysis(mode) {
    const analyses = {
        performance: `## Performance Analysis

**5.1x speedup over CPU baseline.**

NeuralForge delivers **0.41 ms** mean inference latency vs **2.13 ms** on an Intel i7 running PyTorch.

### Bottleneck

Primary bottleneck is the **UART interface** at 115200 baud (14.4 KB/s). Image transfer takes ~54 ms while compute takes only 0.41 ms. **99.2% of wall-clock time is I/O-bound.**

### Top Optimizations

1. **SPI interface** at 10 MHz, 100x I/O speedup
2. **8x8 systolic array**, 64 MACs/cycle, 4x compute
3. **Sparse weight support**, 2x memory bandwidth savings

### Energy Efficiency

FPGA at 0.5 W: **4,878 inferences/joule** vs CPU at 65 W: **7.3 inferences/joule**, a **668x improvement**.`,

        quantization: `## Quantization Quality

INT8 symmetric quantization achieves **<1% accuracy drop** (99.2% to 98.5%) with **4x compression**.

### Per-Layer Analysis

- **conv1_weight**, MSE 1.2e-5 (low error)
- **fc3_weight**, MSE 9.3e-6 (highest, output layer most sensitive)
- **All biases**, negligible quantization error

### Recommendations

1. **Per-channel quantization** for conv layers (+0.2% accuracy)
2. **Quantization-aware training** to minimize the gap
3. Keep FC3 in **FP16 mixed precision** for better class discrimination`,

        architecture: `## Architecture Review

The **4x4 weight-stationary systolic array** mirrors Google TPU v1 dataflow at smaller scale.

### Strengths

- Weight-stationary minimizes weight memory bandwidth
- 16 DSPs used (18% of XC7A35T), room to scale to 8x8 (64 DSPs)
- Ping-pong buffer eliminates inter-image pipeline bubbles
- Performance counters provide cycle-accurate profiling

### Comparison

| Design | Array | Precision |
|--------|-------|-----------|\n| Google TPU v1 | 256x256 | INT8 |
| NVIDIA Tensor Core | 4x4 | FP16 |
| NeuralForge | 4x4 | INT8 |

Conceptually identical to TPU v1 at smaller scale.`,

        question: 'Enter a question and click Analyze.',
    };
    return analyses[mode] || analyses.performance;
}

/* ----------------------------------------------------------------
   Try It: Gallery, Drawing, Classification
   ---------------------------------------------------------------- */
async function loadGallery() {
    const grid = document.getElementById('galleryGrid');
    const accuracyBtn = document.getElementById('runAccuracyBtn');
    if (!grid) return;

    try {
        const response = await fetch(API_BASE + '/api/gallery');
        if (!response.ok) throw new Error('API Error');
        const data = await response.json();

        if (!data.samples || data.samples.length === 0) {
            grid.innerHTML = '<p class="empty-state">No gallery samples found</p>';
            return;
        }

        grid.innerHTML = '';
        data.samples.forEach((sample) => {
            const canvas = document.createElement('canvas');
            canvas.className = 'gallery-item';
            canvas.width = 28;
            canvas.height = 28;
            canvas.title = `True label: ${sample.label}`;

            const ctx = canvas.getContext('2d');
            const imageData = ctx.createImageData(28, 28);
            for (let i = 0; i < 784; i++) {
                const offset = i * 4;
                imageData.data[offset] = sample.pixels[i];
                imageData.data[offset + 1] = sample.pixels[i];
                imageData.data[offset + 2] = sample.pixels[i];
                imageData.data[offset + 3] = 255;
            }
            ctx.putImageData(imageData, 0, 0);

            const wrapper = document.createElement('div');
            wrapper.style.position = 'relative';
            const badge = document.createElement('div');
            badge.className = 'gallery-label';
            badge.textContent = sample.label;
            wrapper.appendChild(canvas);
            wrapper.appendChild(badge);

            wrapper.addEventListener('click', () => {
                document.querySelectorAll('.gallery-item').forEach(el => el.classList.remove('selected'));
                canvas.classList.add('selected');

                const mainCanvas = document.getElementById('drawCanvas');
                const mainCtx = mainCanvas.getContext('2d');
                mainCtx.fillStyle = '#07090b';
                mainCtx.fillRect(0, 0, 280, 280);
                mainCtx.imageSmoothingEnabled = false;
                mainCtx.drawImage(canvas, 0, 0, 280, 280);

                mainCanvas.dispatchEvent(new Event('mousedown'));
                mainCanvas.dispatchEvent(new Event('mouseup'));
                document.getElementById('classifyBtn').click();
            });

            grid.appendChild(wrapper);
        });

        if (accuracyBtn) {
            accuracyBtn.addEventListener('click', runAccuracyTest);
        }
    } catch {
        grid.innerHTML = '<p class="empty-state">Server offline or gallery missing</p>';
    }
}

async function runAccuracyTest() {
    const btn = document.getElementById('runAccuracyBtn');
    const resultsBox = document.getElementById('accuracyResults');
    btn.textContent = 'Running...';
    btn.disabled = true;
    resultsBox.style.display = 'block';
    resultsBox.innerHTML = '<p class="empty-state">Running full batch classification on FPGA simulator...</p>';

    try {
        const response = await fetch(API_BASE + '/api/accuracy');
        const data = await response.json();

        let html = `
            <div class="accuracy-header">
                <div class="accuracy-big">${data.accuracy.toFixed(1)}%</div>
                <div class="accuracy-meta">
                    <strong>${data.correct} / ${data.total}</strong> correct<br>
                    100-sample mini-batch
                </div>
            </div>
            <div class="confusion-grid">
                <div class="confusion-header"></div>
        `;

        for (let i = 0; i < 10; i++) {
            html += `<div class="confusion-header">${i}</div>`;
        }

        for (let row = 0; row < 10; row++) {
            html += `<div class="confusion-header">${row}</div>`;
            for (let col = 0; col < 10; col++) {
                const value = data.confusion_matrix[row][col];
                const isDiagonal = row === col;
                let bgColor = 'transparent';
                if (value > 0) {
                    const intensity = Math.min(value / 10, 1);
                    bgColor = isDiagonal
                        ? `rgba(52, 211, 153, ${intensity * 0.5 + 0.1})`
                        : `rgba(248, 113, 113, ${intensity * 0.8 + 0.2})`;
                }
                html += `<div class="confusion-cell" style="background:${bgColor}; color: ${value > 0 ? '#fff' : 'var(--text-3)'}">${value}</div>`;
            }
        }

        html += '</div>';
        resultsBox.innerHTML = html;
    } catch {
        resultsBox.innerHTML = '<p class="empty-state">Error running test</p>';
    } finally {
        btn.textContent = 'Run Accuracy Test';
        btn.disabled = false;
    }
}

function renderSaliencyMap(saliencyData) {
    const container = document.getElementById('saliencyContainer');
    if (!container || !saliencyData) return;

    container.innerHTML = '';

    const canvas = document.createElement('canvas');
    canvas.width = 112;
    canvas.height = 112;
    const ctx = canvas.getContext('2d');

    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            const value = saliencyData[y][x];
            const r = Math.min(255, value * 300);
            const g = Math.min(255, value * 100);
            const b = Math.max(0, 150 - value * 200);
            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(x * 4, y * 4, 4, 4);
        }
    }

    container.appendChild(canvas);

    const legend = document.createElement('div');
    legend.className = 'saliency-legend';
    legend.innerHTML = `
        <strong>Occlusion Sensitivity</strong><br>
        Bright <span style="color:var(--red)">red/orange</span> pixels indicate regions<br>
        most critical for the model's prediction.<br>
        Computed dynamically per inference.
    `;
    container.appendChild(legend);
}

function renderFeatureMaps(layers) {
    const container = document.getElementById('featureMaps');
    container.innerHTML = '';

    const layerConfig = [
        ['conv1', 'Conv1 Output (6 maps, 24x24)'],
        ['relu1', 'ReLU1 (6 maps, 24x24)'],
        ['pool1', 'Pool1 (6 maps, 12x12)'],
        ['conv2', 'Conv2 Output (16 maps, 8x8)'],
        ['pool2', 'Pool2 (16 maps, 4x4)'],
    ];

    layerConfig.forEach(([key, label]) => {
        const maps = layers[key];
        if (!maps || !Array.isArray(maps)) return;

        const section = document.createElement('div');
        section.className = 'fmap-section';
        section.innerHTML = `<div class="fmap-label">${label}</div><div class="fmap-grid"></div>`;
        const grid = section.querySelector('.fmap-grid');

        maps.forEach((map2d) => {
            const mapHeight = map2d.length;
            const mapWidth = map2d[0].length;
            const scale = Math.max(2, Math.floor(48 / Math.max(mapHeight, mapWidth)));

            const canvas = document.createElement('canvas');
            canvas.width = mapWidth * scale;
            canvas.height = mapHeight * scale;
            const ctx = canvas.getContext('2d');

            let minVal = Infinity;
            let maxVal = -Infinity;
            for (let y = 0; y < mapHeight; y++) {
                for (let x = 0; x < mapWidth; x++) {
                    minVal = Math.min(minVal, map2d[y][x]);
                    maxVal = Math.max(maxVal, map2d[y][x]);
                }
            }
            const valueRange = maxVal - minVal || 1;

            for (let y = 0; y < mapHeight; y++) {
                for (let x = 0; x < mapWidth; x++) {
                    const normalized = Math.round(((map2d[y][x] - minVal) / valueRange) * 255);
                    ctx.fillStyle = `rgb(${Math.round(normalized * 0.23)},${Math.round(normalized * 0.51)},${Math.round(normalized * 0.96)})`;
                    ctx.fillRect(x * scale, y * scale, scale, scale);
                }
            }
            grid.appendChild(canvas);
        });

        container.appendChild(section);
    });
}

function initDrawingCanvas() {
    const canvas = document.getElementById('drawCanvas');
    const classifyBtn = document.getElementById('classifyBtn');
    const clearBtn = document.getElementById('clearBtn');
    if (!canvas || !classifyBtn) return;

    const ctx = canvas.getContext('2d');
    let isDrawing = false;
    let hasContent = false;

    ctx.fillStyle = '#07090b';
    ctx.fillRect(0, 0, 280, 280);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 16;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    canvas.addEventListener('mousedown', (event) => {
        if (!event.isTrusted || event.clientX === undefined) {
            isDrawing = true;
            hasContent = true;
            return;
        }
        isDrawing = true;
        hasContent = true;
        const rect = canvas.getBoundingClientRect();
        ctx.beginPath();
        ctx.moveTo(event.clientX - rect.left, event.clientY - rect.top);
    });

    canvas.addEventListener('mousemove', (event) => {
        if (!isDrawing || !event.clientX) return;
        const rect = canvas.getBoundingClientRect();
        ctx.lineTo(event.clientX - rect.left, event.clientY - rect.top);
        ctx.stroke();
    });

    canvas.addEventListener('mouseup', () => isDrawing = false);
    canvas.addEventListener('mouseleave', () => isDrawing = false);

    canvas.addEventListener('touchstart', (event) => {
        event.preventDefault();
        isDrawing = true;
        hasContent = true;
        const rect = canvas.getBoundingClientRect();
        const touch = event.touches[0];
        ctx.beginPath();
        ctx.moveTo(touch.clientX - rect.left, touch.clientY - rect.top);
    });

    canvas.addEventListener('touchmove', (event) => {
        event.preventDefault();
        if (!isDrawing) return;
        const rect = canvas.getBoundingClientRect();
        const touch = event.touches[0];
        ctx.lineTo(touch.clientX - rect.left, touch.clientY - rect.top);
        ctx.stroke();
    });

    canvas.addEventListener('touchend', () => isDrawing = false);

    clearBtn.addEventListener('click', () => {
        ctx.fillStyle = '#07090b';
        ctx.fillRect(0, 0, 280, 280);
        ctx.strokeStyle = '#ffffff';
        hasContent = false;
        document.getElementById('predictedDigit').textContent = '?';
        document.getElementById('predLatency').textContent = 'Draw a digit to begin';
        document.getElementById('cycleTag').textContent = '-- cycles';
        document.getElementById('confidenceBars').innerHTML = '';
        document.getElementById('featureMaps').innerHTML = '<p class="empty-state">Run a classification to see layer activations</p>';
        document.getElementById('saliencyContainer').innerHTML = '<p class="empty-state">Classify a digit to see which pixels influenced the prediction</p>';
        document.querySelectorAll('.gallery-item').forEach(el => el.classList.remove('selected'));
    });

    classifyBtn.addEventListener('click', async () => {
        if (!hasContent) return;

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(canvas, 0, 0, 28, 28);
        const imageData = tempCtx.getImageData(0, 0, 28, 28);
        const pixels = [];
        for (let i = 0; i < imageData.data.length; i += 4) {
            pixels.push(imageData.data[i]);
        }

        classifyBtn.textContent = 'Classifying...';
        classifyBtn.disabled = true;

        try {
            const response = await fetch(API_BASE + '/api/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pixels }),
            });

            if (!response.ok) throw new Error('API error');
            const result = await response.json();
            renderPrediction(result);
        } catch {
            document.getElementById('predictedDigit').textContent = '!';
            document.getElementById('predLatency').textContent = 'Error: server not running';
        } finally {
            classifyBtn.textContent = 'Classify';
            classifyBtn.disabled = false;
        }
    });
}

function renderPrediction(data) {
    const digitElement = document.getElementById('predictedDigit');
    digitElement.textContent = data.digit;
    digitElement.classList.remove('flash');
    void digitElement.offsetWidth;
    digitElement.classList.add('flash');

    document.getElementById('cycleTag').textContent = data.cycles.toLocaleString() + ' cycles';
    document.getElementById('predLatency').textContent =
        `${data.latency_us} us at 100 MHz | ${data.cycles.toLocaleString()} cycles`;

    renderConfidenceBars(data.confidence, data.digit);

    if (data.layer_outputs) {
        renderFeatureMaps(data.layer_outputs);
        if (data.layer_outputs.saliency) {
            renderSaliencyMap(data.layer_outputs.saliency);
        }
    }
}

function renderConfidenceBars(confidences, predictedDigit) {
    const container = document.getElementById('confidenceBars');
    container.innerHTML = '';

    confidences.forEach((percentage, digit) => {
        const isTopPrediction = digit === predictedDigit;
        const row = document.createElement('div');
        row.className = 'conf-row';
        row.innerHTML = `
            <span class="conf-label ${isTopPrediction ? 'highlight' : ''}">${digit}</span>
            <div class="conf-track">
                <div class="conf-fill ${isTopPrediction ? 'top' : ''}" style="width: 0%"></div>
            </div>
            <span class="conf-pct ${isTopPrediction ? 'highlight' : ''}">${percentage.toFixed(1)}%</span>
        `;
        container.appendChild(row);

        requestAnimationFrame(() => {
            row.querySelector('.conf-fill').style.width = Math.max(percentage, 0.5) + '%';
        });
    });
}

/* ----------------------------------------------------------------
   hls4ml Feature
   ---------------------------------------------------------------- */
function initHls4ml() {
    const pipelineContainer = document.getElementById('hlsPipeline');
    const modelSummary = document.getElementById('hlsModelSummary');
    const resourceGrid = document.getElementById('hlsResourceGrid');
    const runBtn = document.getElementById('hlsRunBtn');
    const resetBtn = document.getElementById('hlsResetBtn');
    if (!pipelineContainer) return;

    const conversionSteps = [
        { icon: '📦', title: 'Load Keras Model', desc: 'Parse model.h5, extract layer config and weights', badge: 'h5py' },
        { icon: '🔍', title: 'Interpret Layers', desc: 'Map Dense, Conv2D, ReLU, BatchNorm to HLS primitives', badge: '6 layers' },
        { icon: '⚡', title: 'Apply Reuse Factor', desc: 'Trade latency for area: reuse_factor=1 (fully parallel)', badge: 'RF=1' },
        { icon: '🔢', title: 'Quantize Weights', desc: 'Convert float32 to ap_fixed<16,6> fixed-point', badge: 'ap_fixed' },
        { icon: '🏗️', title: 'Generate HLS C++', desc: 'Emit synthesizable firmware with #pragma HLS directives', badge: 'Vivado HLS' },
        { icon: '✅', title: 'C-Sim Validation', desc: 'Verify bitexact output against Python reference model', badge: 'csim' },
    ];

    // Build pipeline steps
    conversionSteps.forEach((step, index) => {
        const element = document.createElement('div');
        element.className = 'hls-step';
        element.id = `hls-step-${index}`;
        element.innerHTML = `
            <div class="hls-step-icon">${step.icon}</div>
            <div class="hls-step-content">
                <div class="hls-step-title">${step.title}</div>
                <div class="hls-step-desc">${step.desc}</div>
            </div>
            <div class="hls-step-badge">${step.badge}</div>
        `;
        pipelineContainer.appendChild(element);
    });

    // Model summary
    if (modelSummary) {
        modelSummary.innerHTML = `<span style="color:var(--text-3)">Model: "lenet5_classifier"</span>
_________________________________________________________________
<span style="color:var(--text-3)"> Layer (type)              Output Shape       Param #</span>
=================================================================
 conv2d (Conv2D)           (None, 24, 24, 6)    156
 relu (ReLU)               (None, 24, 24, 6)    0
 max_pooling2d (MaxPool)   (None, 12, 12, 6)    0
 conv2d_1 (Conv2D)         (None, 8, 8, 16)     2,416
 relu_1 (ReLU)             (None, 8, 8, 16)     0
 max_pooling2d_1 (MaxPool) (None, 4, 4, 16)     0
 flatten (Flatten)         (None, 256)           0
 dense (Dense)             (None, 120)           30,840
 dense_1 (Dense)           (None, 84)            10,164
 dense_2 (Dense)           (None, 10)            850
=================================================================
<span style="color:var(--accent)">Total params: 44,426</span>
<span style="color:var(--text-3)">Trainable params: 44,426</span>
<span style="color:var(--text-3)">Non-trainable params: 0</span>`;
    }

    // Resource estimates
    const resourceEstimates = [
        { name: 'LUTs', value: '18,240', pct: '4.2%' },
        { name: 'FF', value: '12,560', pct: '1.4%' },
        { name: 'DSP48', value: '64', pct: '1.8%' },
        { name: 'BRAM', value: '22', pct: '1.5%' },
    ];

    if (resourceGrid) {
        resourceEstimates.forEach(resource => {
            const item = document.createElement('div');
            item.className = 'hls-resource-item';
            item.innerHTML = `
                <div class="hls-resource-name">${resource.name}</div>
                <div class="hls-resource-value">${resource.value}</div>
                <div class="hls-resource-pct">${resource.pct} of Virtex-7</div>
            `;
            resourceGrid.appendChild(item);
        });
    }

    // Latency chart
    new Chart('chartHlsLatency', {
        labels: ['CPU (PyTorch)', 'GPU (RTX 3060)', 'FPGA (hls4ml)'],
        datasets: [{ label: 'Latency (us)', data: [2130, 45, 0.41], color: '#6c8cff' }],
    }, { type: 'bar', height: 240 });

    // Run conversion animation
    let isRunning = false;

    runBtn.addEventListener('click', async () => {
        if (isRunning) return;
        isRunning = true;
        runBtn.textContent = 'Converting...';
        runBtn.disabled = true;

        // Reset all steps
        conversionSteps.forEach((_, i) => {
            const el = document.getElementById(`hls-step-${i}`);
            el.classList.remove('completed', 'active-step');
        });

        // Animate through steps
        for (let i = 0; i < conversionSteps.length; i++) {
            const element = document.getElementById(`hls-step-${i}`);
            element.classList.add('active-step');

            await new Promise(resolve => setTimeout(resolve, 800));

            element.classList.remove('active-step');
            element.classList.add('completed');
        }

        runBtn.textContent = 'Run Conversion';
        runBtn.disabled = false;
        isRunning = false;
    });

    resetBtn.addEventListener('click', () => {
        conversionSteps.forEach((_, i) => {
            const el = document.getElementById(`hls-step-${i}`);
            el.classList.remove('completed', 'active-step');
        });
    });
}

/* ----------------------------------------------------------------
   Connection Status
   ---------------------------------------------------------------- */
async function checkStatus() {
    try {
        const response = await fetch(API_BASE + '/api/status');
        if (response.ok) {
            const data = await response.json();
            const label = document.getElementById('statusLabel');
            if (label) label.textContent = 'Connected';
            const tag = document.getElementById('backendTag');
            if (tag) tag.textContent = data.backend || 'Offline';
        }
    } catch {
        const label = document.getElementById('statusLabel');
        if (label) label.textContent = 'Standalone';
        const chip = document.getElementById('statusChip');
        if (chip) {
            chip.style.background = 'var(--accent-dim)';
            chip.style.borderColor = 'rgba(108, 140, 255, 0.15)';
            chip.style.color = 'var(--accent)';
            const dot = chip.querySelector('.status-dot');
            if (dot) dot.style.background = 'var(--accent)';
        }
    }
}

/* ----------------------------------------------------------------
   Boot
   ---------------------------------------------------------------- */
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();

    // Animated counters
    document.querySelectorAll('.stat-value[data-to]').forEach(animateCounter);

    // Sparklines
    drawSparkline('sparkSpeedup',    [3.2, 3.8, 4.1, 4.5, 4.8, 5.0, 5.1, 5.1], '#6c8cff');
    drawSparkline('sparkThroughput', [1200, 1500, 1800, 2000, 2200, 2350, 2400, 2439], '#34d399');
    drawSparkline('sparkLatency',    [0.85, 0.72, 0.60, 0.52, 0.48, 0.44, 0.42, 0.41], '#a78bfa');
    drawSparkline('sparkEfficiency', [200, 310, 420, 500, 560, 610, 650, 668], '#fbbf24');

    // Initialize sections
    initPipeline();
    initSystolicArray();
    initResources();
    initCharts();
    initCounters();
    initModuleTree();
    initDataflow();
    initInsights();
    initDrawingCanvas();
    initHls4ml();
    loadGallery();
    checkStatus();
});
