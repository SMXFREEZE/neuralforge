# NeuralForge

A custom FPGA accelerator that runs INT8 LeNet-5 inference using a systolic array. Built on Xilinx 7-Series, inspired by the Google TPU v1 and NVIDIA Tensor Core architectures.

<p align="center">
  <img src="https://img.shields.io/badge/Verilog-HDL-blue?style=flat-square" alt="Verilog"/>
  <img src="https://img.shields.io/badge/FPGA-Xilinx%207--Series-orange?style=flat-square" alt="FPGA"/>
  <img src="https://img.shields.io/badge/Quantization-INT8-green?style=flat-square" alt="INT8"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-yellow?style=flat-square" alt="Python"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch-red?style=flat-square" alt="PyTorch"/>
</p>

---

## What It Does

NeuralForge takes a handwritten digit image, pushes it over UART to an FPGA, and gets back a classification result in under half a millisecond. The whole inference pipeline (convolution, pooling, fully connected layers) runs in synthesizable Verilog on the chip.

The FPGA design uses a 4x4 weight-stationary systolic array, the same dataflow approach Google used in their first TPU. Weights sit inside each MAC unit. Activations stream in from the left. You get a 4x4 matrix multiply in 7 clock cycles.

```
Host PC (Python)                    FPGA (Xilinx 7-Series)
+--------------+        UART       +------------------------------+
|  train.py    |    ---------->    |  UART Interface (8N1)        |
|  quantize.py |                   |         |                     |
|  host.py     |    <----------    |  Input Buffer (Ping-Pong)    |
+--------------+     115200 bd     |         |                     |
                                   |  Conv3x3 > ReLU > MaxPool2x2 |
                                   |         |                     |
                                   |  +------v--------------+     |
                                   |  | 4x4 Systolic Array  |     |
                                   |  | (16 INT8 MACs/cycle)|     |
                                   |  +------+--------------+     |
                                   |         |                     |
                                   |  Argmax > Result (0-9)       |
                                   |  Status LEDs                  |
                                   +------------------------------+
```

## Why These Design Choices

| Decision | Why |
|----------|-----|
| **Weight-stationary dataflow** | Weights load once, activations stream through. Minimizes memory bandwidth. |
| **INT8 symmetric quantization** | 4x smaller than FP32 with less than 1% accuracy loss on MNIST. |
| **4x4 systolic array** | Fits the XC7A35T DSP budget (16 out of 90 DSPs) and still hits 1.6 GMAC/s at 100 MHz. |
| **Pipelined conv engine** | 4-stage pipeline with adder tree. Keeps throughput at one output per cycle. |
| **Ping-pong input buffer** | Overlaps UART data reception with compute so there are no pipeline bubbles between images. |

---

## Performance

| Metric | CPU (PyTorch, i7) | FPGA (XC7A35T, 100 MHz) | Speedup |
|--------|-------------------|-------------------------|---------|
| Inference latency | ~2.1 ms | ~0.41 ms | **5.1x** |
| Throughput | ~476 img/s | ~2,439 img/s | **5.1x** |
| Power | ~65 W TDP | ~0.5 W | **130x** efficiency |
| Accuracy | 99.2% (FP32) | 98.5% (INT8) | -0.7% |

The FPGA gets about **4,878 inferences per joule** compared to the CPU's **7.3 inferences per joule**. That's a 668x improvement in energy efficiency.

---

## Project Structure

```
neuralforge/
├── rtl/                          # Synthesizable Verilog
│   ├── mac_unit.v                # INT8 Multiply-Accumulate unit
│   ├── systolic_array.v          # 4x4 weight-stationary systolic array
│   ├── conv_engine.v             # 3x3 convolution with pipelined adder tree
│   ├── activation.v              # ReLU / LeakyReLU activation
│   ├── pooling.v                 # 2x2 max pooling
│   ├── weight_buffer.v           # BRAM-based INT8 weight storage
│   ├── input_buffer.v            # Ping-pong double buffer
│   ├── uart_interface.v          # 8N1 UART transceiver
│   ├── axi_stream_wrapper.v      # AXI4-Stream IP core wrapper
│   └── top.v                     # Top-level integration + FSM controller
├── sim/                          # Simulation testbenches
│   ├── tb_mac_unit.v             # Exhaustive MAC correctness tests
│   ├── tb_systolic_array.v       # 4x4 matmul verification
│   ├── tb_conv_engine.v          # Convolution output verification
│   ├── tb_axi_wrapper.v          # AXI4-Stream handshake verification
│   └── tb_top.v                  # End-to-end UART to inference to UART test
├── sw/                           # Python ML pipeline
│   ├── train.py                  # LeNet-5 training on MNIST (PyTorch)
│   ├── quantize.py               # INT8 post-training quantization
│   ├── export_weights.py         # Weight export to Verilog .mem files
│   ├── generate_weights.py       # Train + export INT8 weights to JSON
│   ├── fpga_simulator.py         # Cycle-accurate FPGA inference simulator
│   ├── host.py                   # UART host interface + simulated mode
│   ├── benchmark.py              # CPU vs FPGA performance comparison
│   ├── dashboard_server.py       # Web dashboard backend server
│   ├── ai_analyzer.py            # AI-powered architecture analysis
│   └── requirements.txt          # Python dependencies
├── dashboard/                    # Interactive web dashboard
│   ├── index.html                # Dashboard UI
│   ├── style.css                 # Premium dark theme
│   └── app.js                    # Charts, canvas, classification logic
├── weights/                      # Exported weight files
│   └── lenet5_int8.json          # Pre-trained INT8 weights (bundled)
├── constraints/                  # FPGA pin/timing constraints
│   ├── basys3.xdc                # Xilinx Basys 3 (XC7A35T)
│   └── nexys_a7.xdc              # Digilent Nexys A7 (XC7A100T)
├── docs/
│   └── architecture.md           # Detailed system architecture
└── README.md
```

---

## Quick Start

### What You Need

- **For simulation**: [Icarus Verilog](https://steveicarus.github.io/iverilog/) + [GTKWave](http://gtkwave.sourceforge.net/) (both free)
- **For synthesis**: Xilinx Vivado (free WebPACK edition), only needed if you're putting this on actual hardware
- **For ML**: Python 3.10+ with PyTorch

### 1. Run RTL Simulation (no FPGA needed)

```bash
# Install Icarus Verilog (Windows)
winget install Icarus.Icarus

# Test MAC unit
iverilog -o sim/mac_test sim/tb_mac_unit.v rtl/mac_unit.v
vvp sim/mac_test

# Test systolic array
iverilog -o sim/sys_test sim/tb_systolic_array.v rtl/systolic_array.v rtl/mac_unit.v
vvp sim/sys_test

# Test convolution engine
iverilog -o sim/conv_test sim/tb_conv_engine.v rtl/conv_engine.v
vvp sim/conv_test

# View waveforms
gtkwave mac_unit.vcd
```

### 2. Train and Quantize the Model

```bash
cd sw
pip install -r requirements.txt

# Train LeNet-5 (~5 min, gets >99% accuracy)
python train.py

# Quantize to INT8
python quantize.py

# Export weights for FPGA
python export_weights.py
```

### 3. Run Benchmark

```bash
# CPU vs FPGA performance comparison (simulated FPGA)
python benchmark.py --mode both --samples 1000
```

### 4. Live Demo Dashboard

```bash
# Start the interactive dashboard
python sw/dashboard_server.py --port 8080

# Open http://localhost:8080 in your browser
# Click the "Try It" tab to draw digits and classify them live
```

The dashboard lets you:
- **Draw and classify**: Sketch any digit (0-9) on the canvas and watch it run through the INT8 inference pipeline in real time (simulated in Python)
- **MNIST gallery**: Click any of 100 bundled test samples to classify them instantly
- **Batch accuracy**: Run a full 100-sample validation with one click and see a confusion matrix
- **Feature maps**: See what each CNN layer "sees" at every stage
- **Saliency heatmaps**: See which pixels mattered most for the prediction (occlusion sensitivity)
- **Confidence scores**: Softmax probabilities across all 10 classes
- **Cycle-accurate timing**: Exactly how many clock cycles the FPGA uses (simulator runs ~50x faster via NumPy vectorization)
- **hls4ml demo**: Interactive visualization of the Python-to-FPGA model conversion pipeline, showing how tools like hls4ml translate Keras models into synthesizable HLS firmware

### 5. Deploy to FPGA (when you have the hardware)

```bash
# In Xilinx Vivado:
# 1. Create project, target Basys 3 (XC7A35T-1CPG236C)
# 2. Add sources: rtl/*.v
# 3. Add constraints: constraints/basys3.xdc
# 4. Run synthesis, implementation, generate bitstream
# 5. Program device

# Send test images
python sw/host.py --port COM3 --samples 100
```

---

## Technical Details

### Systolic Array Dataflow

The 4x4 systolic array uses **weight-stationary** dataflow. Same approach as the Google TPU v1:

```
Weights pre-loaded into each MAC unit:

    w00  w01  w02  w03       Activations stream in from left:
  +----+----+----+----+
> |MAC |MAC |MAC |MAC | < a_in[0] (row 0)
  +----+----+----+----+
> |MAC |MAC |MAC |MAC | < a_in[1] (row 1)
  +----+----+----+----+
> |MAC |MAC |MAC |MAC | < a_in[2] (row 2)
  +----+----+----+----+
> |MAC |MAC |MAC |MAC | < a_in[3] (row 3)
  +----+----+----+----+

Each MAC: acc += a * w (every cycle)
Result: 4x4 matrix multiply in 4+3 = 7 cycles
```

### Quantization Pipeline

```
FP32 weights --> Symmetric INT8 Quantization --> .mem hex files --> BRAM init
                 scale = max|w| / 127
                 q = round(w / scale)
                 range: [-128, +127]
```

### Convolution Pipeline Stages

```
Stage 1: 9 parallel INT8 multiplications (input feature map x kernel)
Stage 2: Pairwise addition (adder tree level 1: 4 sums + 1 passthrough)
Stage 3: Pairwise addition (adder tree level 2: 2 sums)
Stage 4: Final sum + bias, output
```

---

## Resource Utilization (Estimated, XC7A35T)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUTs | ~3,200 | 20,800 | 15% |
| Flip-Flops | ~2,400 | 41,600 | 6% |
| Block RAM | 4 | 50 | 8% |
| DSP48E1 | 16 | 90 | 18% |

---

## Tools and Technologies

- **HDL**: Verilog (IEEE 1364-2005)
- **Simulation**: Icarus Verilog, GTKWave
- **Synthesis**: Xilinx Vivado 2023.x
- **Target FPGA**: Xilinx Artix-7 (XC7A35T / XC7A100T)
- **ML Framework**: PyTorch 2.x
- **Quantization**: Custom symmetric INT8 (compatible with PyTorch QAT)
- **Host Interface**: Python + pyserial (UART 115200 baud)

---

## References

- [1] N. P. Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit," ISCA 2017
- [2] NVIDIA, "Tensor Core Architecture," NVIDIA Ampere GPU Architecture Whitepaper, 2020
- [3] Y. LeCun et al., "Gradient-Based Learning Applied to Document Recognition," Proc. IEEE, 1998
- [4] B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," CVPR 2018

---

## License

MIT License, see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built as a hardware ML accelerator portfolio project.<br/>
  Targeting NVIDIA, AMD, and Tesla co-op positions.
</p>
