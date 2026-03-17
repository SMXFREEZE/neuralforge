# NeuralForge — System Architecture

## Overview

NeuralForge is a custom FPGA-based inference accelerator for the LeNet-5 CNN,
designed to classify 28×28 MNIST handwritten digits in hardware. The architecture
is directly inspired by **Google's TPU v1** (weight-stationary systolic array)
and **NVIDIA Tensor Cores** (fused multiply-accumulate).

## Architecture Block Diagram

```
┌───────────────────────────────────────────────────────────────────┐
│                        HOST PC (Python)                          │
│   ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────┐   │
│   │ train.py │  │quantize  │  │export_wts │  │  host.py     │   │
│   │          │──│   .py    │──│   .py      │  │  (UART I/O)  │   │
│   └──────────┘  └──────────┘  └───────────┘  └──────┬───────┘   │
└──────────────────────────────────────────────────────┼───────────┘
                                                       │ UART 115200
┌──────────────────────────────────────────────────────┼───────────┐
│                     FPGA (Xilinx 7-Series)           │           │
│                                                      ▼           │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │     Dual Interfaces (UART vs AMBA AXI4-Stream)           │    │
│  │                                                          │    │
│  │  [Standalone Mode]            [SoC IP Mode]              │    │
│  │  top.v                        axi_stream_wrapper.v       │    │
│  │  UART RX/TX                   S_AXIS / M_AXIS            │    │
│  └────────────────────────┬─────────────────────────────────┘    │
│                           │                                      │
│  ┌────────────────────────▼─────────────────────────────────┐    │
│  │              Input Buffer (Ping-Pong)                    │    │
│  │         (Double-buffered, 784 bytes each)                │    │
│  └────────────────────────┬─────────────────────────────────┘    │
│                           │                                      │
│  ┌────────────────────────▼─────────────────────────────────┐    │
│  │                 CNN Inference Pipeline                   │    │
│  │                                                          │    │
│  │  ┌─────────┐  ┌──────┐  ┌─────────┐  ┌──────┐          │    │
│  │  │ Conv1   │─▶│ ReLU │─▶│MaxPool  │─▶│ Conv2│──▶...    │    │
│  │  │ (3x3)   │  │      │  │ (2x2)   │  │(3x3) │           │    │
│  │  └─────────┘  └──────┘  └─────────┘  └──────┘           │    │
│  │                                                           │    │
│  │  ┌───────────────────────────────────────────────┐        │    │
│  │  │          4×4 Systolic Array                    │        │    │
│  │  │    ┌───┐ ┌───┐ ┌───┐ ┌───┐                   │        │    │
│  │  │ ──▶│MAC│─│MAC│─│MAC│─│MAC│   (Row 0)         │        │    │
│  │  │    └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘                   │        │    │
│  │  │ ──▶│MAC│─│MAC│─│MAC│─│MAC│   (Row 1)         │        │    │
│  │  │    └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘                   │        │    │
│  │  │ ──▶│MAC│─│MAC│─│MAC│─│MAC│   (Row 2)         │        │    │
│  │  │    └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘                   │        │    │
│  │  │ ──▶│MAC│─│MAC│─│MAC│─│MAC│   (Row 3)         │        │    │
│  │  │    └───┘ └───┘ └───┘ └───┘                    │        │    │
│  │  │    Activations flow →  Weights pre-loaded     │        │    │
│  │  └───────────────────────────────────────────────┘        │    │
│  │                                                           │    │
│  │  ┌─────────────────────┐    ┌────────────────────┐        │    │
│  │  │   Weight Buffer     │    │   Argmax + Output   │        │    │
│  │  │   (BRAM, INT8)      │    │   (Classification)  │        │    │
│  │  └─────────────────────┘    └────────────────────┘        │    │
│  └───────────────────────────────────────────────────────────┘    │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │              Control FSM (top.v)                           │    │
│  │    IDLE → RECV_IMAGE → COMPUTE → ARGMAX → SEND_RESULT    │    │
│  └───────────────────────────────────────────────────────────┘    │
│                                                                   │
│  Status LEDs: [DIGIT 3:0] [BUSY] [DONE] [ERROR]                 │
└───────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Host → FPGA**: Python sends 784 bytes (28×28 image) over UART
2. **Input Buffer**: Ping-pong buffer receives while compute runs on previous image
3. **Conv Engine**: 3×3 convolution with pipelined adder tree (4-stage pipeline)
4. **Activation**: ReLU via sign-bit masking (zero latency)
5. **Pooling**: 2×2 max pooling with comparator tree (2-stage pipeline)
6. **Systolic Array**: 4×4 weight-stationary array for fully-connected layers (16 MACs/cycle)
7. **Argmax**: Finds maximum class score → predicted digit
8. **FPGA → Host**: Sends 1-byte result (predicted class 0-9) via UART

## Quantization Strategy

| Parameter | Value |
|-----------|-------|
| Precision | INT8 (signed, symmetric) |
| Range | [-128, +127] |
| Scheme | Symmetric: scale = max_abs / 127 |
| Zero Point | 0 (symmetric) |
| Accumulator | 32-bit (prevents overflow in MAC) |

## Performance Model

| Metric | Value |
|--------|-------|
| Clock frequency | 100 MHz |
| Systolic array MACs/cycle | 16 |
| Peak throughput | 1.6 GMAC/s |
| LeNet-5 total MACs | ~416,000 |
| Estimated inference time | ~0.41 ms |
| Estimated throughput | ~2,400 images/sec |

## Module Hierarchy

```
NeuralForge/
├── [Top-level implementations]
│   ├── top.v                 — Standalone top (UART 8N1)
│   └── axi_stream_wrapper.v  — IP Core wrapper (AMBA AXI4-Stream)
│
├── [Peripherals & I/O]
│   ├── uart_interface.v      — UART transceiver
│   ├── input_buffer.v        — Ping-pong double buffer
│   └── weight_buffer.v       — BRAM weight storage
│
├── [Compute Engine]
│   ├── layer_controller.v    — Layer-by-layer master FSM
│   ├── conv_engine.v         — 3×3 convolution pipeline
│   ├── activation.v          — ReLU / LeakyReLU masking
│   ├── pooling.v             — 2×2 max pooling tree
│   └── systolic_array.v      — 4×4 weight-stationary array
│       └── mac_unit.v        — INT8 MAC (×16 instances)
```

## FPGA Resource Estimates (XC7A35T)

| Resource | Used (est.) | Available | Utilization |
|----------|-------------|-----------|-------------|
| LUTs | ~3,200 | 20,800 | ~15% |
| FFs | ~2,400 | 41,600 | ~6% |
| BRAM | 4 | 50 | 8% |
| DSPs | 16 | 90 | 18% |
