import { NextRequest } from 'next/server'
import { streamText } from 'ai'
import { openai } from '@ai-sdk/openai'

const SYSTEM_PROMPT = `You are NeuralForge AI, an expert embedded systems and FPGA engineer specializing in neural network hardware acceleration. You analyze inference results, performance metrics, and hardware architectures.

Hardware context:
- FPGA: Xilinx XC7A35T Artix-7, 100 MHz
- Network: LeNet-5 on MNIST (10 classes, 28×28 grayscale)
- Architecture: 4×4 weight-stationary systolic array (16 MACs/cycle, 1.6 GMAC/s peak)
- Quantization: INT8 symmetric, per-tensor scale, 98.5% accuracy (−0.7% vs FP32)
- Resources: 16/90 DSP48E1, 3370/20800 LUTs, 4/50 BRAM
- Interface: UART 115200 baud, AXI4-Stream alternative
- Latency: 0.41 ms FPGA vs 2.13 ms CPU (5.1× speedup)
- Energy: 0.5W FPGA vs 65W CPU (668× efficiency)
- Throughput: 2,439 img/s

Be technically precise. Use tables and markdown for clarity.`

function getPrompt(mode: string, question: string): string {
  switch (mode) {
    case 'performance':
      return `Analyze the NeuralForge FPGA accelerator performance data and provide:
1. **Performance Summary**: Key metrics comparison (FPGA vs CPU vs GPU)
2. **Bottleneck Analysis**: Identify the primary performance bottleneck
3. **Top 3 Optimizations**: Actionable improvements with expected speedup
4. **Energy Efficiency**: Inferences/joule comparison
5. **Scalability**: How performance would scale with an 8×8 systolic array`

    case 'quantization':
      return `Analyze the INT8 quantization results:
1. **Accuracy Impact**: The −0.7% drop from FP32 to INT8 — is it acceptable?
2. **Per-Layer Analysis**: Which layers likely have highest quantization error and why
3. **Compression Ratio**: Memory savings analysis
4. **Improvement Suggestions**: Per-channel vs per-tensor, QAT, mixed precision options
5. **Industry Comparison**: How does this compare to TensorRT and ONNX Runtime quality`

    case 'architecture':
      return `As an FPGA hardware architect, analyze the NeuralForge systolic array design:
1. **Design Review**: Evaluate the 4×4 weight-stationary dataflow choice
2. **Resource Utilization**: Is the XC7A35T well-utilized? Where is slack?
3. **Timing Analysis**: Can this design hit 100 MHz? What are timing-critical paths?
4. **Memory Hierarchy**: Is BRAM allocation optimal?
5. **Industry Comparison**: How does this compare to Google TPU v1 and NVIDIA Tensor Cores?
6. **v2.0 Design**: What would the next generation look like?`

    case 'question':
      return question || 'Provide a general overview of the NeuralForge architecture.'

    default:
      return 'Provide a general technical overview of the NeuralForge FPGA accelerator.'
  }
}

export async function POST(req: NextRequest) {
  const { mode = 'performance', question = '' } = await req.json()

  if (!process.env.OPENAI_API_KEY) {
    // Return offline analysis as a stream
    const offline = getOfflineAnalysis(mode)
    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ text: offline })}\n\n`))
        controller.enqueue(encoder.encode('data: [DONE]\n\n'))
        controller.close()
      },
    })
    return new Response(stream, {
      headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' },
    })
  }

  const result = streamText({
    model: openai('gpt-4o'),
    system: SYSTEM_PROMPT,
    prompt: getPrompt(mode, question),
    maxTokens: 1500,
    temperature: 0.3,
  })

  // Convert AI SDK stream to SSE format the frontend expects: data: {"text":"..."}
  const encoder = new TextEncoder()
  const readable = new ReadableStream({
    async start(controller) {
      try {
        for await (const delta of (await result).textStream) {
          if (delta) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ text: delta })}\n\n`))
          }
        }
      } finally {
        controller.enqueue(encoder.encode('data: [DONE]\n\n'))
        controller.close()
      }
    },
  })

  return new Response(readable, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    },
  })
}

function getOfflineAnalysis(mode: string): string {
  const analyses: Record<string, string> = {
    performance: `## Performance Analysis\n\n**5.1× speedup** over CPU baseline.\n\nNeuralForge delivers **0.41 ms** mean inference latency vs **2.13 ms** on an Intel i7.\n\n### Bottleneck\nPrimary bottleneck: **UART interface** at 115200 baud. Image transfer ~54 ms while compute takes 0.41 ms — 99.2% of wall-clock is I/O-bound.\n\n### Top 3 Optimizations\n1. **SPI interface** at 10 MHz → 100× I/O speedup\n2. **8×8 systolic array** → 64 MACs/cycle, 4× compute\n3. **Sparse weight support** → 2× memory bandwidth savings\n\n### Energy Efficiency\nFPGA at 0.5W: **4,878 inf/J** vs CPU at 65W: **7.3 inf/J** = **668× improvement**\n\n*Note: Set OPENAI_API_KEY for full AI-powered analysis.*`,

    quantization: `## Quantization Quality\n\nINT8 symmetric quantization achieves **<1% accuracy drop** (99.2% → 98.5%) with **4× compression**.\n\n### Per-Layer Analysis\n- **conv1_weight** MSE 1.2e-5 (acceptable)\n- **fc3_weight** MSE 9.3e-6 (output layer most sensitive)\n\n### Recommendations\n1. **Per-channel quantization** for conv layers (+0.2% accuracy)\n2. **Quantization-aware training** recovers most accuracy gap\n3. Keep FC3 in **FP16** for better class discrimination\n\n*Note: Set OPENAI_API_KEY for full AI-powered analysis.*`,

    architecture: `## Architecture Review\n\nThe **4×4 weight-stationary systolic array** mirrors Google TPU v1 dataflow at smaller scale.\n\n### Resource Headroom\n| Resource | Used | Available | Remaining |\n|---|---|---|---|\n| DSPs | 16 | 90 | 74 (82%) |\n| LUTs | 3370 | 20800 | 85% free |\n| BRAM | 4 | 50 | 92% free |\n\n### v2.0 Recommendations\n1. Scale to 8×8 systolic array (64 DSPs, same device)\n2. Add output-stationary mode for conv layers\n3. Implement AXI-DMA for high-speed host transfer\n\n*Note: Set OPENAI_API_KEY for full AI-powered analysis.*`,
  }
  return analyses[mode] || analyses.performance
}
