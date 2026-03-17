"""
NeuralForge — AI-Powered Inference Analysis Engine
Uses Amazon Bedrock (Nova) to provide intelligent analysis of:
  - FPGA inference results and performance metrics
  - Quantization quality assessment
  - Architecture optimization suggestions
  - Natural language Q&A about the accelerator design
"""

import os
import sys
import json
import time
import base64
import argparse
import numpy as np
from io import BytesIO

try:
    import boto3
    from botocore.config import Config
    HAS_AWS = True
except ImportError:
    HAS_AWS = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


SYSTEM_PROMPT = """You are NeuralForge AI, an expert embedded systems and FPGA engineer specializing in neural network hardware acceleration. You analyze inference results, performance metrics, and hardware architectures.

Your role:
1. Analyze FPGA inference performance data (latency, throughput, utilization)
2. Compare INT8 quantized vs FP32 accuracy metrics
3. Suggest RTL optimizations for the systolic array architecture
4. Explain hardware architecture concepts in recruiter-friendly language
5. Generate insights about power efficiency and resource utilization

Always be technically precise. Reference specific metrics, cycle counts, and architecture details.
When suggesting optimizations, specify the expected improvement and implementation complexity.
Format responses in clean markdown with tables where appropriate."""


class BedrockEngine:
    """Amazon Bedrock (Nova) inference analysis engine."""

    def __init__(self, region="us-east-1", model_id="amazon.nova-pro-v1:0"):
        if not HAS_AWS:
            raise ImportError("boto3 not installed. Run: pip install boto3")

        self.config = Config(
            region_name=region,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        self.client = boto3.client('bedrock-runtime', config=self.config)
        self.model_id = model_id
        print(f"[AWS Bedrock] Initialized with model: {model_id}")

    def analyze(self, prompt, context=None, max_tokens=2048):
        """Send analysis request to Bedrock."""
        messages = []

        if context:
            messages.append({
                "role": "user",
                "content": [{"text": f"Context data:\n```json\n{json.dumps(context, indent=2)}\n```"}]
            })
            messages.append({
                "role": "assistant",
                "content": [{"text": "I've reviewed the data. What would you like me to analyze?"}]
            })

        messages.append({
            "role": "user",
            "content": [{"text": prompt}]
        })

        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=messages,
                system=[{"text": SYSTEM_PROMPT}],
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": 0.3,
                    "topP": 0.9
                }
            )

            output_text = response['output']['message']['content'][0]['text']
            usage = response.get('usage', {})
            print(f"[AWS Bedrock] Tokens: {usage.get('inputTokens', '?')} in, "
                  f"{usage.get('outputTokens', '?')} out")
            return output_text

        except Exception as e:
            print(f"[AWS Bedrock] Error: {e}")
            return f"Error communicating with Bedrock: {str(e)}"


class OpenAIEngine:
    """OpenAI fallback engine."""

    def __init__(self, model="gpt-4o-mini"):
        if not HAS_OPENAI:
            raise ImportError("openai not installed. Run: pip install openai")

        self.client = OpenAI()
        self.model = model
        print(f"[OpenAI] Initialized with model: {model}")

    def analyze(self, prompt, context=None, max_tokens=2048):
        """Send analysis request to OpenAI."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if context:
            messages.append({
                "role": "user",
                "content": f"Context data:\n```json\n{json.dumps(context, indent=2)}\n```"
            })
            messages.append({
                "role": "assistant",
                "content": "I've reviewed the data. What would you like me to analyze?"
            })

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            usage = response.usage
            print(f"[OpenAI] Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out")
            return response.choices[0].message.content

        except Exception as e:
            print(f"[OpenAI] Error: {e}")
            return f"Error communicating with OpenAI: {str(e)}"


class NeuralForgeAnalyzer:
    """
    High-level analysis engine that combines hardware data with AI insights.
    Supports both AWS Bedrock and OpenAI backends.
    """

    def __init__(self, backend="auto"):
        if backend == "auto":
            if os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE"):
                self.engine = BedrockEngine()
                self.backend_name = "AWS Bedrock (Nova)"
            elif os.environ.get("OPENAI_API_KEY"):
                self.engine = OpenAIEngine()
                self.backend_name = "OpenAI"
            else:
                print("[WARN] No API credentials found. Running in offline mode.")
                self.engine = None
                self.backend_name = "Offline"
        elif backend == "bedrock":
            self.engine = BedrockEngine()
            self.backend_name = "AWS Bedrock (Nova)"
        elif backend == "openai":
            self.engine = OpenAIEngine()
            self.backend_name = "OpenAI"
        else:
            self.engine = None
            self.backend_name = "Offline"

    def load_benchmark_data(self, path="results/benchmark_results.json"):
        """Load benchmark results if available."""
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def load_quantization_data(self, path="quantized/quantization_config.json"):
        """Load quantization config if available."""
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def load_inference_results(self, path="results/inference_results.json"):
        """Load inference results if available."""
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def _gather_context(self):
        """Gather all available data for context."""
        context = {
            "project": "NeuralForge FPGA Neural Network Accelerator",
            "architecture": {
                "network": "LeNet-5 (modified for MNIST)",
                "fpga_target": "Xilinx XC7A35T (Artix-7)",
                "clock_freq_mhz": 100,
                "systolic_array": "4x4 weight-stationary",
                "macs_per_cycle": 16,
                "peak_throughput_gmacs": 1.6,
                "quantization": "INT8 symmetric",
                "interface": "UART 115200 baud",
            },
            "resource_estimates": {
                "luts": 3200,
                "ffs": 2400,
                "bram_blocks": 4,
                "dsp48e1": 16,
            }
        }

        bench = self.load_benchmark_data()
        if bench:
            context["benchmark"] = bench

        quant = self.load_quantization_data()
        if quant:
            context["quantization"] = quant

        infer = self.load_inference_results()
        if infer:
            # Summarize, don't include all results
            context["inference_summary"] = {
                "accuracy": infer.get("accuracy"),
                "avg_latency_ms": infer.get("avg_latency_ms"),
                "throughput_ips": infer.get("throughput_ips"),
                "num_samples": infer.get("num_samples"),
            }

        return context

    def analyze_performance(self):
        """Generate comprehensive performance analysis."""
        context = self._gather_context()
        prompt = """Analyze the NeuralForge FPGA accelerator performance data and provide:

1. **Performance Summary**: Key metrics comparison (FPGA vs CPU)
2. **Bottleneck Analysis**: Identify the primary performance bottleneck
3. **Optimization Recommendations**: Top 3 actionable improvements with expected speedup
4. **Energy Efficiency**: Calculate and compare energy efficiency (inferences/joule)
5. **Scalability**: How would performance scale with an 8x8 systolic array?

Present findings in a clear, technical format with specific numbers."""

        if self.engine:
            return self.engine.analyze(prompt, context)
        return self._offline_performance_analysis(context)

    def analyze_quantization(self):
        """Analyze quantization quality and accuracy impact."""
        context = self._gather_context()
        prompt = """Analyze the INT8 quantization results for the LeNet-5 model:

1. **Accuracy Impact**: What is the accuracy drop from FP32 to INT8?
2. **Per-Layer Analysis**: Which layers have the highest quantization error (MSE)?
3. **Compression Ratio**: Calculate the actual memory savings
4. **Improvement Suggestions**: Should we use asymmetric quantization? Per-channel scales?
5. **Comparison**: How does our quantization quality compare to industry standards (TensorRT, ONNX Runtime)?

Be specific with numbers and recommendations."""

        if self.engine:
            return self.engine.analyze(prompt, context)
        return self._offline_quantization_analysis(context)

    def analyze_architecture(self):
        """Generate architecture analysis and optimization suggestions."""
        context = self._gather_context()
        prompt = """As an FPGA hardware architect, analyze the NeuralForge systolic array design:

1. **Design Review**: Evaluate the 4x4 weight-stationary dataflow choice
2. **Resource Utilization**: Is the FPGA (XC7A35T) well-utilized? Where is slack?
3. **Timing**: Can this design hit 100 MHz? What are potential timing-critical paths?
4. **Memory Hierarchy**: Is the BRAM allocation optimal? Should we add an L1 cache?
5. **Comparison to Industry**: How does this compare to Google TPU v1 and NVIDIA Tensor Cores?
6. **Next-Gen Recommendations**: What would a v2.0 design look like?

This analysis will be shown to NVIDIA and AMD recruiters, so be technically impressive."""

        if self.engine:
            return self.engine.analyze(prompt, context)
        return self._offline_architecture_analysis(context)

    def interactive_qa(self, question):
        """Answer technical questions about the design."""
        context = self._gather_context()
        if self.engine:
            return self.engine.analyze(question, context)
        return "Interactive Q&A requires an API backend (AWS Bedrock or OpenAI)."

    # ======================================================================
    # Offline analysis fallbacks (no API needed)
    # ======================================================================
    def _offline_performance_analysis(self, ctx):
        arch = ctx['architecture']
        return f"""# NeuralForge Performance Analysis (Offline Mode)

## Key Metrics
| Metric | CPU (PyTorch) | FPGA (NeuralForge) |
|--------|---------------|---------------------|
| Clock | ~3.5 GHz | {arch['clock_freq_mhz']} MHz |
| Parallelism | SIMD (8-wide) | {arch['macs_per_cycle']} MACs/cycle |
| Peak Throughput | ~28 GMAC/s | {arch['peak_throughput_gmacs']} GMAC/s |

## Bottleneck Analysis
- **Primary bottleneck**: UART interface (115200 baud = ~14.4 KB/s)
- **Image transfer time**: 784 bytes / 14.4 KB/s = ~54 ms (dominates total latency)
- **Compute time**: ~416K MACs / 1.6 GMAC/s = ~0.26 ms (negligible)
- **Recommendation**: Replace UART with SPI (10 MHz) or AXI for 100x+ improvement

## Top 3 Optimizations
1. **SPI Interface**: Replace UART with SPI at 10+ MHz → ~100x I/O speedup
2. **8x8 Systolic Array**: 4x compute throughput (64 MACs/cycle)
3. **Pipeline Staging**: Add FIFO between layers for full pipeline utilization

*Note: Connect AWS Bedrock or OpenAI API for detailed AI-powered analysis.*"""

    def _offline_quantization_analysis(self, ctx):
        return """# NeuralForge Quantization Analysis (Offline Mode)

## INT8 Symmetric Quantization
- **Scheme**: Symmetric (zero_point = 0), per-tensor scale
- **Expected accuracy drop**: <1% on MNIST (typical for LeNet-5)
- **Compression ratio**: 4× (FP32 → INT8)

## Recommendations
1. **Per-channel quantization**: Better accuracy for conv layers (different scale per filter)
2. **Quantization-aware training (QAT)**: Recovers most of the accuracy gap
3. **Mixed precision**: Keep FC3 (output layer) in FP16 for better class discrimination

*Note: Connect AWS Bedrock or OpenAI API for per-layer MSE analysis.*"""

    def _offline_architecture_analysis(self, ctx):
        return """# NeuralForge Architecture Analysis (Offline Mode)

## Design Review
- **Weight-stationary dataflow**: Good choice for FC layers (weights reused across batches)
- **4x4 array**: Conservative but fits XC7A35T budget (18% DSP utilization)
- **Ping-pong buffer**: Eliminates bubble cycles between images

## Resource Headroom
| Resource | Used | Available | Remaining |
|----------|------|-----------|-----------|
| DSPs | 16 | 90 | 74 (82%) |
| LUTs | 3200 | 20800 | 17600 (85%) |
| BRAM | 4 | 50 | 46 (92%) |

## v2.0 Recommendations
1. Scale to 8x8 systolic array (64 DSPs, still fits)
2. Add output-stationary mode for conv layers
3. Implement weight compression (sparse × dense)

*Note: Connect AWS Bedrock or OpenAI API for detailed TPU/Tensor Core comparison.*"""


def generate_analysis_report(analyzer, output_dir="results"):
    """Generate the complete AI-powered analysis report."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nUsing analysis backend: {analyzer.backend_name}")
    print("=" * 60)

    sections = {}

    print("\n📊 Analyzing performance...")
    sections['performance'] = analyzer.analyze_performance()

    print("\n🔬 Analyzing quantization...")
    sections['quantization'] = analyzer.analyze_quantization()

    print("\n🏗️ Analyzing architecture...")
    sections['architecture'] = analyzer.analyze_architecture()

    # Combine into full report
    report = f"""# NeuralForge AI Analysis Report
*Generated by {analyzer.backend_name}*
*{time.strftime('%Y-%m-%d %H:%M:%S')}*

---

## 📊 Performance Analysis

{sections['performance']}

---

## 🔬 Quantization Analysis

{sections['quantization']}

---

## 🏗️ Architecture Analysis

{sections['architecture']}

---

*This report was generated using AI-powered analysis ({analyzer.backend_name}).*
*NeuralForge — FPGA Neural Network Inference Accelerator*
"""

    report_path = os.path.join(output_dir, "ai_analysis_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n{'=' * 60}")
    print(f"📄 Full report saved to: {report_path}")
    print(f"{'=' * 60}")

    return report


def main():
    parser = argparse.ArgumentParser(description='NeuralForge AI Analysis Engine')
    parser.add_argument('--backend', choices=['auto', 'bedrock', 'openai', 'offline'],
                        default='auto', help='AI backend to use')
    parser.add_argument('--mode', choices=['report', 'performance', 'quantization',
                                           'architecture', 'interactive'],
                        default='report', help='Analysis mode')
    parser.add_argument('--question', type=str, default=None,
                        help='Question for interactive mode')
    args = parser.parse_args()

    print("=" * 60)
    print("  NeuralForge — AI-Powered Analysis Engine")
    print("=" * 60)

    analyzer = NeuralForgeAnalyzer(backend=args.backend)

    if args.mode == 'report':
        generate_analysis_report(analyzer)
    elif args.mode == 'performance':
        print(analyzer.analyze_performance())
    elif args.mode == 'quantization':
        print(analyzer.analyze_quantization())
    elif args.mode == 'architecture':
        print(analyzer.analyze_architecture())
    elif args.mode == 'interactive':
        if args.question:
            print(analyzer.interactive_qa(args.question))
        else:
            print("\nInteractive Q&A Mode (type 'quit' to exit)")
            print("-" * 40)
            while True:
                q = input("\n❓ Your question: ").strip()
                if q.lower() in ('quit', 'exit', 'q'):
                    break
                if q:
                    print(f"\n{analyzer.interactive_qa(q)}")


if __name__ == "__main__":
    main()
