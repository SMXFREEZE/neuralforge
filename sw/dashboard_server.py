"""
NeuralForge — Web Dashboard Server
Interactive visualization dashboard for the FPGA accelerator.
Serves the dashboard UI and provides API endpoints for:
  - Real-time inference results
  - Performance metrics and charts
  - AI-powered analysis (via Bedrock/OpenAI)
  - Architecture visualization
"""

import os
import sys
import json
import time
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ai_analyzer import NeuralForgeAnalyzer
    HAS_ANALYZER = True
except ImportError:
    HAS_ANALYZER = False

try:
    from fpga_simulator import FPGASimulator
    HAS_SIMULATOR = True
except ImportError:
    HAS_SIMULATOR = False


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP request handler with API routing."""

    # Class-level shared state
    analyzer = None
    simulator = None
    project_root = None
    gallery_data = None

    def __init__(self, *args, **kwargs):
        # Serve from dashboard directory
        kwargs['directory'] = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 'dashboard'
        )
        super().__init__(*args, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith('/api/'):
            self._handle_api(path, parse_qs(parsed.query))
        else:
            super().do_GET()

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/analyze':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body) if body else {}
            self._handle_analyze(data)
        elif path == '/api/classify':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body) if body else {}
            self._handle_classify(data)
        elif path == '/api/accuracy':
            self._handle_accuracy_test()
        else:
            self.send_error(404, "Not Found")

    def _send_json(self, data, status=200):
        response = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', len(response))
        self.end_headers()
        self.wfile.write(response)

    def _handle_api(self, path, params):
        if path == '/api/status':
            self._send_json({
                'status': 'running',
                'version': '1.0.0',
                'backend': DashboardHandler.analyzer.backend_name if DashboardHandler.analyzer else 'none',
                'has_benchmark': os.path.exists(
                    os.path.join(DashboardHandler.project_root, 'results', 'benchmark_results.json')
                ),
                'has_quantization': os.path.exists(
                    os.path.join(DashboardHandler.project_root, 'quantized', 'quantization_config.json')
                ),
            })

        elif path == '/api/benchmark':
            bench_path = os.path.join(DashboardHandler.project_root, 'results', 'benchmark_results.json')
            if os.path.exists(bench_path):
                with open(bench_path) as f:
                    self._send_json(json.load(f))
            else:
                self._send_json(self._generate_demo_benchmark())

        elif path == '/api/quantization':
            quant_path = os.path.join(DashboardHandler.project_root, 'quantized', 'quantization_config.json')
            if os.path.exists(quant_path):
                with open(quant_path) as f:
                    self._send_json(json.load(f))
            else:
                self._send_json(self._generate_demo_quantization())

        elif path == '/api/architecture':
            self._send_json(self._get_architecture_data())

        elif path == '/api/perf_counters':
            self._send_json(self._generate_perf_counter_data())

        elif path == '/api/gallery':
            self._handle_gallery()

        else:
            self.send_error(404, "API endpoint not found")

    def _handle_analyze(self, data):
        if not DashboardHandler.analyzer:
            self._send_json({'error': 'No AI backend configured'}, 500)
            return

        mode = data.get('mode', 'performance')
        question = data.get('question', '')

        try:
            if mode == 'performance':
                result = DashboardHandler.analyzer.analyze_performance()
            elif mode == 'quantization':
                result = DashboardHandler.analyzer.analyze_quantization()
            elif mode == 'architecture':
                result = DashboardHandler.analyzer.analyze_architecture()
            elif mode == 'question':
                result = DashboardHandler.analyzer.interactive_qa(question)
            else:
                result = DashboardHandler.analyzer.analyze_performance()

            self._send_json({'analysis': result})

        except Exception as e:
            self._send_json({'error': str(e)}, 500)

    def _handle_classify(self, data):
        """Run digit classification through the FPGA simulator."""
        if not DashboardHandler.simulator:
            self._send_json({'error': 'Simulator not available'}, 500)
            return

        pixels = data.get('pixels', [])
        if len(pixels) != 784:
            self._send_json({'error': f'Expected 784 pixels, got {len(pixels)}'}, 400)
            return

        try:
            result = DashboardHandler.simulator.classify(pixels)
            self._send_json(result)
        except Exception as e:
            self._send_json({'error': str(e)}, 500)

    def _handle_gallery(self):
        """Return MNIST gallery samples."""
        if DashboardHandler.gallery_data:
            self._send_json(DashboardHandler.gallery_data)
        else:
            self._send_json({'samples': [], 'error': 'Gallery not loaded'})

    def _handle_accuracy_test(self):
        """Run batch accuracy test on gallery data."""
        if not DashboardHandler.simulator:
            self._send_json({'error': 'Simulator not available'}, 500)
            return
        if not DashboardHandler.gallery_data:
            self._send_json({'error': 'Gallery not loaded'}, 500)
            return

        samples = DashboardHandler.gallery_data['samples']
        images = [s['pixels'] for s in samples]
        labels = [s['label'] for s in samples]

        try:
            result = DashboardHandler.simulator.test_accuracy(images, labels)
            self._send_json(result)
        except Exception as e:
            self._send_json({'error': str(e)}, 500)

    def _generate_demo_benchmark(self):
        """Generate demo benchmark data when no real data exists."""
        import random
        random.seed(42)
        return {
            'cpu': {
                'mean_ms': 2.134,
                'median_ms': 2.089,
                'p95_ms': 2.891,
                'p99_ms': 3.245,
                'throughput_ips': 468.6,
                'accuracy': 99.2,
            },
            'fpga': {
                'mean_ms': 0.412,
                'median_ms': 0.408,
                'p95_ms': 0.443,
                'p99_ms': 0.461,
                'throughput_ips': 2427.2,
                'accuracy': 98.5,
            },
            'speedup': 5.18
        }

    def _generate_demo_quantization(self):
        """Generate demo quantization data."""
        return {
            'model': 'LeNet-5',
            'quantization': 'symmetric_int8',
            'fp32_accuracy': 99.2,
            'layers': [
                {'name': 'conv1_weight', 'shape': [6, 1, 5, 5], 'scale': 0.023, 'mse': 1.2e-5, 'num_elements': 150},
                {'name': 'conv1_bias', 'shape': [6], 'scale': 0.005, 'mse': 3.1e-7, 'num_elements': 6},
                {'name': 'conv2_weight', 'shape': [16, 6, 5, 5], 'scale': 0.018, 'mse': 8.7e-6, 'num_elements': 2400},
                {'name': 'conv2_bias', 'shape': [16], 'scale': 0.003, 'mse': 1.8e-7, 'num_elements': 16},
                {'name': 'fc1_weight', 'shape': [120, 256], 'scale': 0.012, 'mse': 5.4e-6, 'num_elements': 30720},
                {'name': 'fc1_bias', 'shape': [120], 'scale': 0.008, 'mse': 4.2e-7, 'num_elements': 120},
                {'name': 'fc2_weight', 'shape': [84, 120], 'scale': 0.015, 'mse': 6.1e-6, 'num_elements': 10080},
                {'name': 'fc2_bias', 'shape': [84], 'scale': 0.006, 'mse': 2.9e-7, 'num_elements': 84},
                {'name': 'fc3_weight', 'shape': [10, 84], 'scale': 0.021, 'mse': 9.3e-6, 'num_elements': 840},
                {'name': 'fc3_bias', 'shape': [10], 'scale': 0.004, 'mse': 1.5e-7, 'num_elements': 10},
            ],
            'total_params': 44426,
        }

    def _get_architecture_data(self):
        return {
            'modules': [
                {'name': 'top', 'type': 'controller', 'luts': 450, 'ffs': 380, 'children': [
                    'uart_interface', 'input_buffer', 'weight_buffer', 'conv_engine',
                    'activation', 'pooling', 'systolic_array', 'layer_controller',
                    'perf_counters', 'register_file'
                ]},
                {'name': 'systolic_array', 'type': 'compute', 'luts': 1200, 'ffs': 800, 'dsps': 16},
                {'name': 'conv_engine', 'type': 'compute', 'luts': 580, 'ffs': 420, 'dsps': 0},
                {'name': 'uart_interface', 'type': 'io', 'luts': 180, 'ffs': 160},
                {'name': 'input_buffer', 'type': 'memory', 'luts': 120, 'ffs': 80, 'bram': 1},
                {'name': 'weight_buffer', 'type': 'memory', 'luts': 80, 'ffs': 60, 'bram': 2},
                {'name': 'layer_controller', 'type': 'controller', 'luts': 320, 'ffs': 280},
                {'name': 'perf_counters', 'type': 'debug', 'luts': 160, 'ffs': 384},
                {'name': 'register_file', 'type': 'io', 'luts': 90, 'ffs': 64},
                {'name': 'activation', 'type': 'compute', 'luts': 45, 'ffs': 35},
                {'name': 'pooling', 'type': 'compute', 'luts': 60, 'ffs': 50},
                {'name': 'fifo', 'type': 'memory', 'luts': 85, 'ffs': 40, 'bram': 1},
            ],
            'total': {'luts': 3370, 'ffs': 2753, 'dsps': 16, 'bram': 4},
            'available': {'luts': 20800, 'ffs': 41600, 'dsps': 90, 'bram': 50},
        }

    def _generate_perf_counter_data(self):
        return {
            'total_cycles': 41283,
            'inference_cycles': 38420,
            'conv_cycles': 24680,
            'pool_cycles': 3200,
            'fc_cycles': 10540,
            'mac_active_cycles': 31200,
            'mem_read_cycles': 8750,
            'stall_cycles': 1890,
            'pipeline_stages': {
                'conv1': {'cycles': 8200, 'macs': 86400},
                'pool1': {'cycles': 1200, 'comparisons': 5184},
                'conv2': {'cycles': 12800, 'macs': 307200},
                'pool2': {'cycles': 800, 'comparisons': 1024},
                'fc1': {'cycles': 3800, 'macs': 30720},
                'fc2': {'cycles': 2100, 'macs': 10080},
                'fc3': {'cycles': 840, 'macs': 840},
            },
            'utilization': {
                'mac_pct': 81.2,
                'memory_bw_pct': 22.8,
                'stall_pct': 4.9,
            }
        }

    def log_message(self, format, *args):
        """Suppress default logging for cleaner output."""
        pass


def main():
    parser = argparse.ArgumentParser(description='NeuralForge Dashboard Server')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--backend', choices=['auto', 'bedrock', 'openai', 'offline'],
                        default='auto', help='AI backend')
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DashboardHandler.project_root = project_root

    if HAS_ANALYZER:
        DashboardHandler.analyzer = NeuralForgeAnalyzer(backend=args.backend)
    else:
        DashboardHandler.analyzer = None

    if HAS_SIMULATOR:
        try:
            DashboardHandler.simulator = FPGASimulator()
            sim_status = 'Ready'
        except Exception as e:
            DashboardHandler.simulator = None
            sim_status = f'Error: {e}'
    else:
        DashboardHandler.simulator = None
        sim_status = 'Not available'

    # Load gallery
    gallery_path = os.path.join(project_root, 'weights', 'mnist_gallery.json')
    if os.path.exists(gallery_path):
        with open(gallery_path) as f:
            DashboardHandler.gallery_data = json.load(f)
        gallery_status = f"{len(DashboardHandler.gallery_data.get('samples', []))} samples"
    else:
        DashboardHandler.gallery_data = None
        gallery_status = 'Not found'

    server = HTTPServer(('0.0.0.0', args.port), DashboardHandler)

    print("=" * 60)
    print("  NeuralForge Dashboard Server")
    print("=" * 60)
    print(f"  URL: http://localhost:{args.port}")
    print(f"  AI Backend: {DashboardHandler.analyzer.backend_name if DashboardHandler.analyzer else 'none'}")
    print(f"  Simulator: {sim_status}")
    print(f"  Gallery: {gallery_status}")
    print(f"  Project: {project_root}")
    print(f"\n  Press Ctrl+C to stop")
    print("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
