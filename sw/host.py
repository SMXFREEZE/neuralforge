"""
NeuralForge — UART Host Communication Script
Sends MNIST test images to FPGA over UART and receives classification results.
Supports simulated mode (no hardware) for testing the software pipeline.
"""

import os
import sys
import time
import json
import argparse
import numpy as np

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


class SimulatedUART:
    """Simulated UART for testing without hardware."""

    def __init__(self):
        self.rx_buffer = bytearray()
        self.tx_buffer = bytearray()
        print("[SIM] Using simulated UART (no hardware connected)")

    def write(self, data):
        """Simulate sending data to FPGA."""
        self.tx_buffer.extend(data)

        # When we've sent a full image (784 bytes), generate a mock result
        if len(self.tx_buffer) >= 784:
            # Simple mock classification based on first pixel
            first_pixel = self.tx_buffer[0]
            mock_result = (first_pixel >> 4) % 10
            self.rx_buffer.append(mock_result)
            self.tx_buffer = bytearray()

    def read(self, size=1):
        """Simulate reading response from FPGA."""
        if len(self.rx_buffer) >= size:
            result = bytes(self.rx_buffer[:size])
            self.rx_buffer = self.rx_buffer[size:]
            return result
        time.sleep(0.001)  # Simulate latency
        return b''

    @property
    def in_waiting(self):
        return len(self.rx_buffer)

    def close(self):
        pass


def load_mnist_samples(num_samples=100):
    """Load MNIST test samples for inference."""
    try:
        from torchvision import datasets, transforms
        import torch

        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        images = []
        labels = []
        for i in range(min(num_samples, len(test_dataset))):
            img, label = test_dataset[i]
            # Convert to INT8 (scale to [-128, 127])
            img_int8 = (img.numpy().flatten() * 255 - 128).astype(np.int8)
            images.append(img_int8)
            labels.append(label)

        return images, labels
    except ImportError:
        print("PyTorch not available. Generating random test data.")
        images = [np.random.randint(-128, 127, 784, dtype=np.int8) for _ in range(num_samples)]
        labels = [np.random.randint(0, 10) for _ in range(num_samples)]
        return images, labels


def run_inference(port, images, labels, timeout=5.0):
    """Send images to FPGA and collect classification results."""
    results = []
    total_time = 0

    for i, (image, true_label) in enumerate(zip(images, labels)):
        # Convert INT8 to unsigned bytes for transmission
        tx_data = bytearray((image.astype(np.int16) + 128).astype(np.uint8))

        # Send image
        start = time.perf_counter()
        port.write(tx_data)

        # Wait for result
        result_byte = b''
        elapsed = 0
        while elapsed < timeout:
            if port.in_waiting > 0:
                result_byte = port.read(1)
                break
            time.sleep(0.001)
            elapsed = time.perf_counter() - start

        end = time.perf_counter()
        inference_time_ms = (end - start) * 1000
        total_time += inference_time_ms

        if result_byte:
            predicted = result_byte[0] & 0x0F
            correct = predicted == true_label
            results.append({
                'index': i,
                'true_label': true_label,
                'predicted': predicted,
                'correct': correct,
                'time_ms': inference_time_ms,
            })

            if i % 10 == 0:
                print(f"  [{i:4d}/{len(images)}] "
                      f"True: {true_label}  Pred: {predicted}  "
                      f"{'✓' if correct else '✗'}  "
                      f"{inference_time_ms:.1f}ms")
        else:
            print(f"  [{i:4d}] TIMEOUT — no response within {timeout}s")
            results.append({
                'index': i,
                'true_label': true_label,
                'predicted': -1,
                'correct': False,
                'time_ms': timeout * 1000,
            })

    return results, total_time


def main():
    parser = argparse.ArgumentParser(description='NeuralForge FPGA Host Interface')
    parser.add_argument('--port', type=str, default='SIM',
                        help='Serial port (e.g., COM3 or /dev/ttyUSB0). Use SIM for simulation.')
    parser.add_argument('--baud', type=int, default=115200, help='UART baud rate')
    parser.add_argument('--samples', type=int, default=100, help='Number of test images')
    parser.add_argument('--timeout', type=float, default=5.0, help='Per-image timeout (seconds)')
    args = parser.parse_args()

    print("=" * 60)
    print("  NeuralForge — FPGA Host Interface")
    print("=" * 60)

    # Setup UART connection
    if args.port == 'SIM':
        port = SimulatedUART()
    else:
        if not HAS_SERIAL:
            print("ERROR: pyserial not installed. Run: pip install pyserial")
            sys.exit(1)
        port = serial.Serial(args.port, args.baud, timeout=args.timeout)
        print(f"Connected to {args.port} at {args.baud} baud")

    # Load test images
    print(f"\nLoading {args.samples} MNIST test images...")
    images, labels = load_mnist_samples(args.samples)
    print(f"Loaded {len(images)} images")

    # Run inference
    print(f"\n" + "-" * 60)
    print(f"Running inference...")
    print("-" * 60)

    results, total_time = run_inference(port, images, labels, args.timeout)

    # Statistics
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = 100.0 * correct / total if total > 0 else 0
    avg_time = total_time / total if total > 0 else 0
    throughput = 1000.0 / avg_time if avg_time > 0 else 0

    print(f"\n" + "=" * 60)
    print(f"  Results Summary")
    print(f"  Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"  Avg latency: {avg_time:.2f} ms/image")
    print(f"  Throughput: {throughput:.1f} images/sec")
    print(f"  Total time: {total_time:.1f} ms")
    print("=" * 60)

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/inference_results.json", 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'avg_latency_ms': avg_time,
            'throughput_ips': throughput,
            'num_samples': total,
            'results': results,
        }, f, indent=2)
    print(f"\nDetailed results saved to results/inference_results.json")

    port.close()


if __name__ == "__main__":
    main()
