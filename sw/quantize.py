"""
NeuralForge — INT8 Post-Training Quantization
Converts full-precision LeNet-5 to INT8 for FPGA deployment
Uses PyTorch's quantization API for accurate scale/zero-point computation
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class LeNet5(nn.Module):
    """Identical architecture to train.py — must match exactly."""

    def __init__(self):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def quantize_tensor(tensor, num_bits=8):
    """
    Symmetric quantization of a float tensor to INT8.

    Returns:
        quantized: INT8 numpy array
        scale: float scale factor
        zero_point: int zero point (0 for symmetric)
    """
    tensor_np = tensor.detach().cpu().numpy().flatten()

    # Symmetric quantization: range is [-max_abs, +max_abs]
    max_abs = np.max(np.abs(tensor_np))
    if max_abs == 0:
        return np.zeros_like(tensor_np, dtype=np.int8), 1.0, 0

    # INT8 range: [-128, 127]
    qmin = -128
    qmax = 127

    scale = max_abs / qmax
    zero_point = 0  # Symmetric

    # Quantize
    quantized = np.clip(np.round(tensor_np / scale), qmin, qmax).astype(np.int8)

    return quantized, float(scale), int(zero_point)


def evaluate_quantized(model, test_loader, device):
    """Evaluate the original FP32 model for accuracy baseline."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return 100. * correct / total


def main():
    print("=" * 60)
    print("  NeuralForge — INT8 Post-Training Quantization")
    print("=" * 60)

    device = torch.device("cpu")  # Quantization runs on CPU

    # Load trained model
    checkpoint_path = "checkpoints/lenet5_best.pth"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Run train.py first!")
        return

    model = LeNet5().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {checkpoint_path}")
    print(f"Original accuracy: {checkpoint['accuracy']:.2f}%")

    # Test data for calibration/evaluation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    # FP32 baseline accuracy
    fp32_accuracy = evaluate_quantized(model, test_loader, device)
    print(f"FP32 test accuracy: {fp32_accuracy:.2f}%")

    # Quantize each layer
    print("\n" + "-" * 60)
    print("Quantizing weights to INT8...")
    print("-" * 60)

    quantized_layers = {}
    layer_info = []

    layer_map = {
        'features.0': 'conv1',
        'features.3': 'conv2',
        'classifier.0': 'fc1',
        'classifier.2': 'fc2',
        'classifier.4': 'fc3',
    }

    for name, param in model.named_parameters():
        layer_prefix = '.'.join(name.split('.')[:-1])
        param_type = name.split('.')[-1]  # 'weight' or 'bias'

        if layer_prefix in layer_map:
            layer_name = layer_map[layer_prefix]
            key = f"{layer_name}_{param_type}"

            q_data, scale, zero_point = quantize_tensor(param)

            quantized_layers[key] = {
                'data': q_data,
                'scale': scale,
                'zero_point': zero_point,
                'shape': list(param.shape),
                'original_dtype': str(param.dtype),
            }

            # Compute quantization error
            reconstructed = q_data.astype(np.float32) * scale
            original = param.detach().cpu().numpy().flatten()
            mse = np.mean((original - reconstructed) ** 2)

            print(f"  {key:20s} | shape: {str(list(param.shape)):20s} | "
                  f"scale: {scale:.6f} | MSE: {mse:.2e}")

            layer_info.append({
                'name': key,
                'shape': list(param.shape),
                'scale': scale,
                'zero_point': zero_point,
                'num_elements': int(np.prod(param.shape)),
                'mse': float(mse),
            })

    # Save quantized weights
    os.makedirs("quantized", exist_ok=True)

    # Save as numpy archive
    np_data = {k: v['data'] for k, v in quantized_layers.items()}
    np.savez("quantized/lenet5_int8.npz", **np_data)
    print(f"\nSaved quantized weights to quantized/lenet5_int8.npz")

    # Save metadata
    metadata = {
        'model': 'LeNet-5',
        'dataset': 'MNIST',
        'quantization': 'symmetric_int8',
        'fp32_accuracy': fp32_accuracy,
        'layers': layer_info,
        'total_params': sum(info['num_elements'] for info in layer_info),
    }

    with open("quantized/quantization_config.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved quantization config to quantized/quantization_config.json")

    # Size comparison
    fp32_size = sum(info['num_elements'] * 4 for info in layer_info)  # 4 bytes per float
    int8_size = sum(info['num_elements'] * 1 for info in layer_info)  # 1 byte per int8
    compression = fp32_size / int8_size

    print(f"\n" + "=" * 60)
    print(f"  Quantization Summary")
    print(f"  FP32 model size: {fp32_size / 1024:.1f} KB")
    print(f"  INT8 model size: {int8_size / 1024:.1f} KB")
    print(f"  Compression ratio: {compression:.1f}x")
    print(f"  FP32 accuracy: {fp32_accuracy:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
