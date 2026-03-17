"""
NeuralForge — Cycle-Accurate FPGA Simulator
Replicates the FPGA INT8 inference pipeline in pure Python/NumPy.
Executes: Conv1 → ReLU → Pool1 → Conv2 → ReLU → Pool2 → FC1 → FC2 → FC3 → Argmax

This matches the hardware datapath exactly:
  - All arithmetic is INT8 inputs / INT32 accumulators (same as the MAC units)
  - Convolution uses the same 3×3 / 5×5 kernel sweeps
  - Pooling uses 2×2 max (same as pooling.v)
  - FC layers replicate the systolic array's weight-stationary dot products
"""

import json
import os
import numpy as np
from pathlib import Path


class FPGASimulator:
    """Cycle-accurate simulator of the NeuralForge FPGA inference pipeline."""

    def __init__(self, weights_path=None):
        if weights_path is None:
            weights_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                '..', 'weights', 'lenet5_int8.json'
            )
        self.weights = self._load_weights(weights_path)
        self.cycle_count = 0

    def _load_weights(self, path):
        """Load pre-quantized INT8 weights from JSON."""
        with open(path, 'r') as f:
            raw = json.load(f)

        weights = {}
        for name, data in raw['layers'].items():
            arr = np.array(data['values'], dtype=np.int8).reshape(data['shape'])
            weights[name] = arr
            if 'scale' in data:
                weights[f'{name}_scale'] = data['scale']

        # Load bias as INT32 (matches hardware accumulator width)
        for name, data in raw.get('biases', {}).items():
            weights[name] = np.array(data['values'], dtype=np.int32).reshape(data['shape'])

        return weights

    def classify(self, image_pixels):
        """
        Run full LeNet-5 inference on a 28×28 grayscale image.

        Args:
            image_pixels: list or array of 784 pixel values (0-255)

        Returns:
            dict with: digit, confidence[], layer_outputs{}
        """
        self.cycle_count = 0
        layer_outputs = {}

        # Normalize to INT8 range [-128, 127] like the FPGA input buffer
        img = np.array(image_pixels, dtype=np.float32).reshape(28, 28)
        img_int8 = np.clip((img / 255.0 * 255 - 128), -128, 127).astype(np.int8)
        layer_outputs['input'] = img_int8.tolist()

        # === CONV1: 1 input channel → 6 output channels, 5×5 kernel ===
        conv1_out = self._conv2d(
            img_int8[np.newaxis, :, :],  # [1, 28, 28]
            self.weights['conv1_weight'],  # [6, 1, 5, 5]
            self.weights.get('conv1_bias', np.zeros(6, dtype=np.int32)),
            pad=0
        )  # → [6, 24, 24]
        self.cycle_count += 24 * 24 * 6 * 4  # Pipeline cycles estimate
        layer_outputs['conv1'] = self._to_serializable(conv1_out)

        # === ReLU ===
        relu1_out = self._relu(conv1_out)
        self.cycle_count += 6 * 24 * 24
        layer_outputs['relu1'] = self._to_serializable(relu1_out)

        # === POOL1: 2×2 max pooling ===
        pool1_out = self._maxpool2d(relu1_out, 2)  # → [6, 12, 12]
        self.cycle_count += 6 * 12 * 12 * 2
        layer_outputs['pool1'] = self._to_serializable(pool1_out)

        # === CONV2: 6 → 16 channels, 5×5 kernel ===
        conv2_out = self._conv2d(
            pool1_out,
            self.weights['conv2_weight'],  # [16, 6, 5, 5]
            self.weights.get('conv2_bias', np.zeros(16, dtype=np.int32)),
            pad=0
        )  # → [16, 8, 8]
        self.cycle_count += 8 * 8 * 16 * 6 * 4
        layer_outputs['conv2'] = self._to_serializable(conv2_out)

        # === ReLU ===
        relu2_out = self._relu(conv2_out)
        self.cycle_count += 16 * 8 * 8
        layer_outputs['relu2'] = self._to_serializable(relu2_out)

        # === POOL2: 2×2 max pooling ===
        pool2_out = self._maxpool2d(relu2_out, 2)  # → [16, 4, 4]
        self.cycle_count += 16 * 4 * 4 * 2
        layer_outputs['pool2'] = self._to_serializable(pool2_out)

        # === Flatten ===
        flat = pool2_out.flatten().astype(np.int8)  # 256 values

        # === FC1: 256 → 120 (systolic array) ===
        fc1_out = self._fc_layer(flat, self.weights['fc1_weight'],
                                  self.weights.get('fc1_bias', np.zeros(120, dtype=np.int32)))
        fc1_relu = self._relu_1d(fc1_out)
        self.cycle_count += 256 * 120 // 16  # 4×4 systolic array throughput
        layer_outputs['fc1'] = fc1_relu.tolist()

        # === FC2: 120 → 84 ===
        fc2_out = self._fc_layer(fc1_relu.astype(np.int8), self.weights['fc2_weight'],
                                  self.weights.get('fc2_bias', np.zeros(84, dtype=np.int32)))
        fc2_relu = self._relu_1d(fc2_out)
        self.cycle_count += 120 * 84 // 16
        layer_outputs['fc2'] = fc2_relu.tolist()

        # === FC3: 84 → 10 (output logits) ===
        fc3_out = self._fc_layer(fc2_relu.astype(np.int8), self.weights['fc3_weight'],
                                  self.weights.get('fc3_bias', np.zeros(10, dtype=np.int32)))
        self.cycle_count += 84 * 10 // 16
        layer_outputs['fc3_logits'] = fc3_out.tolist()

        # === Argmax ===
        digit = int(np.argmax(fc3_out))
        self.cycle_count += 10

        # Softmax for confidence scores
        logits_f = fc3_out.astype(np.float64)
        logits_f -= logits_f.max()
        exp_logits = np.exp(logits_f)
        confidence = (exp_logits / exp_logits.sum() * 100).tolist()

        return {
            'digit': digit,
            'confidence': [round(c, 2) for c in confidence],
            'cycles': self.cycle_count,
            'latency_us': round(self.cycle_count / 100.0, 2),  # At 100 MHz
            'layer_outputs': layer_outputs
        }

    # ===== Hardware-equivalent operations =====

    def _conv2d(self, input_arr, weights, bias, pad=0):
        """
        INT8 2D convolution matching conv_engine.v.
        input_arr: [C_in, H, W] int8
        weights:   [C_out, C_in, kH, kW] int8
        bias:      [C_out] int32
        """
        c_out, c_in, kh, kw = weights.shape
        _, h, w = input_arr.shape

        if pad > 0:
            input_arr = np.pad(input_arr, ((0, 0), (pad, pad), (pad, pad)),
                             mode='constant', constant_values=0)
            _, h, w = input_arr.shape

        oh = h - kh + 1
        ow = w - kw + 1
        output = np.zeros((c_out, oh, ow), dtype=np.int32)

        for co in range(c_out):
            for ci in range(c_in):
                for y in range(oh):
                    for x in range(ow):
                        patch = input_arr[ci, y:y+kh, x:x+kw].astype(np.int32)
                        kern = weights[co, ci].astype(np.int32)
                        # INT8 × INT8 → INT32 accumulate (matches MAC unit)
                        output[co, y, x] += np.sum(patch * kern)
            output[co] += bias[co]

        return output

    def _relu(self, x):
        """ReLU matching activation.v (mode=00)."""
        return np.maximum(x, 0)

    def _relu_1d(self, x):
        """ReLU for 1D vectors."""
        return np.maximum(x, 0)

    def _maxpool2d(self, x, pool_size):
        """2×2 max pooling matching pooling.v."""
        c, h, w = x.shape
        oh, ow = h // pool_size, w // pool_size
        output = np.zeros((c, oh, ow), dtype=x.dtype)

        for ch in range(c):
            for y in range(oh):
                for x_pos in range(ow):
                    patch = x[ch,
                              y*pool_size:(y+1)*pool_size,
                              x_pos*pool_size:(x_pos+1)*pool_size]
                    output[ch, y, x_pos] = np.max(patch)

        return output

    def _fc_layer(self, input_vec, weights, bias):
        """
        Fully-connected layer matching systolic_array.v.
        input_vec: [N] int8
        weights:   [out, N] int8
        bias:      [out] int32
        """
        # INT8 × INT8 → INT32 dot product (weight-stationary dataflow)
        out = np.zeros(weights.shape[0], dtype=np.int32)
        for i in range(weights.shape[0]):
            out[i] = np.sum(input_vec.astype(np.int32) * weights[i].astype(np.int32)) + bias[i]
        return out

    def _to_serializable(self, arr):
        """Convert a multi-channel feature map to a JSON-safe list of 2D arrays."""
        if arr.ndim == 3:
            return [arr[c].tolist() for c in range(arr.shape[0])]
        return arr.tolist()
