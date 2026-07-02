"""Tests for the cycle-accurate FPGA simulator model (numpy-only, no torch)."""

import json
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "sw"))

from fpga_simulator import FPGASimulator  # noqa: E402

GALLERY_PATH = os.path.join(REPO_ROOT, "weights", "mnist_gallery.json")

# The per-layer cycle model in FPGASimulator.classify sums fixed layer sizes,
# so the count is input-independent. Keep in sync with SIMULATOR_CYCLES in
# components/TryIt.tsx.
EXPECTED_CYCLES = 47_732


@pytest.fixture(scope="module")
def simulator():
    return FPGASimulator()


@pytest.fixture(scope="module")
def gallery():
    with open(GALLERY_PATH) as f:
        data = json.load(f)
    return data["samples"]


def test_gallery_accuracy_at_least_95_percent(simulator, gallery):
    images = [s["pixels"] for s in gallery]
    labels = [s["label"] for s in gallery]
    result = simulator.test_accuracy(images, labels, max_samples=len(gallery))
    assert result["total"] == len(gallery)
    assert result["accuracy"] >= 95.0, (
        f"Bundled-gallery accuracy dropped to {result['accuracy']}% "
        f"({result['correct']}/{result['total']})"
    )


def test_cycle_count_is_deterministic(simulator, gallery):
    pixels = gallery[0]["pixels"]
    first = simulator.classify(pixels)
    second = simulator.classify(pixels)
    assert first["cycles"] == second["cycles"] == EXPECTED_CYCLES
    assert first["latency_us"] == pytest.approx(EXPECTED_CYCLES / 100.0)
    # The cycle model has no data dependence: a different image costs the same.
    other = simulator.classify(gallery[1]["pixels"])
    assert other["cycles"] == EXPECTED_CYCLES


def test_classify_output_shape(simulator, gallery):
    result = simulator.classify(gallery[0]["pixels"])
    assert result["digit"] in range(10)
    assert len(result["confidence"]) == 10
    assert sum(result["confidence"]) == pytest.approx(100.0, abs=0.5)


def test_int8_weight_quantize_dequantize_roundtrip():
    """Symmetric INT8 quantization (scale = max|w| / 127) roundtrips within
    half a quantization step, matching the scheme documented in README and
    used by sw/quantize.py."""
    rng = np.random.default_rng(1234)
    w = rng.normal(0.0, 0.2, size=(6, 1, 5, 5)).astype(np.float32)

    scale = np.abs(w).max() / 127.0
    q = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    dequant = q.astype(np.float32) * scale

    assert q.dtype == np.int8
    assert q.min() >= -128 and q.max() <= 127
    assert np.max(np.abs(dequant - w)) <= scale / 2 + 1e-6


def test_activation_quantizer_roundtrip_identity(simulator):
    """INT32 accumulators whose magnitude already peaks at 127 must pass
    through the dynamic activation quantizer unchanged."""
    x = np.array([-127, -64, 0, 1, 64, 127], dtype=np.int32)
    out = simulator._quantize_activations(x)
    assert out.dtype == np.int8
    assert np.array_equal(out, x.astype(np.int8))

    # And an all-zero tensor stays all-zero without dividing by zero.
    zeros = np.zeros(16, dtype=np.int32)
    assert np.array_equal(simulator._quantize_activations(zeros), zeros.astype(np.int8))
