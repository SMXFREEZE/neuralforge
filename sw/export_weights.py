"""
NeuralForge — Weight Export to Verilog-Ready Format
Converts INT8 quantized weights to .mem (hex) files for BRAM initialization
Also generates individual layer weight files for modular synthesis
"""

import os
import json
import numpy as np


def to_hex_signed(value, bits=8):
    """Convert signed integer to hex string (two's complement)."""
    if value < 0:
        value = (1 << bits) + value
    return format(value, f'0{bits // 4}x')


def export_mem_file(weights, filename, values_per_line=1):
    """Export weights as Verilog $readmemh compatible .mem file."""
    with open(filename, 'w') as f:
        f.write(f"// NeuralForge weight file: {os.path.basename(filename)}\n")
        f.write(f"// Total entries: {len(weights)}\n")
        f.write(f"// Format: INT8 signed (two's complement hex)\n\n")

        for i in range(0, len(weights), values_per_line):
            chunk = weights[i:i + values_per_line]
            hex_values = ' '.join(to_hex_signed(int(v)) for v in chunk)
            f.write(hex_values + '\n')


def export_coe_file(weights, filename):
    """Export weights as Xilinx COE file for Block RAM IP."""
    with open(filename, 'w') as f:
        f.write("; NeuralForge weight file (Xilinx COE format)\n")
        f.write(f"; Total entries: {len(weights)}\n")
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")

        for i, w in enumerate(weights):
            hex_val = to_hex_signed(int(w))
            if i < len(weights) - 1:
                f.write(f"{hex_val},\n")
            else:
                f.write(f"{hex_val};\n")


def main():
    print("=" * 60)
    print("  NeuralForge — Weight Export (Verilog .mem)")
    print("=" * 60)

    # Load quantized weights
    quant_path = "quantized/lenet5_int8.npz"
    config_path = "quantized/quantization_config.json"

    if not os.path.exists(quant_path):
        print(f"ERROR: {quant_path} not found. Run quantize.py first!")
        return

    data = np.load(quant_path)
    with open(config_path, 'r') as f:
        config = json.load(f)

    os.makedirs("weights", exist_ok=True)

    print("\nExporting layers:")
    print("-" * 60)

    all_weights = []       # Combined flat array for single BRAM
    weight_map = {}        # Address mapping for each layer
    current_addr = 0

    layer_order = ['conv1_weight', 'conv1_bias', 'conv2_weight', 'conv2_bias',
                   'fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias',
                   'fc3_weight', 'fc3_bias']

    for layer_name in layer_order:
        if layer_name not in data:
            print(f"  [SKIP] {layer_name} not found in quantized data")
            continue

        weights = data[layer_name].flatten()
        num_elements = len(weights)

        # Record address mapping
        weight_map[layer_name] = {
            'start_addr': current_addr,
            'end_addr': current_addr + num_elements - 1,
            'num_elements': num_elements,
        }

        # Find layer config for shape info
        layer_config = next(
            (l for l in config['layers'] if l['name'] == layer_name),
            None
        )
        shape_str = str(layer_config['shape']) if layer_config else "?"

        print(f"  {layer_name:20s} | {num_elements:6d} elements | "
              f"shape: {shape_str:20s} | addr: 0x{current_addr:04x}-0x{current_addr + num_elements - 1:04x}")

        # Export individual layer .mem file
        export_mem_file(weights, f"weights/{layer_name}.mem")

        # Export Xilinx COE file
        export_mem_file(weights, f"weights/{layer_name}.coe")

        # Append to combined array
        all_weights.extend(weights.tolist())
        current_addr += num_elements

    # Export combined weights file
    print(f"\n  Combined weights: {len(all_weights)} entries")
    export_mem_file(all_weights, "weights/weights.mem")
    export_coe_file(all_weights, "weights/weights_combined.coe")

    # Export address map as JSON
    weight_map['_total_entries'] = len(all_weights)
    weight_map['_memory_depth'] = 2 ** (len(all_weights) - 1).bit_length()  # Next power of 2

    with open("weights/weight_map.json", 'w') as f:
        json.dump(weight_map, f, indent=2)

    # Export scale factors for software-side dequantization
    scale_factors = {}
    for layer_info in config['layers']:
        scale_factors[layer_info['name']] = {
            'scale': layer_info['scale'],
            'zero_point': layer_info['zero_point'],
        }

    with open("weights/scale_factors.json", 'w') as f:
        json.dump(scale_factors, f, indent=2)

    print(f"\nFiles written to weights/:")
    print(f"  - weights.mem           (combined Verilog init file)")
    print(f"  - weights_combined.coe  (Xilinx COE format)")
    print(f"  - weight_map.json       (address mapping)")
    print(f"  - scale_factors.json    (quantization parameters)")
    print(f"  - <layer>_weight.mem    (per-layer files)")

    print(f"\n" + "=" * 60)
    print(f"  Export Complete! Total: {len(all_weights)} INT8 weights")
    print(f"  BRAM depth needed: {weight_map['_memory_depth']} entries")
    print(f"  Memory usage: {len(all_weights)} bytes = {len(all_weights) / 1024:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()
