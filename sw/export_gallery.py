"""
Export a curated set of MNIST test images as JSON for the dashboard gallery.
Saves 10 samples per digit (100 total) as pixel arrays with labels.
"""

import json
import os
import sys
import numpy as np

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'weights')


def export_gallery():
    try:
        from torchvision import datasets, transforms
        print("Loading MNIST test set...")
        test_data = datasets.MNIST('./data', train=False, download=True)

        gallery = {'samples': [], 'version': '1.0'}
        per_class = {i: 0 for i in range(10)}
        target_per_class = 10

        for img, label in test_data:
            if per_class[label] >= target_per_class:
                continue
            pixels = [int(x) for x in np.array(img).flatten()]
            gallery['samples'].append({
                'label': int(label),
                'pixels': pixels
            })
            per_class[label] += 1
            if all(v >= target_per_class for v in per_class.values()):
                break

        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        out_path = os.path.join(WEIGHTS_DIR, 'mnist_gallery.json')
        with open(out_path, 'w') as f:
            json.dump(gallery, f)

        size_kb = os.path.getsize(out_path) / 1024
        print(f"Gallery saved: {len(gallery['samples'])} samples ({size_kb:.1f} KB)")
        return True

    except ImportError:
        print("PyTorch/torchvision not available. Generating synthetic gallery...")
        return generate_synthetic_gallery()


def generate_synthetic_gallery():
    """Generate a minimal synthetic gallery when torch is not available."""
    np.random.seed(42)
    gallery = {'samples': [], 'version': '1.0'}

    for digit in range(10):
        for _ in range(10):
            # Create a simple synthetic digit image
            img = np.zeros((28, 28), dtype=int)
            # Draw a simple vertical/horizontal line pattern per digit
            if digit == 0:
                img[8:20, 10] = 200; img[8:20, 17] = 200; img[8, 10:18] = 200; img[19, 10:18] = 200
            elif digit == 1:
                img[6:22, 14] = 220; img[6, 12:15] = 180
            elif digit == 2:
                img[8, 10:18] = 200; img[8:14, 17] = 200; img[13, 10:18] = 200; img[13:20, 10] = 200; img[19, 10:18] = 200
            elif digit == 3:
                img[8, 10:18] = 200; img[8:20, 17] = 200; img[13, 10:18] = 200; img[19, 10:18] = 200
            elif digit == 4:
                img[8:14, 10] = 200; img[13, 10:18] = 200; img[8:20, 17] = 200
            elif digit == 5:
                img[8, 10:18] = 200; img[8:14, 10] = 200; img[13, 10:18] = 200; img[13:20, 17] = 200; img[19, 10:18] = 200
            elif digit == 6:
                img[8, 10:18] = 200; img[8:20, 10] = 200; img[13, 10:18] = 200; img[13:20, 17] = 200; img[19, 10:18] = 200
            elif digit == 7:
                img[8, 10:18] = 200; img[8:20, 17] = 200
            elif digit == 8:
                img[8, 10:18] = 200; img[8:20, 10] = 200; img[8:20, 17] = 200; img[13, 10:18] = 200; img[19, 10:18] = 200
            elif digit == 9:
                img[8, 10:18] = 200; img[8:14, 10] = 200; img[8:20, 17] = 200; img[13, 10:18] = 200; img[19, 10:18] = 200

            # Add a bit of noise for variation
            noise = np.random.randint(-15, 15, (28, 28))
            img = np.clip(img + noise, 0, 255)
            gallery['samples'].append({'label': digit, 'pixels': img.flatten().tolist()})

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    out_path = os.path.join(WEIGHTS_DIR, 'mnist_gallery.json')
    with open(out_path, 'w') as f:
        json.dump(gallery, f)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Synthetic gallery saved: {len(gallery['samples'])} samples ({size_kb:.1f} KB)")
    return True


if __name__ == '__main__':
    export_gallery()
