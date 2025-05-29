#!/usr/bin/env python3
"""
Simple demo script showing BNN model usage
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from BNN_model import build_cam4_deep, build_cam4_shallow, TALLClassifier


def demo_basic_usage():
    """Demonstrate basic BNN model usage"""
    print("="*60)
    print("BNN MODEL DEMO")
    print("="*60)
    
    # Create models
    print("Creating BNN models...")
    deep_model = build_cam4_deep(num_classes=10)
    shallow_model = build_cam4_shallow(num_classes=10)
    
    # Print model info
    deep_params = sum(p.numel() for p in deep_model.parameters())
    shallow_params = sum(p.numel() for p in shallow_model.parameters())
    
    print(f"Deep model parameters:    {deep_params:,}")
    print(f"Shallow model parameters: {shallow_params:,}")
    
    # Create dummy MNIST-like input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 28, 28)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    
    # Deep model
    deep_model.eval()
    with torch.no_grad():
        deep_output = deep_model(dummy_input)
        deep_pred = deep_output.argmax(dim=1)
    
    print(f"Deep model output shape: {deep_output.shape}")
    print(f"Deep model predictions: {deep_pred.tolist()}")
    
    # Shallow model
    shallow_model.eval()
    with torch.no_grad():
        shallow_output = shallow_model(dummy_input)
        shallow_pred = shallow_output.argmax(dim=1)
    
    print(f"Shallow model output shape: {shallow_output.shape}")
    print(f"Shallow model predictions: {shallow_pred.tolist()}")
    
    # Test TALL voting
    print("\nTesting TALL voting...")
    tall_deep = TALLClassifier(deep_model, num_iter=10, flip_p=0.3)
    tall_shallow = TALLClassifier(shallow_model, num_iter=10, flip_p=0.3)
    
    with torch.no_grad():
        tall_deep_pred = tall_deep(dummy_input)
        tall_shallow_pred = tall_shallow(dummy_input)
    
    print(f"TALL deep predictions:    {tall_deep_pred.tolist()}")
    print(f"TALL shallow predictions: {tall_shallow_pred.tolist()}")
    
    # Test different TALL parameters
    print("\nTesting different TALL parameters...")
    flip_probs = [0.1, 0.3, 0.5]
    iterations = [5, 10, 20]
    
    for flip_p in flip_probs:
        tall_model = TALLClassifier(deep_model, num_iter=10, flip_p=flip_p)
        with torch.no_grad():
            pred = tall_model(dummy_input)
        print(f"flip_p={flip_p:.1f}, iter=10: {pred.tolist()}")
    
    for num_iter in iterations:
        tall_model = TALLClassifier(deep_model, num_iter=num_iter, flip_p=0.3)
        with torch.no_grad():
            pred = tall_model(dummy_input)
        print(f"flip_p=0.3, iter={num_iter:2d}: {pred.tolist()}")


def demo_model_size():
    """Compare model sizes"""
    print("\n" + "="*60)
    print("MODEL SIZE COMPARISON")
    print("="*60)
    
    models = {
        'Deep BNN': build_cam4_deep(),
        'Shallow BNN': build_cam4_shallow(),
    }
    
    print(f"{'Model':<15} {'Parameters':<12} {'Memory (MB)':<12}")
    print("-" * 40)
    
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        # Estimate memory usage (assuming 32-bit floats)
        memory_mb = total_params * 4 / (1024 * 1024)
        print(f"{name:<15} {total_params:<12,} {memory_mb:<12.2f}")


def demo_binarization():
    """Demonstrate the binarization process"""
    print("\n" + "="*60)
    print("BINARIZATION DEMO")
    print("="*60)
    
    # Create a small example
    x = torch.randn(5, 3)
    print("Original values:")
    print(x.numpy())
    
    # Show binarization
    from BNN_model import binarize
    x_bin = binarize(x)
    print("\nBinarized values:")
    print(x_bin.numpy())
    
    # Show the effect on a simple computation
    weight = torch.randn(3, 2)
    print(f"\nOriginal weight matrix:")
    print(weight.numpy())
    
    weight_bin = binarize(weight)
    print(f"\nBinarized weight matrix:")
    print(weight_bin.numpy())
    
    # Compare outputs
    output_fp = torch.mm(x, weight)
    output_bin = torch.mm(x_bin, weight_bin)
    
    print(f"\nFP32 output:")
    print(output_fp.numpy())
    print(f"\nBinary output:")
    print(output_bin.numpy())


if __name__ == '__main__':
    demo_basic_usage()
    demo_model_size()
    demo_binarization()
