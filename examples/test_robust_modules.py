"""
Example script to test robustness modules.

Demonstrates how to build models with and without robustness modules.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models import build_model
from src.utils.config import load_config


def test_baseline_vs_robust():
    """Compare baseline and robust models."""
    
    print("="*60)
    print("Testing Baseline Model (no robustness modules)")
    print("="*60)
    
    config = load_config('configs/base.yaml')
    model_baseline = build_model(config, num_classes=1, img_size=640)
    
    info = model_baseline.get_model_info()
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Robustness module params: {info.get('robust_params', 0):,}")
    
    # Test forward pass
    dummy_images = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        preds = model_baseline(dummy_images, mode='val')
    print(f"Output scales: {list(preds.keys())}")
    print("Baseline model works!\n")
    
    print("="*60)
    print("Testing Robust Model (all modules enabled)")
    print("="*60)
    
    # Enable robustness modules
    config['robust_modules']['use_background_suppression'] = True
    config['robust_modules']['use_frequency_enhancement'] = True
    config['robust_modules']['use_condition_aware'] = True
    
    model_robust = build_model(config, num_classes=1, img_size=640)
    info_robust = model_robust.get_model_info()
    
    print(f"Total parameters: {info_robust['total_params']:,}")
    print(f"Robustness module params: {info_robust.get('robust_params', 0):,}")
    print(f"Additional params: {info_robust['total_params'] - info['total_params']:,}")
    
    # Test forward pass
    with torch.no_grad():
        preds_robust = model_robust(dummy_images, mode='val')
    print(f"Output scales: {list(preds_robust.keys())}")
    print("Robust model works!\n")
    
    print("="*60)
    print("Comparison:")
    print(f"  Baseline: {info['total_params']:,} params")
    print(f"  Robust:   {info_robust['total_params']:,} params")
    print(f"  Overhead: {info_robust['total_params'] - info['total_params']:,} params ({100 * (info_robust['total_params'] - info['total_params']) / info['total_params']:.2f}%)")
    print("="*60)


if __name__ == "__main__":
    test_baseline_vs_robust()

