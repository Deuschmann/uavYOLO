"""
Example script to test the model architecture.

This script demonstrates how to:
1. Load configuration
2. Build model from config
3. Test forward pass
4. Print model info
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models import build_model
from src.utils.config import load_config


def test_model_forward():
    """Test model forward pass with dummy data."""
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "base.yaml"
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return
    
    config = load_config(config_path)
    
    # Update num_classes in config if needed
    num_classes = config.get('classes', {}).get('num_classes', 1)
    
    print("Building model...")
    model = build_model(
        config=config,
        num_classes=num_classes,
        img_size=config['data']['img_size'],
    )
    
    # Print model info
    info = model.get_model_info()
    print("\n" + "="*50)
    print("Model Information:")
    print("="*50)
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    print(f"  - Backbone: {info['backbone_params']:,}")
    print(f"  - Neck: {info['neck_params']:,}")
    print(f"  - Head: {info['head_params']:,}")
    print(f"Input image size: {info['img_size']}")
    print("="*50 + "\n")
    
    # Test forward pass
    batch_size = 2
    img_size = config['data']['img_size']
    
    print(f"Testing forward pass with batch_size={batch_size}, img_size={img_size}...")
    
    # Create dummy input (normalized [0, 1])
    dummy_images = torch.randn(batch_size, 3, img_size, img_size)
    
    # Forward pass (training mode)
    model.eval()  # Set to eval mode for inference
    with torch.no_grad():
        predictions = model(dummy_images, mode='val')
    
    print("\nForward pass successful!")
    print("\nOutput structure:")
    for scale, preds in predictions.items():
        print(f"  {scale}:")
        print(f"    bbox: {preds['bbox'].shape}")
        print(f"    obj: {preds['obj'].shape}")
        print(f"    cls: {preds['cls'].shape}")
    
    # Test training mode (with targets)
    print("\nTesting training mode...")
    dummy_targets = {
        'boxes': [
            torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),  # Image 1
            torch.tensor([[0.3, 0.3, 0.1, 0.1]], dtype=torch.float32),  # Image 2
        ],
        'labels': [
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
        ],
    }
    
    model.train()
    predictions_train = model(dummy_images, targets=dummy_targets, mode='train')
    
    print("Training forward pass successful!")
    print(f"Output structure matches: {list(predictions_train.keys()) == list(predictions.keys())}")
    
    # Test model size estimation
    param_size_mb = info['total_params'] * 4 / (1024 * 1024)  # Assuming float32
    print(f"\nEstimated model size: ~{param_size_mb:.2f} MB (FP32)")
    
    print("\n✓ Model test completed successfully!")


if __name__ == "__main__":
    test_model_forward()

