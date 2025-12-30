#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.abspath('.'))
import torch
from src.models.l_refine_yolo import build_l_refine_yolo

print("Testing L_Refine_YOLO model...")
try:
    m = build_l_refine_yolo(
        num_classes=3,
        backbone_kwargs={'in_channels': 3, 'width_mult': 0.5, 'depth_mult': 0.5},
        neck_type='fpn',
        neck_out_channels=128,
        head_kwargs={'num_layers': 2}
    )
    print("✓ Model built successfully")
    
    # Get model info
    info = m.get_model_info()
    print(f"✓ Model params: {info['total_params']:,}")
    
    # Test forward
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        preds = m(x)
    print(f"✓ Forward pass OK: {list(preds.keys())}")
    print(f"✓ P3 bbox shape: {preds['P3']['bbox'].shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
