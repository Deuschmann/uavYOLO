import sys, os
sys.path.insert(0, os.path.abspath('.'))
import torch
from src.models.l_refine_yolo import build_l_refine_yolo
from src.models.yolo_model import build_model
from src.utils.config import load_config, merge_configs

# L_Refine_YOLO 轻量化
print("=== L_Refine_YOLO (Light) ===")
m1 = build_l_refine_yolo(
    num_classes=3,
    backbone_kwargs={'in_channels': 3, 'width_mult': 0.5, 'depth_mult': 0.5},
    neck_type='fpn',
    neck_out_channels=128,
)
def count_params(module, name=""):
    total = sum(p.numel() for p in module.parameters())
    if name:
        print(f"  {name}: {total:,}")
    return total

print("L_Refine_YOLO component breakdown:")
b_params = count_params(m1.backbone, "Backbone")
asfb_params = count_params(m1.asfb, "ASFB")
neck_params = count_params(m1.neck, "Neck")
head_params = count_params(m1.head, "Head")
total = count_params(m1)
print(f"  Total: {total:,}")
print(f"  Breakdown: Backbone {b_params:,} + ASFB {asfb_params:,} + Neck {neck_params:,} + Head {head_params:,}")

# 原始 Baseline (标准宽度)
print("\n=== Baseline CSPBackbone (Standard) ===")
def load_config_recursive(config_path):
    config = load_config(config_path)
    if '_extends' in config:
        base_path = os.path.join(os.path.dirname(config_path), config['_extends'])
        base_config = load_config_recursive(base_path)
        config = merge_configs(base_config, config)
    return config

config = load_config_recursive('configs/baseline.yaml')
m2 = build_model(config=config, num_classes=3, img_size=640)
print(f"Total params: {count_params(m2):,}")
