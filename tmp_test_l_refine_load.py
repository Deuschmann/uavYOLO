import sys, os
sys.path.insert(0, os.path.abspath('.'))
import torch
from src.models import build_model
from src.utils.config import load_config, merge_configs

# Load l_refine_yolo config
def load_config_recursive(config_path):
    config = load_config(config_path)
    if '_extends' in config:
        base_path = os.path.join(os.path.dirname(config_path), config['_extends'])
        base_config = load_config_recursive(base_path)
        config = merge_configs(base_config, config)
    return config

config = load_config_recursive('configs/l_refine_yolo.yaml')
print('Config loaded. Model type:', config['model'].get('type', 'yolo'))

model = build_model(config, num_classes=3, img_size=640)
print('Model built:', type(model).__name__)

x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    preds = model(x)

print('Predictions keys:', preds.keys())
for scale in ['P3', 'P4', 'P5']:
    if scale in preds:
        print(f"{scale} bbox shape:", preds[scale]['bbox'].shape)
