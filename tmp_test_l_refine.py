import sys, os
sys.path.insert(0, os.path.abspath('.'))
import torch
from src.models.l_refine_yolo import build_l_refine_yolo

m = build_l_refine_yolo(
    num_classes=1,
    backbone_kwargs={'in_channels':3,'width_mult':1.0,'depth_mult':0.5,'use_depthwise':False},
    neck_type='fpn',
    neck_out_channels=256,
)
print('Model built')
x = torch.randn(1,3,256,256)
with torch.no_grad():
    preds = m(x)
print('Pred keys:', preds.keys())
print('P3 bbox shape:', preds['P3']['bbox'].shape)
print('P4 bbox shape:', preds['P4']['bbox'].shape)
print('P5 bbox shape:', preds['P5']['bbox'].shape)
