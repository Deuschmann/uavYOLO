import sys, os
sys.path.insert(0, os.path.abspath('.'))
import torch
from src.models.backbone import CSPBackbone

# Build backbone with use_depthwise True to exercise IRDCB path
m = CSPBackbone(in_channels=3, width_mult=1.0, depth_mult=0.1, use_depthwise=True)
print('Backbone built. Out channels:', m.get_out_channels())
x = torch.randn(1,3,256,256)
with torch.no_grad():
    feats = m(x)
for k,v in feats.items():
    print(k, v.shape)
