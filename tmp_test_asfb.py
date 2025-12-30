import sys, os
sys.path.insert(0, os.path.abspath('.'))
import torch
from src.models.asfb import ASFB

# Example shapes: shallow (1,128,64,64), mid (1,256,32,32), deep (1,512,16,16)
shallow = torch.randn(1,128,64,64)
mid = torch.randn(1,256,32,32)
deep = torch.randn(1,512,16,16)

m = ASFB(deep_channels=512, mid_channels=256, shallow_channels=128, out_channels=256)
with torch.no_grad():
    out = m(deep, mid, shallow)
print('out.shape =', out.shape)
