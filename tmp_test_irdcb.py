import sys, os
sys.path.insert(0, os.path.abspath('.'))
import torch
from src.models.irdcb import IRDCB

def run_case(in_c,out_c,stride,expand):
    m=IRDCB(in_c,out_c,stride,expand)
    m.eval()
    x=torch.randn(2,in_c,64,64)
    with torch.no_grad():
        y=m(x)
    print(f"in={in_c}, out={out_c}, stride={stride}, expand={expand} -> out.shape={y.shape}")

if __name__ == '__main__':
    run_case(32,32,1,2.0)   # residual case
    run_case(32,64,1,2.0)   # no residual, same spatial
    run_case(32,32,2,2.0)   # downsample
    run_case(32,32,1,1.0)   # expand_ratio=1 (no expand conv)
