"""
IRDCB - Improved Reversed Inverted Residual (Expand -> Depthwise x2 -> Compress -> Residual)

Steps:
1) Expand: 1x1 conv expand c1 -> c1 * t
2) Filter: DWConv3x3 -> BN -> SiLU -> DWConv3x3 -> BN -> SiLU
3) Compress: 1x1 conv project back to c2
4) Residual: add skip connection when c1==c2 and stride==1

"""
from typing import Optional

import torch
import torch.nn as nn


class IRDCB(nn.Module):
    """Improved reversed inverted residual block.

    Args:
        in_channels (int): input channels (c1)
        out_channels (int): output channels (c2)
        stride (int): stride for spatial downsampling (applied to first DW conv)
        expand_ratio (float): expansion factor t (default=2)
        norm_layer: normalization layer to use (default: nn.BatchNorm2d)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: float = 2.0,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(in_channels * expand_ratio)

        # Whether to use residual connection
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        # Expand: 1x1 conv (only if hidden_dim != in_channels)
        if hidden_dim != in_channels:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                norm_layer(hidden_dim),
                nn.SiLU(inplace=True),
            )
        else:
            # identity when no expansion required
            self.expand = nn.Identity()

        # Filter: two depthwise 3x3 convs with BN + SiLU
        self.dw = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            norm_layer(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=False),
            norm_layer(hidden_dim),
            nn.SiLU(inplace=True),
        )

        # Compress: project back to out_channels
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.dw(out)
        out = self.project(out)

        if self.use_res_connect:
            return x + out
        return out


__all__ = ["IRDCB"]
