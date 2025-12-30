"""
ASFB - Adaptive Scale Fusion Block (单模态简化实现)

Components:
  - CRU: 内容重构上采样 (1x1 conv 压缩 -> bilinear upsample -> 3x3 DWConv 精修)
  - GAD: 几何对齐下采样 (3x3 conv stride=2 -> 1x1 conv)
  - AWF: 自适应加权融合 (可学习标量权重 w1,w2,w3)

输入: deep (小分辨率), mid (中分辨率), shallow (大分辨率)
输出: 融合后的特征（与 mid 分辨率一致，通道为 out_channels）
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRU(nn.Module):
    """Content Reconstruction Upsample (工程近似实现)

    1x1 conv -> bilinear upsample (x2) -> 3x3 depthwise conv -> BN -> SiLU
    """

    def __init__(self, in_channels: int, out_channels: int, norm_layer: Optional[nn.Module] = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.SiLU(inplace=True),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False),
            norm_layer(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compress(x)
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.refine(x)
        return x


class GAD(nn.Module):
    """Geometric Alignment Downsample (工程近似实现)

    3x3 conv stride=2 -> BN -> SiLU -> 1x1 conv -> BN -> SiLU
    """

    def __init__(self, in_channels: int, out_channels: int, norm_layer: Optional[nn.Module] = None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class AWF(nn.Module):
    """Adaptive Weight Fusion: learnable scalar weights for three inputs."""

    def __init__(self, init_val: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(float(init_val)))
        self.w2 = nn.Parameter(torch.tensor(float(init_val)))
        self.w3 = nn.Parameter(torch.tensor(float(init_val)))
        self.eps = eps

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # a,b,c are expected to have same shape
        w1 = torch.relu(self.w1)
        w2 = torch.relu(self.w2)
        w3 = torch.relu(self.w3)
        denom = w1 + w2 + w3 + self.eps
        out = (w1 * a + w2 * b + w3 * c) / denom
        return out


class ASFB(nn.Module):
    """Adaptive Scale Fusion Block (单模态简化版)

    Args:
        deep_channels: channels of deep (small) feature
        mid_channels: channels of mid feature
        shallow_channels: channels of shallow (large) feature
        out_channels: desired output channels (defaults to mid_channels)
    """

    def __init__(
        self,
        deep_channels: int,
        mid_channels: int,
        shallow_channels: int,
        out_channels: Optional[int] = None,
        norm_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = mid_channels
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # CRU: refine deep -> upsample to mid
        self.cru = CRU(deep_channels, out_channels, norm_layer=norm_layer)

        # Project mid to out_channels if needed
        if mid_channels != out_channels:
            self.mid_proj = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
                norm_layer(out_channels),
                nn.SiLU(inplace=True),
            )
        else:
            self.mid_proj = nn.Identity()

        # GAD: downsample shallow -> to mid resolution
        self.gad = GAD(shallow_channels, out_channels, norm_layer=norm_layer)

        # AWF: adaptive weighted fusion
        self.awf = AWF(init_val=1.0, eps=1e-6)

    def forward(self, deep: torch.Tensor, mid: torch.Tensor, shallow: torch.Tensor) -> torch.Tensor:
        """Expect shapes:
        deep: (B, C_d, H/4, W/4)
        mid: (B, C_m, H/2, W/2)
        shallow: (B, C_s, H, W)
        Returns fused feature at mid resolution (B, out_channels, H/2, W/2)
        """
        deep_ref = self.cru(deep)           # upsampled to mid
        mid_proj = self.mid_proj(mid)       # projected to out_channels
        shallow_ref = self.gad(shallow)     # downsampled to mid

        out = self.awf(deep_ref, mid_proj, shallow_ref)
        return out


__all__ = ["ASFB", "CRU", "GAD", "AWF"]
