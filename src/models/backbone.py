"""
Lightweight CNN Backbone for UAV Object Detection.

Implements CSP-style blocks optimized for small object detection with
multi-scale feature extraction (P3, P4, P5).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class ConvBNSiLU(nn.Module):
    """Standard convolution block: Conv -> BN -> SiLU (Swish activation)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # Swish activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for lightweight design."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        # Depthwise convolution
        self.dw_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dw_bn(self.dw_conv(x)))
        x = self.act(self.pw_bn(self.pw_conv(x)))
        return x


class CSPBlock(nn.Module):
    """
    Cross Stage Partial (CSP) block.
    Splits input into two paths and concatenates them.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        use_depthwise: bool = False,
        expansion: float = 0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        
        # Main path (shortcut)
        self.conv1 = ConvBNSiLU(in_channels, hidden_channels, 1)
        
        # Sub-path (with residual blocks)
        self.conv2 = ConvBNSiLU(in_channels, hidden_channels, 1)
        
        # Residual blocks in sub-path
        blocks = []
        ConvBlock = DepthwiseSeparableConv if use_depthwise else ConvBNSiLU
        
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    ConvBlock(hidden_channels, hidden_channels, 3),
                    ConvBlock(hidden_channels, hidden_channels, 3),
                )
            )
        self.blocks = nn.Sequential(*blocks)
        
        # Final conv
        self.conv3 = ConvBNSiLU(hidden_channels * 2, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into two paths
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        # Process sub-path
        x2 = self.blocks(x2)
        
        # Concatenate and fuse
        out = torch.cat([x1, x2], dim=1)
        return self.conv3(out)


class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast - for better receptive field."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBNSiLU(in_channels, hidden_channels, 1)
        self.conv2 = ConvBNSiLU(hidden_channels * 4, out_channels, 1)
        self.maxpool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x1 = self.maxpool(x)
        x2 = self.maxpool(x1)
        x3 = self.maxpool(x2)
        return self.conv2(torch.cat([x, x1, x2, x3], dim=1))


class CSPBackbone(nn.Module):
    """
    Lightweight CSP-style backbone for UAV object detection.
    
    Produces multi-scale feature maps:
        - P3: 1/8 of input resolution (for small objects)
        - P4: 1/16 of input resolution
        - P5: 1/32 of input resolution
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        width_mult: Width multiplier for scaling model size (0.5, 0.75, 1.0, etc.)
        depth_mult: Depth multiplier for scaling model depth
        use_depthwise: Use depthwise separable convolutions for further efficiency
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        use_depthwise: bool = False,
    ):
        super().__init__()
        
        def make_divisible(channels: int, divisor: int = 8) -> int:
            """Make channels divisible by divisor for efficient hardware usage."""
            return int((channels * width_mult + divisor // 2) // divisor) * divisor
        
        # Stem
        base_channels = [64, 128, 256, 512, 1024]
        channels = [make_divisible(c) for c in base_channels]
        
        self.stem = ConvBNSiLU(in_channels, channels[0], kernel_size=6, stride=2, padding=2)
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBNSiLU(channels[0], channels[1], kernel_size=3, stride=2),
            CSPBlock(channels[1], channels[1], num_blocks=int(3 * depth_mult), use_depthwise=use_depthwise),
        )
        
        # Stage 2 (P3 output: 1/8 resolution)
        self.stage2 = nn.Sequential(
            ConvBNSiLU(channels[1], channels[2], kernel_size=3, stride=2),
            CSPBlock(channels[2], channels[2], num_blocks=int(6 * depth_mult), use_depthwise=use_depthwise),
        )
        
        # Stage 3 (P4 output: 1/16 resolution)
        self.stage3 = nn.Sequential(
            ConvBNSiLU(channels[2], channels[3], kernel_size=3, stride=2),
            CSPBlock(channels[3], channels[3], num_blocks=int(6 * depth_mult), use_depthwise=use_depthwise),
        )
        
        # Stage 4 (P5 output: 1/32 resolution)
        self.stage4 = nn.Sequential(
            ConvBNSiLU(channels[3], channels[4], kernel_size=3, stride=2),
            CSPBlock(channels[4], channels[4], num_blocks=int(3 * depth_mult), use_depthwise=use_depthwise),
            SPPF(channels[4], channels[4]),
        )
        
        self.out_channels = {
            'P3': channels[2],
            'P4': channels[3],
            'P5': channels[4],
        }
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Dictionary with keys 'P3', 'P4', 'P5' containing feature maps
        """
        x = self.stem(x)  # 1/2
        x = self.stage1(x)  # 1/4
        
        p3 = self.stage2(x)  # 1/8
        p4 = self.stage3(p3)  # 1/16
        p5 = self.stage4(p4)  # 1/32
        
        return {
            'P3': p3,
            'P4': p4,
            'P5': p5,
        }
    
    def get_out_channels(self) -> Dict[str, int]:
        """Return output channel sizes for each scale."""
        return self.out_channels


def build_backbone(
    backbone_type: str = "csp",
    in_channels: int = 3,
    width_mult: float = 1.0,
    depth_mult: float = 1.0,
    **kwargs
) -> nn.Module:
    """
    Build backbone model.
    
    Args:
        backbone_type: Type of backbone ("csp" only for now)
        in_channels: Number of input channels
        width_mult: Width multiplier
        depth_mult: Depth multiplier
        **kwargs: Additional arguments
    
    Returns:
        Backbone model
    """
    if backbone_type.lower() == "csp":
        return CSPBackbone(
            in_channels=in_channels,
            width_mult=width_mult,
            depth_mult=depth_mult,
            use_depthwise=kwargs.get('use_depthwise', False),
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")

