"""
Feature Fusion Neck (FPN/PAN) for Multi-scale Detection.

Fuses multi-scale features from backbone for better small object detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class ConvBNSiLU(nn.Module):
    """Standard convolution block: Conv -> BN -> SiLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = None,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FPNNeck(nn.Module):
    """
    Feature Pyramid Network (FPN) neck.
    
    Top-down pathway that fuses high-level semantic features with
    low-level detailed features for multi-scale detection.
    
    Input scales: P3 (1/8), P4 (1/16), P5 (1/32)
    Output scales: P3, P4, P5 (all with same channel dimensions)
    """
    
    def __init__(
        self,
        in_channels: Dict[str, int],  # {'P3': 256, 'P4': 512, 'P5': 1024}
        out_channels: int = 256,  # Unified output channels
    ):
        super().__init__()
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral_p5 = ConvBNSiLU(in_channels['P5'], out_channels, 1)
        self.lateral_p4 = ConvBNSiLU(in_channels['P4'], out_channels, 1)
        self.lateral_p3 = ConvBNSiLU(in_channels['P3'], out_channels, 1)
        
        # Top-down fusions
        self.fusion_p4 = ConvBNSiLU(out_channels * 2, out_channels, 3)
        self.fusion_p3 = ConvBNSiLU(out_channels * 2, out_channels, 3)
        
        # Output projections
        self.output_p5 = ConvBNSiLU(out_channels, out_channels, 3)
        self.output_p4 = ConvBNSiLU(out_channels, out_channels, 3)
        self.output_p3 = ConvBNSiLU(out_channels, out_channels, 3)
        
        self.out_channels = out_channels
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Dictionary with keys 'P3', 'P4', 'P5'
        
        Returns:
            Dictionary with fused features at P3, P4, P5 scales
        """
        p3, p4, p5 = features['P3'], features['P4'], features['P5']
        
        # Lateral connections
        p5_lat = self.lateral_p5(p5)  # P5: 1/32
        p4_lat = self.lateral_p4(p4)  # P4: 1/16
        p3_lat = self.lateral_p3(p3)  # P3: 1/8
        
        # Top-down pathway (start from P5, fuse down)
        # P5 -> P4
        p5_up = F.interpolate(p5_lat, size=p4_lat.shape[2:], mode='nearest')
        p4_fused = torch.cat([p4_lat, p5_up], dim=1)
        p4_out = self.fusion_p4(p4_fused)
        
        # P4 -> P3
        p4_up = F.interpolate(p4_out, size=p3_lat.shape[2:], mode='nearest')
        p3_fused = torch.cat([p3_lat, p4_up], dim=1)
        p3_out = self.fusion_p3(p3_fused)
        
        # Output projections
        p5_out = self.output_p5(p5_lat)
        p4_out = self.output_p4(p4_out)
        p3_out = self.output_p3(p3_out)
        
        return {
            'P3': p3_out,
            'P4': p4_out,
            'P5': p5_out,
        }
    
    def get_out_channels(self) -> int:
        """Return output channel size."""
        return self.out_channels


class PANNeck(nn.Module):
    """
    Path Aggregation Network (PAN) neck.
    
    Extends FPN with bottom-up pathway for better feature propagation.
    Better for small object detection compared to FPN alone.
    
    Input scales: P3 (1/8), P4 (1/16), P5 (1/32)
    Output scales: P3, P4, P5 (all with same channel dimensions)
    """
    
    def __init__(
        self,
        in_channels: Dict[str, int],  # {'P3': 256, 'P4': 512, 'P5': 1024}
        out_channels: int = 256,  # Unified output channels
    ):
        super().__init__()
        
        # Lateral connections (top-down path)
        self.lateral_p5 = ConvBNSiLU(in_channels['P5'], out_channels, 1)
        self.lateral_p4 = ConvBNSiLU(in_channels['P4'], out_channels, 1)
        self.lateral_p3 = ConvBNSiLU(in_channels['P3'], out_channels, 1)
        
        # Top-down fusions
        self.fusion_p4_top = ConvBNSiLU(out_channels * 2, out_channels, 3)
        self.fusion_p3_top = ConvBNSiLU(out_channels * 2, out_channels, 3)
        
        # Bottom-up fusions (PAN-specific)
        self.fusion_p4_bottom = ConvBNSiLU(out_channels * 2, out_channels, 3)
        self.fusion_p5_bottom = ConvBNSiLU(out_channels * 2, out_channels, 3)
        
        # Output projections
        self.output_p3 = ConvBNSiLU(out_channels, out_channels, 3)
        self.output_p4 = ConvBNSiLU(out_channels, out_channels, 3)
        self.output_p5 = ConvBNSiLU(out_channels, out_channels, 3)
        
        self.out_channels = out_channels
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Dictionary with keys 'P3', 'P4', 'P5'
        
        Returns:
            Dictionary with fused features at P3, P4, P5 scales
        """
        p3, p4, p5 = features['P3'], features['P4'], features['P5']
        
        # ===== Top-down pathway (FPN-style) =====
        p5_lat = self.lateral_p5(p5)
        p4_lat = self.lateral_p4(p4)
        p3_lat = self.lateral_p3(p3)
        
        # P5 -> P4
        p5_up = F.interpolate(p5_lat, size=p4_lat.shape[2:], mode='nearest')
        p4_top = self.fusion_p4_top(torch.cat([p4_lat, p5_up], dim=1))
        
        # P4 -> P3
        p4_up = F.interpolate(p4_top, size=p3_lat.shape[2:], mode='nearest')
        p3_top = self.fusion_p3_top(torch.cat([p3_lat, p4_up], dim=1))
        
        # ===== Bottom-up pathway (PAN-specific) =====
        # P3 -> P4
        p3_down = F.max_pool2d(p3_top, kernel_size=2, stride=2)
        p4_bottom = self.fusion_p4_bottom(torch.cat([p4_top, p3_down], dim=1))
        
        # P4 -> P5
        p4_down = F.max_pool2d(p4_bottom, kernel_size=2, stride=2)
        p5_bottom = self.fusion_p5_bottom(torch.cat([p5_lat, p4_down], dim=1))
        
        # Output projections
        p3_out = self.output_p3(p3_top)
        p4_out = self.output_p4(p4_bottom)
        p5_out = self.output_p5(p5_bottom)
        
        return {
            'P3': p3_out,
            'P4': p4_out,
            'P5': p5_out,
        }
    
    def get_out_channels(self) -> int:
        """Return output channel size."""
        return self.out_channels


def build_neck(
    neck_type: str = "fpn",
    in_channels: Dict[str, int] = None,
    out_channels: int = 256,
    **kwargs
) -> nn.Module:
    """
    Build neck model.
    
    Args:
        neck_type: Type of neck ("fpn" or "pan")
        in_channels: Dictionary of input channels per scale
        out_channels: Output channels for each scale
        **kwargs: Additional arguments
    
    Returns:
        Neck model
    """
    if in_channels is None:
        raise ValueError("in_channels must be provided")
    
    if neck_type.lower() == "fpn":
        return FPNNeck(in_channels=in_channels, out_channels=out_channels)
    elif neck_type.lower() == "pan":
        return PANNeck(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError(f"Unknown neck type: {neck_type}")

