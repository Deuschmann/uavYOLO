"""
Detection Head for YOLO-style Object Detection.

Anchor-free head that predicts bounding boxes, objectness, and class logits
at multiple scales.
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
        kernel_size: int = 3,
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


class DetectionHead(nn.Module):
    """
    Anchor-free detection head.
    
    Predicts at each spatial location:
        - Bounding box (4 params: x_center, y_center, width, height) - normalized
        - Objectness score (1 param)
        - Class logits (num_classes params)
    
    Args:
        in_channels: Input channel size from neck (same for all scales)
        num_classes: Number of object classes
        num_layers: Number of convolution layers in head (default: 2)
        hidden_channels: Hidden channel size (default: same as in_channels)
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_layers: int = 2,
        hidden_channels: Optional[int] = None,
    ):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        
        # Shared stem (common feature extraction)
        layers = []
        for i in range(num_layers):
            layers.append(ConvBNSiLU(
                in_channels if i == 0 else hidden_channels,
                hidden_channels,
                kernel_size=3,
            ))
        self.stem = nn.Sequential(*layers)
        
        # Prediction branches
        # Bbox branch: predicts (x_center, y_center, width, height) - all normalized
        self.bbox_conv = nn.Conv2d(hidden_channels, 4, kernel_size=1, bias=True)
        
        # Objectness branch: predicts object presence probability
        self.obj_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)
        
        # Classification branch: predicts class logits
        self.cls_conv = nn.Conv2d(hidden_channels, num_classes, kernel_size=1, bias=True)
        
        # Initialize bias for objectness branch (helps with training stability)
        # Bias objectness to low values initially (e.g., -4.6 ≈ sigmoid(0.01))
        nn.init.constant_(self.obj_conv.bias, -4.6)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input feature map (B, C, H, W)
        
        Returns:
            Dictionary with:
                - 'bbox': (B, 4, H, W) - bounding box parameters (normalized)
                - 'obj': (B, 1, H, W) - objectness logits
                - 'cls': (B, num_classes, H, W) - class logits
        """
        x = self.stem(x)
        
        bbox = self.bbox_conv(x)  # (B, 4, H, W)
        obj = self.obj_conv(x)  # (B, 1, H, W)
        cls = self.cls_conv(x)  # (B, num_classes, H, W)
        
        return {
            'bbox': bbox,
            'obj': obj,
            'cls': cls,
        }


class MultiScaleDetectionHead(nn.Module):
    """
    Multi-scale detection head for P3, P4, P5 scales.
    
    Creates separate heads for each scale or shares weights (configurable).
    
    Args:
        in_channels: Input channel size from neck (same for all scales)
        num_classes: Number of object classes
        num_layers: Number of convolution layers per head
        hidden_channels: Hidden channel size
        share_weights: If True, use shared head for all scales (more efficient)
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_layers: int = 2,
        hidden_channels: Optional[int] = None,
        share_weights: bool = False,
    ):
        super().__init__()
        self.share_weights = share_weights
        
        if share_weights:
            # Single shared head for all scales (more memory efficient)
            self.head = DetectionHead(
                in_channels=in_channels,
                num_classes=num_classes,
                num_layers=num_layers,
                hidden_channels=hidden_channels,
            )
        else:
            # Separate heads for each scale (more flexible)
            self.head_p3 = DetectionHead(in_channels, num_classes, num_layers, hidden_channels)
            self.head_p4 = DetectionHead(in_channels, num_classes, num_layers, hidden_channels)
            self.head_p5 = DetectionHead(in_channels, num_classes, num_layers, hidden_channels)
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass on multi-scale features.
        
        Args:
            features: Dictionary with keys 'P3', 'P4', 'P5', each (B, C, H, W)
        
        Returns:
            Dictionary with keys 'P3', 'P4', 'P5', each containing:
                - 'bbox': (B, 4, H, W)
                - 'obj': (B, 1, H, W)
                - 'cls': (B, num_classes, H, W)
        """
        if self.share_weights:
            return {
                'P3': self.head(features['P3']),
                'P4': self.head(features['P4']),
                'P5': self.head(features['P5']),
            }
        else:
            return {
                'P3': self.head_p3(features['P3']),
                'P4': self.head_p4(features['P4']),
                'P5': self.head_p5(features['P5']),
            }


def build_head(
    head_type: str = "anchor_free",
    in_channels: int = 256,
    num_classes: int = 1,
    num_layers: int = 2,
    hidden_channels: Optional[int] = None,
    share_weights: bool = False,
    **kwargs
) -> nn.Module:
    """
    Build detection head.
    
    Args:
        head_type: Type of head ("anchor_free" only for now)
        in_channels: Input channel size
        num_classes: Number of classes
        num_layers: Number of layers in head
        hidden_channels: Hidden channel size
        share_weights: Share weights across scales
        **kwargs: Additional arguments
    
    Returns:
        Detection head model
    """
    if head_type.lower() == "anchor_free":
        return MultiScaleDetectionHead(
            in_channels=in_channels,
            num_classes=num_classes,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            share_weights=share_weights,
        )
    else:
        raise ValueError(f"Unknown head type: {head_type}")

