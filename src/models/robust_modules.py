"""
Robustness Modules for Adverse Weather and Complex Backgrounds.

Optional modules that can be inserted into the model pipeline:
    - BackgroundSuppressionModule: Suppresses uniform backgrounds via attention
    - FrequencyEnhancementModule: Enhances small objects via frequency-domain processing
    - ConditionAwareModulation: Modulates features based on weather condition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


# ============================================================================
# Background Suppression Module
# ============================================================================

class BackgroundSuppressionModule(nn.Module):
    """
    Suppresses uniform background regions using spatial and channel attention.
    
    Uses attention mechanisms to down-weight large uniform background areas,
    helping the model focus on foreground objects (especially small ones).
    
    Design:
        - Channel attention: Identifies important feature channels
        - Spatial attention: Identifies important spatial locations
        - Residual connection: x = x + alpha * attention(x)
    
    Args:
        in_channels: Input feature channels
        reduction: Channel reduction ratio for efficiency (default: 16)
        alpha: Residual scaling factor (default: 0.1, small correction)
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        
        # Channel attention branch
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        
        # Spatial attention branch
        # Uses variance across channels to detect uniform regions
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1, in_channels // reduction, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (B, C, H, W)
        
        Returns:
            Enhanced features (B, C, H, W) = x + alpha * attention(x)
        """
        # Channel attention
        ca = self.channel_attention(x)  # (B, C, 1, 1)
        x_ca = x * ca
        
        # Spatial attention (compute variance across channels as input)
        # Uniform backgrounds have low variance, objects have high variance
        x_var = torch.var(x_ca, dim=1, keepdim=True)  # (B, 1, H, W)
        sa = self.spatial_attention(x_var)  # (B, 1, H, W)
        
        # Apply both attentions
        x_attended = x_ca * sa
        
        # Residual connection with small scaling
        out = x + self.alpha * x_attended
        
        return out


# ============================================================================
# Frequency Enhancement Module
# ============================================================================

class FrequencyEnhancementModule(nn.Module):
    """
    Lightweight frequency-domain enhancement for small object detection.
    
    Uses learnable high-pass filtering to emphasize fine-grained structures
    that help detect small objects under blur, haze, or noise.
    
    Design:
        - Learnable frequency filters (implemented as conv layers)
        - Emphasizes high-frequency components (edges, textures)
        - Residual connection: x = x + alpha * high_freq(x)
    
    Args:
        in_channels: Input feature channels
        kernel_size: Size of frequency filter kernel (default: 3)
        alpha: Residual scaling factor (default: 0.1)
    """
    
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        
        # Learnable high-pass filter (Laplacian-like)
        # This acts as a frequency-domain enhancement
        self.high_pass = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=in_channels,  # Depthwise for efficiency
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            # No activation - preserve both positive and negative responses
        )
        
        # Initialize filter as Laplacian (high-pass) kernel
        self._init_laplacian_kernel()
        
        # Optional: channel-wise scaling
        self.channel_scale = nn.Parameter(torch.ones(in_channels))
    
    def _init_laplacian_kernel(self):
        """Initialize high-pass filter with Laplacian-like kernel."""
        # Laplacian kernel: emphasizes edges and fine details
        kernel_size = self.high_pass[0].kernel_size[0]
        if kernel_size == 3:
            kernel = torch.tensor([
                [0.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 0.0],
            ], dtype=torch.float32)
        elif kernel_size == 5:
            kernel = torch.tensor([
                [0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0],
                [-1.0, -1.0, 8.0, -1.0, -1.0],
                [0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0],
            ], dtype=torch.float32)
        else:
            # Default: zero initialization (learn from scratch)
            kernel = torch.zeros(kernel_size, kernel_size)
        
        # Expand to match number of channels
        with torch.no_grad():
            for i, module in enumerate(self.high_pass):
                if isinstance(module, nn.Conv2d):
                    for j in range(module.out_channels):
                        # Normalize kernel
                        kernel_norm = kernel / (kernel.abs().sum() + 1e-8)
                        module.weight[j, 0] = kernel_norm
                    break
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (B, C, H, W)
        
        Returns:
            Enhanced features (B, C, H, W) = x + alpha * high_freq(x)
        """
        # Apply high-pass filter
        x_high_freq = self.high_pass(x)
        
        # Channel-wise scaling
        x_high_freq = x_high_freq * self.channel_scale.view(1, -1, 1, 1)
        
        # Residual connection with small scaling
        out = x + self.alpha * x_high_freq
        
        return out


# ============================================================================
# Condition-Aware Modulation Module
# ============================================================================

class ConditionPredictor(nn.Module):
    """
    Lightweight condition predictor (weather/lighting) from input image.
    
    Predicts a condition code that can be used to modulate features.
    """
    
    def __init__(
        self,
        in_channels: int = 3,  # RGB input
        condition_dim: int = 32,  # Dimension of condition code
    ):
        super().__init__()
        # Very lightweight - just a few conv layers + global pooling
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.predictor = nn.Sequential(
            nn.Linear(32, condition_dim),
            nn.ReLU(inplace=True),
            nn.Linear(condition_dim, condition_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input image (B, 3, H, W)
        
        Returns:
            Condition code (B, condition_dim)
        """
        x = self.encoder(x)
        x = x.flatten(1)
        condition = self.predictor(x)
        return condition


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    
    Modulates features using condition code:
        out = gamma(condition) * x + beta(condition)
    
    Args:
        in_channels: Input feature channels
        condition_dim: Dimension of condition code
    """
    
    def __init__(
        self,
        in_channels: int,
        condition_dim: int,
    ):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(condition_dim, in_channels),
            nn.Sigmoid(),  # Keep values in [0, 1]
        )
        self.beta_net = nn.Sequential(
            nn.Linear(condition_dim, in_channels),
            nn.Tanh(),  # Keep values in [-1, 1]
        )
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (B, C, H, W)
            condition: Condition code (B, condition_dim)
        
        Returns:
            Modulated features (B, C, H, W)
        """
        gamma = self.gamma_net(condition)  # (B, C)
        beta = self.beta_net(condition)  # (B, C)
        
        # Apply modulation
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        
        out = gamma * x + beta
        return out


class ConditionAwareModulation(nn.Module):
    """
    Condition-aware feature modulation using FiLM.
    
    Predicts weather/lighting condition from input image and uses it to
    modulate features at specific layers (e.g., after certain backbone stages).
    
    Design:
        - Condition predictor: Lightweight CNN to predict condition code
        - FiLM layers: Modulate features based on condition
        - Can be applied at multiple points in the network
    
    Args:
        in_channels: Input feature channels (for FiLM layers)
        condition_dim: Dimension of condition code (default: 32)
        use_residual: Use residual connection (default: True)
        alpha: Residual scaling if use_residual (default: 0.2)
    """
    
    def __init__(
        self,
        in_channels: int,
        condition_dim: int = 32,
        use_residual: bool = True,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.alpha = alpha
        
        # Condition predictor (shared across all FiLM layers)
        # Note: This will be created separately and passed in forward
        self.condition_dim = condition_dim
        
        # FiLM layer for this module
        self.film = FiLMLayer(in_channels, condition_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (B, C, H, W)
            condition: Condition code from ConditionPredictor (B, condition_dim)
        
        Returns:
            Modulated features (B, C, H, W)
        """
        # Apply FiLM modulation
        x_modulated = self.film(x, condition)
        
        # Residual connection
        if self.use_residual:
            out = x + self.alpha * x_modulated
        else:
            out = x_modulated
        
        return out


# ============================================================================
# Helper Functions
# ============================================================================

def build_background_suppression(
    in_channels: int,
    reduction: int = 16,
    alpha: float = 0.1,
) -> BackgroundSuppressionModule:
    """Build BackgroundSuppressionModule."""
    return BackgroundSuppressionModule(
        in_channels=in_channels,
        reduction=reduction,
        alpha=alpha,
    )


def build_frequency_enhancement(
    in_channels: int,
    kernel_size: int = 3,
    alpha: float = 0.1,
) -> FrequencyEnhancementModule:
    """Build FrequencyEnhancementModule."""
    return FrequencyEnhancementModule(
        in_channels=in_channels,
        kernel_size=kernel_size,
        alpha=alpha,
    )


def build_condition_aware_modulation(
    in_channels: int,
    condition_dim: int = 32,
    use_residual: bool = True,
    alpha: float = 0.2,
) -> ConditionAwareModulation:
    """Build ConditionAwareModulation."""
    return ConditionAwareModulation(
        in_channels=in_channels,
        condition_dim=condition_dim,
        use_residual=use_residual,
        alpha=alpha,
    )


def build_condition_predictor(
    condition_dim: int = 32,
) -> ConditionPredictor:
    """Build ConditionPredictor."""
    return ConditionPredictor(condition_dim=condition_dim)

