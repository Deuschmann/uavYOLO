"""
YOLO Model - Main wrapper combining backbone, neck, and head.

Provides a clean interface for training and inference.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union

from .backbone import build_backbone
from .neck import build_neck
from .head import build_head
from .robust_modules import (
    build_background_suppression,
    build_frequency_enhancement,
    build_condition_aware_modulation,
    build_condition_predictor,
)


class YOLOModel(nn.Module):
    """
    Complete YOLO-style detection model.
    
    Architecture: Backbone -> Neck -> Head
        - Backbone: Extracts multi-scale features (P3, P4, P5)
        - Neck: Fuses features across scales (FPN/PAN)
        - Head: Predicts bboxes, objectness, classes at each scale
    
    Args:
        backbone: Backbone model (or config dict)
        neck: Neck model (or config dict)
        head: Head model (or config dict)
        img_size: Input image size (for info/debugging)
    """
    
    def __init__(
        self,
        backbone: Union[nn.Module, Dict],
        neck: Union[nn.Module, Dict],
        head: Union[nn.Module, Dict],
        img_size: int = 640,
        robust_modules_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.img_size = img_size
        robust_modules_config = robust_modules_config or {}
        
        # Build or use provided backbone
        if isinstance(backbone, dict):
            self.backbone = build_backbone(**backbone)
        else:
            self.backbone = backbone
        
        # Get backbone output channels
        backbone_out_channels = self.backbone.get_out_channels()
        
        # Build robustness modules for backbone stages (optional)
        self.robust_modules_backbone = nn.ModuleDict()
        self.use_background_suppression = robust_modules_config.get('use_background_suppression', False)
        self.use_frequency_enhancement = robust_modules_config.get('use_frequency_enhancement', False)
        self.use_condition_aware = robust_modules_config.get('use_condition_aware', False)
        
        if self.use_background_suppression or self.use_frequency_enhancement or self.use_condition_aware:
            # Insert modules after each backbone stage (P3, P4, P5)
            for scale in ['P3', 'P4', 'P5']:
                modules = nn.ModuleList()
                ch = backbone_out_channels[scale]
                
                if self.use_background_suppression:
                    bg_cfg = robust_modules_config.get('background_suppression', {})
                    modules.append(build_background_suppression(
                        in_channels=ch,
                        reduction=bg_cfg.get('reduction', 16),
                        alpha=bg_cfg.get('alpha', 0.1),
                    ))
                
                if self.use_frequency_enhancement:
                    freq_cfg = robust_modules_config.get('frequency_enhancement', {})
                    modules.append(build_frequency_enhancement(
                        in_channels=ch,
                        kernel_size=freq_cfg.get('kernel_size', 3),
                        alpha=freq_cfg.get('alpha', 0.1),
                    ))
                
                if modules:
                    self.robust_modules_backbone[scale] = modules
        
        # Condition predictor (if condition-aware modulation is used)
        self.condition_predictor = None
        if self.use_condition_aware:
            cond_cfg = robust_modules_config.get('condition_aware', {})
            condition_dim = cond_cfg.get('condition_dim', 32)
            self.condition_predictor = build_condition_predictor(condition_dim=condition_dim)
            
            # Condition-aware modules for backbone stages
            if not hasattr(self, 'condition_modules_backbone'):
                self.condition_modules_backbone = nn.ModuleDict()
            for scale in ['P3', 'P4', 'P5']:
                ch = backbone_out_channels[scale]
                self.condition_modules_backbone[scale] = build_condition_aware_modulation(
                    in_channels=ch,
                    condition_dim=condition_dim,
                    use_residual=cond_cfg.get('use_residual', True),
                    alpha=cond_cfg.get('alpha', 0.2),
                )
        
        # Build or use provided neck
        if isinstance(neck, dict):
            # Add backbone output channels to neck config
            neck['in_channels'] = backbone_out_channels
            self.neck = build_neck(**neck)
        else:
            self.neck = neck
        
        # Get neck output channels
        neck_out_channels = self.neck.get_out_channels()
        
        # Condition-aware modules for neck (optional)
        self.condition_modules_neck = None
        if self.use_condition_aware and self.condition_predictor is not None:
            cond_cfg = robust_modules_config.get('condition_aware', {})
            condition_dim = cond_cfg.get('condition_dim', 32)
            self.condition_modules_neck = build_condition_aware_modulation(
                in_channels=neck_out_channels,
                condition_dim=condition_dim,
                use_residual=cond_cfg.get('use_residual', True),
                alpha=cond_cfg.get('alpha', 0.2),
            )
        
        # Build or use provided head
        if isinstance(head, dict):
            # Add neck output channels to head config
            head['in_channels'] = neck_out_channels
            self.head = build_head(**head)
        else:
            self.head = head
    
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[Dict] = None,
        mode: str = 'train'
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Input images (B, C, H, W), normalized [0, 1]
            targets: Optional targets dict (for loss computation in training)
                - 'boxes': List of tensors, each (N, 4) in YOLO format
                - 'labels': List of tensors, each (N,) class indices
                - 'image_ids': List of image IDs
            mode: 'train' or 'val' / 'inference'
        
        Returns:
            If mode == 'train':
                Dictionary with predictions at each scale:
                    {
                        'P3': {'bbox': (B, 4, H3, W3), 'obj': (B, 1, H3, W3), 'cls': (B, num_classes, H3, W3)},
                        'P4': {...},
                        'P5': {...}
                    }
            If mode == 'val' / 'inference':
                Same format, ready for post-processing (NMS, etc.)
        """
        # Predict condition code if condition-aware modulation is enabled
        condition = None
        if self.use_condition_aware and self.condition_predictor is not None:
            condition = self.condition_predictor(images)
        
        # Backbone: extract multi-scale features
        backbone_features = self.backbone(images)
        
        # Apply robustness modules to backbone features
        if self.use_background_suppression or self.use_frequency_enhancement or self.use_condition_aware:
            for scale in ['P3', 'P4', 'P5']:
                feat = backbone_features[scale]
                
                # Apply background suppression and frequency enhancement
                if scale in self.robust_modules_backbone:
                    for module in self.robust_modules_backbone[scale]:
                        feat = module(feat)
                
                # Apply condition-aware modulation
                if self.use_condition_aware and scale in self.condition_modules_backbone:
                    feat = self.condition_modules_backbone[scale](feat, condition)
                
                backbone_features[scale] = feat
        
        # Neck: fuse features across scales
        neck_features = self.neck(backbone_features)
        
        # Apply condition-aware modulation to neck features (optional)
        if self.use_condition_aware and self.condition_modules_neck is not None:
            for scale in ['P3', 'P4', 'P5']:
                neck_features[scale] = self.condition_modules_neck(neck_features[scale], condition)
        
        # Head: predict detections at each scale
        predictions = self.head(neck_features)
        
        return predictions
    
    def get_model_info(self) -> Dict:
        """Get model information (parameters, FLOPs, etc.)."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count parameters per component
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        neck_params = sum(p.numel() for p in self.neck.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        
        robust_params = 0
        if hasattr(self, 'robust_modules_backbone'):
            robust_params += sum(p.numel() for p in self.robust_modules_backbone.parameters())
        if hasattr(self, 'condition_modules_backbone'):
            robust_params += sum(p.numel() for p in self.condition_modules_backbone.parameters())
        if hasattr(self, 'condition_modules_neck') and self.condition_modules_neck is not None:
            robust_params += sum(p.numel() for p in self.condition_modules_neck.parameters())
        if self.condition_predictor is not None:
            robust_params += sum(p.numel() for p in self.condition_predictor.parameters())
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'backbone_params': backbone_params,
            'neck_params': neck_params,
            'head_params': head_params,
            'robust_params': robust_params,
            'img_size': self.img_size,
        }


def build_model(
    config: Dict,
    num_classes: Optional[int] = None,
    img_size: int = 640,
) -> YOLOModel:
    """
    Build complete YOLO model from config.
    
    Args:
        config: Configuration dictionary with 'model' section
        num_classes: Number of classes (overrides config if provided)
        img_size: Input image size
    
    Returns:
        YOLOModel instance or L_Refine_YOLO instance depending on config['model']['type']
    """
    model_cfg = config.get('model', {})
    model_type = model_cfg.get('type', 'yolo').lower()
    
    # If using L_Refine_YOLO model type, import and use it
    if model_type == 'l_refine_yolo':
        from .l_refine_yolo import build_l_refine_yolo
        backbone_cfg = model_cfg.get('backbone', {})
        if isinstance(backbone_cfg, dict):
            backbone_cfg = backbone_cfg.copy()
            backbone_cfg.setdefault('in_channels', 3)
        
        head_kwargs = model_cfg.get('head', {})
        if isinstance(head_kwargs, dict):
            head_kwargs = head_kwargs.copy()
            head_kwargs.setdefault('num_layers', 2)
            head_kwargs.setdefault('hidden_channels', None)
            head_kwargs.setdefault('share_weights', False)
        
        return build_l_refine_yolo(
            num_classes=num_classes or config.get('classes', {}).get('num_classes', 1),
            backbone_kwargs=backbone_cfg,
            neck_type=model_cfg.get('neck', {}).get('type', 'fpn'),
            neck_out_channels=model_cfg.get('neck', {}).get('out_channels', 256),
            head_kwargs=head_kwargs,
        )
    
    # Default: use standard YOLOModel
    # Backbone config
    backbone_cfg = model_cfg.get('backbone', {})
    if isinstance(backbone_cfg, dict):
        backbone_cfg = backbone_cfg.copy()
        backbone_cfg.setdefault('in_channels', 3)
    
    # Neck config
    neck_cfg = model_cfg.get('neck', {})
    if isinstance(neck_cfg, dict):
        neck_cfg = neck_cfg.copy()
        neck_cfg.setdefault('type', 'fpn')
        neck_cfg.setdefault('out_channels', 256)
    
    # Head config
    head_cfg = model_cfg.get('head', {})
    if isinstance(head_cfg, dict):
        head_cfg = head_cfg.copy()
        head_cfg.setdefault('type', 'anchor_free')
        if num_classes is not None:
            head_cfg['num_classes'] = num_classes
        elif 'num_classes' not in head_cfg:
            # Try to get from config
            head_cfg['num_classes'] = config.get('classes', {}).get('num_classes', 1)
        head_cfg.setdefault('num_layers', 2)
        head_cfg.setdefault('share_weights', False)
    
    # Robustness modules config
    robust_modules_config = config.get('robust_modules', {})
    
    model = YOLOModel(
        backbone=backbone_cfg,
        neck=neck_cfg,
        head=head_cfg,
        img_size=img_size,
        robust_modules_config=robust_modules_config,
    )
    
    return model

