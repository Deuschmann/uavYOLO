"""
L_Refine_YOLO - Assembled model using modified Backbone and Neck with ASFB and IRDCB

This module builds a YOLO-style model where:
 - Backbone is based on `CSPBackbone` but replaces stride-2 Conv blocks with `LDown`
   and replaces deep-stage CSPBlock internals with `IRDCB`.
 - ASFB is applied to backbone outputs (P3, P4, P5) producing a fused mid-level feature.
 - The fused feature is used as the `P4` input to the neck; neck fusion layers are
   replaced with `IRDCB` modules.

The final detection head is the existing `build_head`.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CSPBackbone, ConvBNSiLU
from .irdcb import IRDCB
from .asfb import ASFB
from .neck import build_neck, FPNNeck, PANNeck
from .head import build_head


class LDown(nn.Module):
    """Lightweight downsample module replacing stride-2 Conv blocks.

    Implemented as: 3x3 Conv stride=2 -> BN -> SiLU
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


def replace_backbone_down_and_deep_blocks(backbone: CSPBackbone):
    """Modify a CSPBackbone in-place:
    - Replace all `ConvBNSiLU` with stride=2 by `LDown`.
    - For deep stages (stage3, stage4) replace `CSPBlock.blocks` with sequences of `IRDCB`.
    """
    # Replace stem if stride-2
    if isinstance(backbone.stem, ConvBNSiLU) and backbone.stem.conv.stride[0] == 2:
        in_c = backbone.stem.conv.in_channels
        out_c = backbone.stem.conv.out_channels
        backbone.stem = LDown(in_c, out_c)

    # Stages: stage1..stage4 each is nn.Sequential(ConvBNSiLU(stride=2), CSPBlock, ...)
    for stage_name in ['stage1', 'stage2', 'stage3', 'stage4']:
        stage = getattr(backbone, stage_name)
        # replace first module if it's ConvBNSiLU with stride=2
        if len(stage) > 0 and isinstance(stage[0], ConvBNSiLU) and stage[0].conv.stride[0] == 2:
            in_c = stage[0].conv.in_channels
            out_c = stage[0].conv.out_channels
            stage[0] = LDown(in_c, out_c)

    # Replace deep CSPBlock internals (stage3 -> P4, stage4 -> P5)
    for deep_stage in ['stage3', 'stage4']:
        stage = getattr(backbone, deep_stage)
        # CSPBlock is expected at index 1
        if len(stage) > 1:
            csp = stage[1]
            if hasattr(csp, 'blocks'):
                # Determine number of blocks and hidden channels
                try:
                    num_blocks = len(csp.blocks)
                except Exception:
                    num_blocks = 1
                # hidden channels from conv1 output
                hidden = csp.conv1.conv.out_channels
                new_blocks = []
                for _ in range(num_blocks):
                    new_blocks.append(IRDCB(hidden, hidden, stride=1, expand_ratio=2.0))
                csp.blocks = nn.Sequential(*new_blocks)


class L_Refine_YOLO(nn.Module):
    """Main model class assembling modified backbone, ASFB, modified neck and head."""

    def __init__(
        self,
        num_classes: int = 1,
        backbone_kwargs: Optional[Dict] = None,
        neck_type: str = 'fpn',
        neck_out_channels: int = 256,
        head_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        backbone_kwargs = backbone_kwargs or {}
        head_kwargs = head_kwargs or {}
        
        # Remove 'type' from backbone_kwargs if present
        backbone_kwargs = backbone_kwargs.copy()
        backbone_kwargs.pop('type', None)

        # Build base CSPBackbone
        self.backbone = CSPBackbone(**backbone_kwargs)

        # Apply replacements: LDown for stride-2 convs, IRDCB for deep CSPBlock internals
        replace_backbone_down_and_deep_blocks(self.backbone)

        # ASFB: inputs are (deep=P5, mid=P4, shallow=P3)
        out_ch = self.backbone.get_out_channels()
        self.asfb = ASFB(
            deep_channels=out_ch['P5'],
            mid_channels=out_ch['P4'],
            shallow_channels=out_ch['P3'],
            out_channels=out_ch['P4'],
        )

        # Build neck; initial in_channels mapping uses original P3,P4,P5 but P4 will be replaced by ASFB output
        in_ch = {
            'P3': out_ch['P3'],
            'P4': out_ch['P4'],  # ASFB will output same channels
            'P5': out_ch['P5'],
        }

        self.neck = build_neck(neck_type, in_channels=in_ch, out_channels=neck_out_channels)

        # Replace fusion convs in neck with IRDCB where applicable
        # For FPNNeck: fusion_p4, fusion_p3
        # For PANNeck: fusion_* attributes (top and bottom)
        if isinstance(self.neck, FPNNeck):
            # fusion layers take in_channels = out_channels*2 -> out_channels
            self.neck.fusion_p4 = IRDCB(neck_out_channels * 2, neck_out_channels, stride=1)
            self.neck.fusion_p3 = IRDCB(neck_out_channels * 2, neck_out_channels, stride=1)
        elif isinstance(self.neck, PANNeck):
            self.neck.fusion_p4_top = IRDCB(neck_out_channels * 2, neck_out_channels, stride=1)
            self.neck.fusion_p3_top = IRDCB(neck_out_channels * 2, neck_out_channels, stride=1)
            self.neck.fusion_p4_bottom = IRDCB(neck_out_channels * 2, neck_out_channels, stride=1)
            self.neck.fusion_p5_bottom = IRDCB(neck_out_channels * 2, neck_out_channels, stride=1)

        # Build head (remove num_classes from head_kwargs if present)
        head_kwargs = head_kwargs.copy()
        head_kwargs.pop('num_classes', None)
        self.head = build_head(
            in_channels=neck_out_channels,
            num_classes=num_classes,
            **head_kwargs,
        )

    def forward(self, x: torch.Tensor, targets=None, mode='train'):
        # Backbone -> get P3,P4,P5
        feats = self.backbone(x)
        p3, p4, p5 = feats['P3'], feats['P4'], feats['P5']

        # ASFB fusion: deep=P5, mid=P4, shallow=P3 -> fused at mid resolution (P4)
        fused_p4 = self.asfb(deep=p5, mid=p4, shallow=p3)

        # Prepare neck inputs: use fused_p4 as P4
        neck_inputs = {
            'P3': p3,
            'P4': fused_p4,
            'P5': p5,
        }

        neck_out = self.neck(neck_inputs)

        # Head predictions
        preds = self.head(neck_out)
        return preds

    def get_model_info(self) -> Dict:
        """Get model information (parameters, FLOPs, etc.)."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count parameters per component
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        asfb_params = sum(p.numel() for p in self.asfb.parameters())
        neck_params = sum(p.numel() for p in self.neck.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'backbone_params': backbone_params,
            'asfb_params': asfb_params,
            'neck_params': neck_params,
            'head_params': head_params,
        }


def build_l_refine_yolo(
    num_classes: int = 1,
    backbone_kwargs: Optional[Dict] = None,
    neck_type: str = 'fpn',
    neck_out_channels: int = 256,
    head_kwargs: Optional[Dict] = None,
):
    return L_Refine_YOLO(
        num_classes=num_classes,
        backbone_kwargs=backbone_kwargs,
        neck_type=neck_type,
        neck_out_channels=neck_out_channels,
        head_kwargs=head_kwargs,
    )


__all__ = ["L_Refine_YOLO", "build_l_refine_yolo"]
