"""
Detection Loss for YOLO-style Object Detection (Fixed Version).
Includes:
1. Grid generation (Make Grid)
2. Prediction decoding (Sigmoid/Exp)
3. Scale assignment (Small objects -> P3, Large -> P5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import math

from ..utils.ops import (
    box_cxcywh_to_xyxy,
    box_iou,
    box_giou,
    box_ciou,
)

class DetectionLoss(nn.Module):
    def __init__(
        self,
        box_loss_type: str = "ciou",
        obj_loss_type: str = "bce",
        cls_loss_type: str = "bce",
        box_weight: float = 7.5,
        obj_weight: float = 1.0,
        cls_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        img_size: int = 640,
    ):
        super().__init__()
        self.box_loss_type = box_loss_type.lower()
        self.obj_loss_type = obj_loss_type.lower()
        self.cls_loss_type = cls_loss_type.lower()
        
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Loss functions
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
        # 定义每个 Scale 负责的目标尺寸范围 (像素)以及步长
        # P3(stride 8), P4(stride 16), P5(stride 32)
        self.scales_info = {
            'P3': {'stride': 8,  'range': [0, 64]},     # 小目标
            'P4': {'stride': 16, 'range': [64, 128]},   # 中目标
            'P5': {'stride': 32, 'range': [128, 10000]} # 大目标
        }

    def decode_boxes(self, pred_reg, stride, grid_x, grid_y, img_size):
        """
        将 Raw Logits 解码为归一化的 cx, cy, w, h
        pred_reg: (B, 4, H, W)
        """
        # 1. Sigmoid: 限制中心点偏移在 (0, 1) 之间，即当前网格内
        tx = torch.sigmoid(pred_reg[:, 0])
        ty = torch.sigmoid(pred_reg[:, 1])
        
        # 2. Exp: 确保宽高为正数
        tw = torch.exp(pred_reg[:, 2])
        th = torch.exp(pred_reg[:, 3])

        # 3. 还原回原图像素坐标
        # cx = (grid_x + tx) * stride
        cx_pix = (grid_x + tx) * stride
        cy_pix = (grid_y + ty) * stride
        w_pix = tw * stride
        h_pix = th * stride

        # 4. 归一化到 [0, 1] (因为 Loss 计算需要归一化坐标)
        return torch.stack([
            cx_pix / img_size,
            cy_pix / img_size,
            w_pix / img_size,
            h_pix / img_size
        ], dim=1)

    def compute_box_loss(self, pred_boxes, target_boxes):
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_xyxy = box_cxcywh_to_xyxy(target_boxes)
        
        if self.box_loss_type == "ciou":
            iou = box_ciou(pred_xyxy, target_xyxy).diag()
        elif self.box_loss_type == "giou":
            iou = box_giou(pred_xyxy, target_xyxy).diag()
        else:
            iou = box_iou(pred_xyxy, target_xyxy).diag()
        # 加上 clamp 防止 NaN
        return 1.0 - iou.clamp(min=-1.0, max=1.0)
    
    def forward(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]], 
        targets: Dict[str, List[torch.Tensor]], 
        img_size: int = 640
    ) -> Dict[str, torch.Tensor]:
        
        # 确保 img_size 是浮点数，用于除法
        current_img_size = float(img_size)
        
        # 获取基本信息
        batch_size = predictions['P3']['bbox'].shape[0]
        # 注意：这里假设 cls 通道数正确
        num_classes = predictions['P3']['cls'].shape[1]
        device = predictions['P3']['bbox'].device
        
        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        
        total_num_positives = 0
        
        for scale_name in ['P3', 'P4', 'P5']:
            # pred_reg 是原始卷积输出，没有经过 sigmoid
            pred_reg = predictions[scale_name]['bbox'] # (B, 4, H, W)
            pred_obj = predictions[scale_name]['obj']   # (B, 1, H, W)
            pred_cls = predictions[scale_name]['cls']   # (B, num_classes, H, W)
            
            stride = self.scales_info[scale_name]['stride']
            min_sz, max_sz = self.scales_info[scale_name]['range']
            
            B, _, H, W = pred_reg.shape
            
            # --- 1. 生成网格 (Make Grid) ---
            y_coords, x_coords = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            grid_y = y_coords.expand(B, H, W)
            grid_x = x_coords.expand(B, H, W)
            
            # --- 2. 解码预测框 (Decoding) ---
            # 这一步至关重要：把卷积输出变成真实的归一化坐标
            decoded_pred_boxes = self.decode_boxes(pred_reg, stride, grid_x, grid_y, current_img_size)
            # decoded_pred_boxes: (B, 4, H, W)
            
            # 遍历 Batch
            for b in range(batch_size):
                t_boxes = targets['boxes'][b]  # (N, 4) cxcywh normalized
                t_labels = targets['labels'][b] 
                
                # --- 空数据处理 ---
                if len(t_boxes) == 0:
                    obj_target = torch.zeros_like(pred_obj[b])
                    # 只有负样本，降低权重
                    total_obj_loss += self.bce(pred_obj[b], obj_target).mean() * 0.1
                    continue

                # --- 尺度筛选 (Scale Match) ---
                # 还原回像素尺寸进行筛选
                t_pixel_w = t_boxes[:, 2] * current_img_size
                t_pixel_h = t_boxes[:, 3] * current_img_size
                t_pixel_max = torch.max(t_pixel_w, t_pixel_h)
                
                # 核心逻辑：只让当前层负责大小合适的目标
                mask_in_scale = (t_pixel_max > min_sz) & (t_pixel_max <= max_sz)
                
                if mask_in_scale.sum() == 0:
                    obj_target = torch.zeros_like(pred_obj[b])
                    total_obj_loss += self.bce(pred_obj[b], obj_target).mean() * 0.1
                    continue
                
                valid_boxes = t_boxes[mask_in_scale]
                valid_labels = t_labels[mask_in_scale]
                
                # --- 3. 正样本匹配 (Grid Matching) ---
                # 找到目标中心点所在的网格坐标
                gx = (valid_boxes[:, 0] * W).long().clamp(0, W - 1)
                gy = (valid_boxes[:, 1] * H).long().clamp(0, H - 1)
                
                # --- 计算 Loss ---
                
                # [Box Loss]
                # 拿出对应网格“解码后”的预测框
                matched_pred_box = decoded_pred_boxes[b, :, gy, gx].T # (Num_Pos, 4)
                box_loss_b = self.compute_box_loss(matched_pred_box, valid_boxes)
                total_box_loss += box_loss_b.sum()
                
                # [Class Loss]
                matched_pred_cls = pred_cls[b, :, gy, gx].T
                cls_target = F.one_hot(valid_labels, num_classes).float()
                total_cls_loss += self.bce(matched_pred_cls, cls_target).sum()
                
                # [Objectness Loss - Positive]
                matched_pred_obj = pred_obj[b, 0, gy, gx]
                obj_target_pos = torch.ones_like(matched_pred_obj)
                total_obj_loss += self.bce(matched_pred_obj, obj_target_pos).sum()
                
                # [Objectness Loss - Negative (Full Map)]
                # 简单方案：计算全图 Loss，正样本位置的 Loss 会重复计算（一次正一次负），
                # 但正样本数量远小于负样本，影响可控。
                # 为了准确，可以给全图 Loss 一个较小的权重
                full_obj_target = torch.zeros((H, W), device=device)
                # full_obj_target[gy, gx] = 1.0 # 严谨做法是不在全图里算正样本，或者Mask掉
                
                full_obj_pred = pred_obj[b, 0]
                loss_map = self.bce(full_obj_pred, full_obj_target)
                
                # Mask掉正样本位置，避免冲突
                mask_neg = torch.ones((H, W), device=device)
                mask_neg[gy, gx] = 0
                
                total_obj_loss += (loss_map * mask_neg).sum() * 0.05 # 负样本权重系数
                
                total_num_positives += len(valid_boxes)

        # --- 4. 归一化 ---
        norm_factor = max(total_num_positives, 1)
        
        box_loss = total_box_loss / norm_factor
        cls_loss = total_cls_loss / norm_factor
        # Obj loss 通常数值较大，多除一点
        obj_loss = total_obj_loss / (batch_size * 3) 

        return {
            'loss': self.box_weight * box_loss + self.obj_weight * obj_loss + self.cls_weight * cls_loss,
            'box_loss': box_loss,
            'obj_loss': obj_loss,
            'cls_loss': cls_loss
        }


def build_loss(config: Dict, img_size: int = 640) -> DetectionLoss:
    loss_cfg = config.get('training', {}).get('loss', {})
    return DetectionLoss(
        box_loss_type=loss_cfg.get('box_loss_type', 'ciou'),
        obj_loss_type=loss_cfg.get('obj_loss_type', 'bce'),
        cls_loss_type=loss_cfg.get('cls_loss_type', 'bce'),
        box_weight=loss_cfg.get('box_weight', 7.5),
        obj_weight=loss_cfg.get('obj_weight', 1.0),
        cls_weight=loss_cfg.get('cls_weight', 0.5),
        focal_alpha=loss_cfg.get('focal_alpha', 0.25),
        focal_gamma=loss_cfg.get('focal_gamma', 2.0),
        img_size=img_size,
    )
