"""
Evaluation metrics for object detection.

Implements AP (Average Precision) and mAP computation with breakdowns
by object size and condition.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .ops import box_iou, box_cxcywh_to_xyxy


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute Average Precision (AP) from recall-precision curve.
    
    Args:
        recall: Recall values
        precision: Precision values
    
    Returns:
        AP value (0-1)
    """
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Find points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


# 在 src/utils/metrics.py 文件里

# 这个函数 box_iou 应该已经存在，确保它在上面
# def box_iou(...): ...

def compute_class_ap(
    pred_boxes,
    pred_scores,
    pred_labels,
    target_boxes,
    target_labels,
    target_difficulties, # 这个参数可能没用到，但保持签名一致
    iou_threshold=0.5,
):
    """
    Compute Average Precision (AP) for a single class.
    (Final Fixed Version)
    """
    # ----------------------------------------------------
    # 关键修复：在这里增加保护措施
    # ----------------------------------------------------
    # Case 1: 没有预测框，AP 直接为 0
    if pred_boxes.numel() == 0:
        return 0.0, {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(target_boxes)}

    # Case 2: 有预测框，但没有真值框。所有预测都是误报 (False Positive)
    if target_boxes.numel() == 0:
        return 0.0, {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(pred_boxes), 'fn': 0}
        
    # Sort predictions by score (descending)
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]

    # Compute IoU matrix
    ious = box_iou(pred_boxes, target_boxes)

    # For each prediction, find the target with the highest IoU
    max_iou, best_target_idx = ious.max(dim=1) # This line is now safe

    # Determine True Positives (TP) and False Positives (FP)
    tp = torch.zeros(len(pred_boxes))
    fp = torch.zeros(len(pred_boxes))
    
    # Keep track of which targets have been matched
    matched_targets = torch.zeros(len(target_boxes))
    
    for i in range(len(pred_boxes)):
        if max_iou[i] >= iou_threshold:
            target_idx = best_target_idx[i]
            if matched_targets[target_idx] == 0:
                tp[i] = 1
                matched_targets[target_idx] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1
            
    # Compute precision and recall
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    
    num_targets = len(target_boxes)
    recalls = tp_cumsum / (num_targets + 1e-6)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # VOC 11-point interpolated AP
    ap = 0.0
    for t in torch.arange(0.0, 1.1, 0.1):
        if torch.sum(recalls >= t) == 0:
            p_max = 0.0
        else:
            p_max = torch.max(precisions[recalls >= t])
        ap += p_max / 11.0
        
    final_tp = int(tp.sum())
    final_fp = int(fp.sum())
    final_fn = num_targets - final_tp
    final_precision = final_tp / (final_tp + final_fp + 1e-6)
    final_recall = final_tp / (final_tp + final_fn + 1e-6)
    final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall + 1e-6)

    metrics = {
        'precision': final_precision,
        'recall': final_recall,
        'f1': final_f1,
        'tp': final_tp,
        'fp': final_fp,
        'fn': final_fn,
    }
    
    return ap, metrics

# 在 src/utils/metrics.py 文件里

def compute_map(
    all_pred_boxes: List[torch.Tensor],
    all_pred_scores: List[torch.Tensor],
    all_pred_labels: List[torch.Tensor],
    all_target_boxes: List[torch.Tensor],
    all_target_labels: List[torch.Tensor],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute mAP (mean Average Precision) across all classes.
    (Fixed Version with per-class filtering)
    """
    # 将所有图片的结果拼接成一个大 Tensor
    # 这一步是为了方便按类别进行全局筛选
    pred_boxes = torch.cat(all_pred_boxes, dim=0) if any(t.numel() > 0 for t in all_pred_boxes) else torch.empty((0, 4))
    pred_scores = torch.cat(all_pred_scores, dim=0) if any(t.numel() > 0 for t in all_pred_scores) else torch.empty((0,))
    pred_labels = torch.cat(all_pred_labels, dim=0) if any(t.numel() > 0 for t in all_pred_labels) else torch.empty((0,), dtype=torch.long)
    
    target_boxes = torch.cat(all_target_boxes, dim=0) if any(t.numel() > 0 for t in all_target_boxes) else torch.empty((0, 4))
    target_labels = torch.cat(all_target_labels, dim=0) if any(t.numel() > 0 for t in all_target_labels) else torch.empty((0,), dtype=torch.long)
    
    # target_difficulties 可能是未来扩展用的，暂时创建一个空的
    target_difficulties = torch.zeros_like(target_labels, dtype=torch.bool)

    aps = []
    class_aps = {}
    
    for class_id in range(num_classes):
        # --- 关键修复：按类别筛选数据 ---
        
        # 1. 筛选属于当前 class_id 的预测
        pred_mask = (pred_labels == class_id)
        pred_boxes_c = pred_boxes[pred_mask]
        pred_scores_c = pred_scores[pred_mask]
        pred_labels_c = pred_labels[pred_mask] # 虽然都是同一个id，但保持参数一致性
        
        # 2. 筛选属于当前 class_id 的真值
        target_mask = (target_labels == class_id)
        target_boxes_c = target_boxes[target_mask]
        target_labels_c = target_labels[target_mask]
        target_difficulties_c = target_difficulties[target_mask]

        # ------------------------------------

        # 调用 compute_class_ap，现在传入的是筛选后的数据
        ap, metrics = compute_class_ap(
            pred_boxes_c,
            pred_scores_c,
            pred_labels_c,
            target_boxes_c,
            target_labels_c,
            target_difficulties_c,
            iou_threshold
        )
        
        # 只有当这个类别存在真值时，才计入mAP计算
        # 这是 COCO 的标准做法，避免没有物体的类别拉低mAP
        if target_boxes_c.numel() > 0:
            aps.append(ap)
            
        class_aps[f'AP_class_{class_id}'] = ap
    
    # Compute mAP (mean of all classes that have ground truth)
    map_value = np.mean([ap for ap in aps if not np.isnan(ap)]) if len(aps) > 0 else 0.0
    
    result = {
        'mAP': map_value,
        **class_aps,
    }
    
    return result

def categorize_by_size(
    boxes: torch.Tensor,  # (N, 4) in xyxy format
    img_size: int = 640,
) -> Dict[str, torch.Tensor]:
    """
    Categorize boxes by size (small, medium, large).
    
    Uses COCO-style size definitions:
        - Small: area < 32^2
        - Medium: 32^2 <= area < 96^2
        - Large: area >= 96^2
    
    Args:
        boxes: Boxes in xyxy format (normalized 0-1)
        img_size: Image size (for area calculation)
    
    Returns:
        Dictionary with masks for each size category
    """
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (img_size ** 2)
    
    small_mask = areas < (32 ** 2)
    medium_mask = (areas >= (32 ** 2)) & (areas < (96 ** 2))
    large_mask = areas >= (96 ** 2)
    
    return {
        'small': small_mask,
        'medium': medium_mask,
        'large': large_mask,
    }


def compute_map_by_size(
    all_pred_boxes: List[torch.Tensor],
    all_pred_scores: List[torch.Tensor],
    all_pred_labels: List[torch.Tensor],
    all_target_boxes: List[torch.Tensor],
    all_target_labels: List[torch.Tensor],
    num_classes: int,
    img_size: int = 640,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute mAP broken down by object size.
    
    Args:
        all_pred_boxes: List of predicted boxes for each image
        all_pred_scores: List of predicted scores
        all_pred_labels: List of predicted labels
        all_target_boxes: List of target boxes
        all_target_labels: List of target labels
        num_classes: Number of classes
        img_size: Image size
        iou_threshold: IoU threshold
    
    Returns:
        Dictionary with mAP for each size category
    """
    # Categorize predictions and targets by size
    results = {}
    
    for size_name in ['small', 'medium', 'large']:
        pred_boxes_size = []
        pred_scores_size = []
        pred_labels_size = []
        target_boxes_size = []
        target_labels_size = []
        
        for i in range(len(all_pred_boxes)):
            # Categorize predictions
            if len(all_pred_boxes[i]) > 0:
                size_mask_pred = categorize_by_size(all_pred_boxes[i], img_size)[size_name]
                pred_boxes_size.append(all_pred_boxes[i][size_mask_pred])
                pred_scores_size.append(all_pred_scores[i][size_mask_pred])
                pred_labels_size.append(all_pred_labels[i][size_mask_pred])
            else:
                pred_boxes_size.append(torch.empty((0, 4)))
                pred_scores_size.append(torch.empty((0,)))
                pred_labels_size.append(torch.empty((0,), dtype=torch.long))
            
            # Categorize targets
            if len(all_target_boxes[i]) > 0:
                size_mask_target = categorize_by_size(all_target_boxes[i], img_size)[size_name]
                target_boxes_size.append(all_target_boxes[i][size_mask_target])
                target_labels_size.append(all_target_labels[i][size_mask_target])
            else:
                target_boxes_size.append(torch.empty((0, 4)))
                target_labels_size.append(torch.empty((0,), dtype=torch.long))
        
        # Compute mAP for this size category
        map_result = compute_map(
            pred_boxes_size, pred_scores_size, pred_labels_size,
            target_boxes_size, target_labels_size,
            num_classes, iou_threshold
        )
        results[f'mAP_{size_name}'] = map_result['mAP']
    
    return results


def compute_map_by_condition(
    all_pred_boxes: List[torch.Tensor],
    all_pred_scores: List[torch.Tensor],
    all_pred_labels: List[torch.Tensor],
    all_target_boxes: List[torch.Tensor],
    all_target_labels: List[torch.Tensor],
    conditions: List[str],  # List of condition labels for each image (e.g., "clean", "fog", "rain")
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute mAP broken down by condition (weather/lighting).
    
    Args:
        all_pred_boxes: List of predicted boxes
        all_pred_scores: List of predicted scores
        all_pred_labels: List of predicted labels
        all_target_boxes: List of target boxes
        all_target_labels: List of target labels
        conditions: List of condition labels for each image
        num_classes: Number of classes
        iou_threshold: IoU threshold
    
    Returns:
        Dictionary with mAP for each condition
    """
    # Group by condition
    condition_groups = defaultdict(lambda: {
        'pred_boxes': [],
        'pred_scores': [],
        'pred_labels': [],
        'target_boxes': [],
        'target_labels': [],
    })
    
    for i, condition in enumerate(conditions):
        condition_groups[condition]['pred_boxes'].append(all_pred_boxes[i])
        condition_groups[condition]['pred_scores'].append(all_pred_scores[i])
        condition_groups[condition]['pred_labels'].append(all_pred_labels[i])
        condition_groups[condition]['target_boxes'].append(all_target_boxes[i])
        condition_groups[condition]['target_labels'].append(all_target_labels[i])
    
    # Compute mAP for each condition
    results = {}
    for condition, data in condition_groups.items():
        map_result = compute_map(
            data['pred_boxes'], data['pred_scores'], data['pred_labels'],
            data['target_boxes'], data['target_labels'],
            num_classes, iou_threshold
        )
        results[f'mAP_{condition}'] = map_result['mAP']
    
    return results

