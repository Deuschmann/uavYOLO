"""
Common operations for object detection.

Includes IoU calculations, box format conversions, NMS, etc.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2).
    
    Args:
        boxes: (..., 4) tensor in format (cx, cy, w, h)
    
    Returns:
        (..., 4) tensor in format (x1, y1, x2, y2)
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (x1, y1, x2, y2) to (center_x, center_y, width, height).
    
    Args:
        boxes: (..., 4) tensor in format (x1, y1, x2, y2)
    
    Returns:
        (..., 4) tensor in format (cx, cy, w, h)
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute area of boxes.
    
    Args:
        boxes: (..., 4) tensor in format (x1, y1, x2, y2)
    
    Returns:
        (...,) tensor of areas
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    return (x2 - x1) * (y2 - y1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU (Intersection over Union) between two sets of boxes.
    
    Args:
        boxes1: (N, 4) tensor in format (x1, y1, x2, y2)
        boxes2: (M, 4) tensor in format (x1, y1, x2, y2)
    
    Returns:
        (N, M) tensor of IoU values
    """
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)
    
    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    # Compute union
    union = area1[:, None] + area2 - inter  # (N, M)
    
    iou = inter / (union + 1e-6)
    return iou


def box_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute GIoU (Generalized Intersection over Union).
    
    Args:
        boxes1: (N, 4) tensor in format (x1, y1, x2, y2)
        boxes2: (M, 4) tensor in format (x1, y1, x2, y2)
    
    Returns:
        (N, M) tensor of GIoU values
    """
    iou = box_iou(boxes1, boxes2)  # (N, M)
    
    # Compute enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    area_c = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    # Compute GIoU
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)
    union = area1[:, None] + area2 - (iou * (area1[:, None] + area2) - iou * torch.minimum(area1[:, None], area2))
    
    giou = iou - (area_c - union) / (area_c + 1e-6)
    return giou


def box_ciou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute CIoU (Complete Intersection over Union).
    
    Args:
        boxes1: (N, 4) tensor in format (x1, y1, x2, y2) - predictions
        boxes2: (M, 4) tensor in format (x1, y1, x2, y2) - targets
    
    Returns:
        (N, M) tensor of CIoU values
    """
    # Convert to (cx, cy, w, h) for center distance calculation
    boxes1_cxcywh = box_xyxy_to_cxcywh(boxes1)
    boxes2_cxcywh = box_xyxy_to_cxcywh(boxes2)
    
    # IoU
    iou = box_iou(boxes1, boxes2)  # (N, M)
    
    # Center distance
    center1 = boxes1_cxcywh[:, None, :2]  # (N, 1, 2)
    center2 = boxes2_cxcywh[:, :2]  # (M, 2)
    center_distance_sq = ((center1 - center2) ** 2).sum(dim=-1)  # (N, M)
    
    # Enclosing box diagonal
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    diagonal_sq = ((rb - lt) ** 2).sum(dim=-1)  # (N, M)
    
    # Aspect ratio consistency
    w1, h1 = boxes1_cxcywh[:, None, 2], boxes1_cxcywh[:, None, 3]  # (N, 1)
    w2, h2 = boxes2_cxcywh[:, 2], boxes2_cxcywh[:, 3]  # (M,)
    
    v = (4 / (torch.pi ** 2)) * torch.pow(
        torch.atan(w2 / (h2 + 1e-6)) - torch.atan(w1 / (h1 + 1e-6)),
        2
    )  # (N, M)
    
    alpha = v / (1 - iou + v + 1e-6)  # (N, M)
    
    # CIoU
    ciou = iou - (center_distance_sq / (diagonal_sq + 1e-6)) - alpha * v
    return ciou


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Non-Maximum Suppression (NMS).
    
    Args:
        boxes: (N, 4) tensor in format (x1, y1, x2, y2)
        scores: (N,) tensor of scores
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    
    # Sort by scores (descending)
    _, indices = scores.sort(descending=True)
    keep = []
    
    while len(indices) > 0:
        # Keep the box with highest score
        current = indices[0]
        keep.append(current.item())
        
        if len(indices) == 1:
            break
        
        # Remove current box
        indices = indices[1:]
        
        # Compute IoU with remaining boxes
        current_box = boxes[current:current+1]
        remaining_boxes = boxes[indices]
        ious = box_iou(current_box, remaining_boxes).squeeze(0)
        
        # Remove boxes with IoU > threshold
        mask = ious <= iou_threshold
        indices = indices[mask]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Batched NMS (per-class NMS).
    
    Args:
        boxes: (N, 4) tensor in format (x1, y1, x2, y2)
        scores: (N,) tensor of scores
        labels: (N,) tensor of class labels
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Indices of boxes to keep
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    
    # Get unique labels
    unique_labels = labels.unique()
    keep = []
    
    for label in unique_labels:
        # Get boxes for this class
        mask = labels == label
        class_boxes = boxes[mask]
        class_scores = scores[mask]
        class_indices = torch.where(mask)[0]
        
        # Apply NMS
        nms_indices = nms(class_boxes, class_scores, iou_threshold)
        keep.extend(class_indices[nms_indices].tolist())
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

