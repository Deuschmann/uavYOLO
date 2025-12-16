"""
Evaluation script for UAV Object Detection.
(Fully Patched Version)
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np

from src.models import build_model
from src.data import UAVDetectionDataset, build_val_transform
from src.data.dataset_uav import collate_fn
from src.utils.config import load_config
from src.utils.ops import (
    box_cxcywh_to_xyxy,
    batched_nms,
)
from src.utils.metrics import (
    compute_map,
    compute_map_by_size,
    compute_map_by_condition,
)


def post_process_predictions(
    predictions: Dict[str, Dict[str, torch.Tensor]], 
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    max_detections: int = 300,
    img_size: int = 640,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Corrected post-processing function with decoder logic.
    """
    all_boxes = []
    all_scores = []
    all_labels = []
    
    scales_info = {'P3': 8, 'P4': 16, 'P5': 32}
    
    for scale_name, stride in scales_info.items():
        if scale_name not in predictions:
            continue
            
        pred_bbox_logits = predictions[scale_name]['bbox']
        pred_obj_logits = predictions[scale_name]['obj']
        pred_cls_logits = predictions[scale_name]['cls']
        
        B, _, H, W = pred_bbox_logits.shape
        device = pred_bbox_logits.device
        
        y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        
        obj_scores = torch.sigmoid(pred_obj_logits).squeeze(1)
        cls_probs = torch.softmax(pred_cls_logits, dim=1)
        final_scores = obj_scores.unsqueeze(1) * cls_probs
        
        # Decoder logic
        tx = torch.sigmoid(pred_bbox_logits[:, 0, ...])
        ty = torch.sigmoid(pred_bbox_logits[:, 1, ...])
        tw = torch.exp(pred_bbox_logits[:, 2, ...])
        th = torch.exp(pred_bbox_logits[:, 3, ...])
        
        cx = (x_coords.unsqueeze(0) + tx) * stride
        cy = (y_coords.unsqueeze(0) + ty) * stride
        w = tw * stride
        h = th * stride
        
        cx /= img_size
        cy /= img_size
        w /= img_size
        h /= img_size
        
        boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1).view(B, -1, 4)
        boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
        
        final_scores = final_scores.permute(0, 2, 3, 1).view(B, -1, final_scores.shape[1])
        
        for b in range(B):
            scores_b = final_scores[b]
            boxes_b = boxes_xyxy[b]
            
            max_scores, class_ids = scores_b.max(dim=1)
            
            mask = max_scores >= conf_threshold
            if mask.sum() == 0:
                continue
                
            all_boxes.append(boxes_b[mask])
            all_scores.append(max_scores[mask])
            all_labels.append(class_ids[mask])
    
    if len(all_boxes) == 0:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)
    
    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    keep = batched_nms(all_boxes, all_scores, all_labels, nms_threshold)
    all_boxes = all_boxes[keep]
    all_scores = all_scores[keep]
    all_labels = all_labels[keep]
    
    if len(all_boxes) > max_detections:
        top_indices = torch.argsort(all_scores, descending=True)[:max_detections]
        all_boxes = all_boxes[top_indices]
        all_scores = all_scores[top_indices]
        all_labels = all_labels[top_indices]
    
    return all_boxes, all_scores, all_labels


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    max_detections: int = 300,
    img_size: int = 640,
    num_classes: int = 1,
    conditions: List[str] = None,
) -> Dict:
    # This function should be OK, we don't need to change it
    model.eval()
    all_pred_boxes, all_pred_scores, all_pred_labels = [], [], []
    all_target_boxes, all_target_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            target_boxes_list = batch['boxes']
            target_labels_list = batch['labels']
            
            predictions = model(images, mode='val')
            
            for i in range(len(images)):
                pred_dict_i = {k: {sk: sv[i:i+1] for sk, sv in v.items()} for k, v in predictions.items()}
                
                pred_boxes, pred_scores, pred_labels = post_process_predictions(
                    pred_dict_i, conf_threshold, nms_threshold, max_detections, img_size
                )
                
                all_pred_boxes.append(pred_boxes.cpu())
                all_pred_scores.append(pred_scores.cpu())
                all_pred_labels.append(pred_labels.cpu())
                
                target_boxes_i = target_boxes_list[i]
                if len(target_boxes_i) > 0:
                    target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes_i)
                else:
                    target_boxes_xyxy = torch.empty((0, 4))
                
                all_target_boxes.append(target_boxes_xyxy.cpu())
                all_target_labels.append(target_labels_list[i].cpu())

    iou_threshold = 0.5
    
    map_result = compute_map(
        all_pred_boxes, all_pred_scores, all_pred_labels,
        all_target_boxes, all_target_labels,
        num_classes, iou_threshold
    )
    results = {'overall': map_result}
    
    size_map = compute_map_by_size(
        all_pred_boxes, all_pred_scores, all_pred_labels,
        all_target_boxes, all_target_labels,
        num_classes, img_size, iou_threshold
    )
    results['by_size'] = size_map
    
    if conditions is not None:
        condition_map = compute_map_by_condition(...)
        results['by_condition'] = condition_map
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate UAV Object Detection Model')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file for results')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    device_str = args.device or config['training']['device']
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Building model...")
    num_classes = config.get('classes', {}).get('num_classes')
    img_size = config.get('data', {}).get('img_size')
    if num_classes is None or img_size is None:
        raise ValueError("'num_classes' or 'img_size' not found in config.")
    
    model = build_model(config=config, num_classes=num_classes, img_size=img_size)
    model = model.to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint loaded from {args.checkpoint}")
    
    print("Building dataset...")
    val_transform = build_val_transform(img_size=img_size)
    
    # Use 'data/val' as the root for evaluation dataset
    eval_dataset = UAVDetectionDataset(
        root_dir='data/val', 
        images_dir=config['data']['images_dir'],
        labels_dir=config['data']['labels_dir'],
        transforms=val_transform,
    )
    print(f"Loaded {len(eval_dataset)} image-label pairs from data/val")
    
    # NOTE: The Subset code for debugging is removed. We run on the full dataset.
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,  # Using batch_size=1 to avoid OOM
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0),
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    print(f"Evaluation samples: {len(eval_dataset)}")
    print("\nEvaluating...")
    
    eval_cfg = config['evaluation']
    results = evaluate_model(
        model, eval_loader, device,
        conf_threshold=eval_cfg['conf_threshold'],
        nms_threshold=eval_cfg['nms_threshold'],
        max_detections=eval_cfg['max_detections'],
        img_size=img_size,
        num_classes=num_classes,
    )
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Overall mAP: {results['overall']['mAP']:.4f}")
    for key, value in results['overall'].items():
        if key != 'mAP':
            print(f"  {key}: {value:.4f}")
    
    if 'by_size' in results:
        print("\nBy Size:")
        for key, value in results['by_size'].items():
            print(f"  {key}: {value:.4f}")

    print("="*60)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {output_path}")

if __name__ == '__main__':
    main()