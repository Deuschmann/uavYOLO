"""
Training script for UAV Object Detection.

Supports:
    - Configurable training from YAML config
    - Optimizer and learning rate scheduling
    - Loss computation and logging
    - Periodic evaluation and checkpointing
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Handle both 'python src/train.py' and 'python -m src.train' execution
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.models import build_model
from src.data import UAVDetectionDataset, build_train_transform, build_val_transform
from src.data.dataset_uav import collate_fn
from src.loss import build_loss
from src.utils.config import load_config, merge_configs


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    opt_cfg = config['training']['optimizer']
    opt_type = opt_cfg['type'].lower()
    
    if opt_type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg.get('weight_decay', 0.0),
        )
    elif opt_type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg.get('weight_decay', 0.0005),
        )
    elif opt_type == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=opt_cfg['lr'],
            momentum=opt_cfg.get('momentum', 0.937),
            weight_decay=opt_cfg.get('weight_decay', 0.0005),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict, num_iterations_per_epoch: int):
    """Build learning rate scheduler from config."""
    sched_cfg = config['training']['scheduler']
    sched_type = sched_cfg['type'].lower()
    
    if sched_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=sched_cfg.get('min_lr', 0.0),
        )
    elif sched_type == 'step':
        from torch.optim.lr_scheduler import StepLR
        return StepLR(
            optimizer,
            step_size=sched_cfg.get('step_size', 30),
            gamma=sched_cfg.get('gamma', 0.1),
        )
    elif sched_type == 'warmup_cosine':
        # Custom warmup + cosine annealing
        from torch.optim.lr_scheduler import LambdaLR
        warmup_epochs = sched_cfg.get('warmup_epochs', 3)
        min_lr = sched_cfg.get('min_lr', 0.0)
        warmup_lr = sched_cfg.get('warmup_lr', 0.0001)
        max_lr = config['training']['optimizer']['lr']
        num_epochs = config['training']['num_epochs']
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Warmup: linear increase from warmup_lr to max_lr
                return warmup_lr + (max_lr - warmup_lr) * (epoch / warmup_epochs)
            else:
                # Cosine annealing from max_lr to min_lr
                progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
                cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                return min_lr + (max_lr - min_lr) * cosine_factor
        
        return LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    else:
        # No scheduler
        return None


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    loss: float,
    save_path: Path,
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {epoch}, loss: {loss:.4f}")
    
    return epoch, loss


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_freq: int = 10,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_box_loss = 0.0
    total_obj_loss = 0.0
    total_cls_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['images'].to(device)
        targets = {
            'boxes': [boxes.to(device) for boxes in batch['boxes']],
            'labels': [labels.to(device) for labels in batch['labels']],
        }
        
        # Forward pass
        predictions = model(images, targets=targets, mode='train')
        
        # Compute loss
        loss_dict = loss_fn(predictions, targets)
        loss = loss_dict['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_box_loss += loss_dict['box_loss'].item()
        total_obj_loss += loss_dict['obj_loss'].item()
        total_cls_loss += loss_dict['cls_loss'].item()
        num_batches += 1
        
        # Logging
        if (batch_idx + 1) % log_freq == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch}, Batch {batch_idx+1}/{len(dataloader)}, "
                f"Loss: {loss.item():.4f} "
                f"(Box: {loss_dict['box_loss'].item():.4f}, "
                f"Obj: {loss_dict['obj_loss'].item():.4f}, "
                f"Cls: {loss_dict['cls_loss'].item():.4f}), "
                f"LR: {current_lr:.6f}"
            )
    
    avg_loss = total_loss / num_batches
    avg_box_loss = total_box_loss / num_batches
    avg_obj_loss = total_obj_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    
    return {
        'loss': avg_loss,
        'box_loss': avg_box_loss,
        'obj_loss': avg_obj_loss,
        'cls_loss': avg_cls_loss,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_box_loss = 0.0
    total_obj_loss = 0.0
    total_cls_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images'].to(device)
            targets = {
                'boxes': [boxes.to(device) for boxes in batch['boxes']],
                'labels': [labels.to(device) for labels in batch['labels']],
            }
            
            predictions = model(images, targets=targets, mode='val')
            loss_dict = loss_fn(predictions, targets)
            
            total_loss += loss_dict['loss'].item()
            total_box_loss += loss_dict['box_loss'].item()
            total_obj_loss += loss_dict['obj_loss'].item()
            total_cls_loss += loss_dict['cls_loss'].item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_box_loss = total_box_loss / num_batches
    avg_obj_loss = total_obj_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    
    return {
        'loss': avg_loss,
        'box_loss': avg_box_loss,
        'obj_loss': avg_obj_loss,
        'cls_loss': avg_cls_loss,
    }


def main():
    parser = argparse.ArgumentParser(description='Train UAV Object Detection Model')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu, overrides config)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--experiment', type=str, default=None, help='Experiment name')
    args = parser.parse_args()
    
    # Load config with recursive inheritance
    def load_config_recursive(config_path):
        config = load_config(config_path)
        if '_extends' in config:
            base_path = os.path.join(os.path.dirname(config_path), config['_extends'])
            base_config = load_config_recursive(base_path)
            config = merge_configs(base_config, config)
        return config
    
    config = load_config_recursive(args.config)
    # Debug: print merged config keys to help diagnose missing fields
    print("Loaded config keys:", list(config.keys()))
    if 'training' not in config:
        raise KeyError(
            f"'training' not found in merged config keys: {list(config.keys())}. "
            f"Check your config inheritance chain for {args.config} and its _extends files."
        )
    
    # Set device
    device_str = args.device or config['training']['device']
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(config['training']['seed'])
    
    # Create experiment directory
    experiment_name = args.experiment or f"experiment_{int(time.time())}"
    exp_dir = Path(config['paths']['checkpoint_dir']) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")
    
    # Save config
    import yaml
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Build datasets
    print("Building datasets...")
    train_transform = build_train_transform(
        img_size=config['data']['img_size'],
        use_geometric=config['augmentation']['use_geometric'],
        use_fog=config['augmentation']['use_fog'],
        use_rain=config['augmentation']['use_rain'],
        use_blur=config['augmentation']['use_blur'],
        use_noise=config['augmentation']['use_noise'],
        use_brightness=config['augmentation']['use_brightness'],
        use_shadow=config['augmentation']['use_shadow'],
        config=config['augmentation'],
    )
    
    val_transform = build_val_transform(img_size=config['data']['img_size'])
    
    train_dataset = UAVDetectionDataset(
        root_dir=config['data']['root_dir'],
        images_dir=config['data']['images_dir'],
        labels_dir=config['data']['labels_dir'],
        transforms=train_transform,
        class_names=config.get('classes', {}).get('names', []),
    )
    
    val_dataset = UAVDetectionDataset(
        root_dir=config['data']['root_dir'],
        images_dir=config['data']['val_images_dir'],
        labels_dir=config['data']['val_labels_dir'],
        transforms=val_transform,
        class_names=config.get('classes', {}).get('names', []),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Build model
    print("Building model...")
    num_classes = config['classes']['num_classes']
    model = build_model(
        config=config,
        num_classes=num_classes,
        img_size=config['data']['img_size'],
    )
    model = model.to(device)
    
    # Print model info
    info = model.get_model_info()
    print(f"Model parameters: {info['total_params']:,}")
    
    # Build loss
    loss_fn = build_loss(config, img_size=config['data']['img_size'])
    loss_fn = loss_fn.to(device)
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume or config['training'].get('resume'):
        checkpoint_path = Path(args.resume or config['training']['resume'])
        start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        start_epoch += 1
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    save_freq = config['training']['save_freq']
    eval_freq = config['training']['eval_freq']
    log_freq = config['training']['log_freq']
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("="*60)
    
    best_val_loss = float('inf')

    # --- Early Stopping Initialization ---
    early_stopping_cfg = config['training'].get('early_stopping', {'enabled': False})
    early_stopping_enabled = early_stopping_cfg.get('enabled', False)
    if early_stopping_enabled:
        patience = early_stopping_cfg.get('patience', 10)
        min_delta = early_stopping_cfg.get('min_delta', 0.0001)
        epochs_no_improve = 0
        print(f"Early stopping enabled with patience={patience} and min_delta={min_delta}")
    # ------------------------------------

    # --- BN Freeze Definition ---
    freeze_bn_epoch = int(num_epochs * 0.8) 
    print(f"BatchNorm layers will be frozen starting from epoch {freeze_bn_epoch}")
    # -----------------------------------
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-"*60)

        # Freeze BatchNorm layers if applicable
        if epoch >= freeze_bn_epoch:
            print("Freezing BatchNorm layers...")
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval() 
        
        # Set model to train mode (this will be overridden for BN layers if frozen)
        model.train() 
        
        # Re-apply BN freeze if needed after model.train()
        if epoch >= freeze_bn_epoch:
             for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device,
            epoch+1, log_freq
        )
        print(f"Train - Loss: {train_metrics['loss']:.4f} "
              f"(Box: {train_metrics['box_loss']:.4f}, "
              f"Obj: {train_metrics['obj_loss']:.4f}, "
              f"Cls: {train_metrics['cls_loss']:.4f})")
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
        
        # Validate
        if (epoch + 1) % eval_freq == 0:
            val_metrics = validate(model, val_loader, loss_fn, device)
            print(f"Val   - Loss: {val_metrics['loss']:.4f} "
                  f"(Box: {val_metrics['box_loss']:.4f}, "
                  f"Obj: {val_metrics['obj_loss']:.4f}, "
                  f"Cls: {val_metrics['cls_loss']:.4f})")
            
            # Check for improvement and save best model
            improved = (val_metrics['loss'] < best_val_loss - min_delta) if early_stopping_enabled else (val_metrics['loss'] < best_val_loss)
            if improved:
                best_val_loss = val_metrics['loss']
                best_path = exp_dir / 'best_model.pth'
                save_checkpoint(model, optimizer, scheduler, epoch+1, val_metrics['loss'], best_path)
                if early_stopping_enabled:
                    epochs_no_improve = 0
            elif early_stopping_enabled:
                epochs_no_improve += 1
                print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

            # Early Stopping Check
            if early_stopping_enabled and epochs_no_improve >= patience:
                print(f"\nValidation loss has not improved for {patience} epochs. Stopping training early.")
                break
        
        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            checkpoint_path = exp_dir / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(model, optimizer, scheduler, epoch+1, train_metrics['loss'], checkpoint_path)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {exp_dir}")


if __name__ == '__main__':
    main()

