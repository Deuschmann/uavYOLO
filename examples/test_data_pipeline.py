"""
Example script to test the data pipeline.

This script demonstrates how to:
1. Load configuration
2. Create dataset with augmentations
3. Visualize augmented images
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import UAVDetectionDataset, build_train_transform, build_val_transform
from src.utils.config import load_config
from torch.utils.data import DataLoader
from src.data.dataset_uav import collate_fn
import matplotlib.pyplot as plt
import numpy as np


def visualize_batch(dataloader, num_samples=4, save_path="test_augmentations.png"):
    """Visualize a batch of augmented images with bounding boxes."""
    batch = next(iter(dataloader))
    
    images = batch['images']
    boxes_list = batch['boxes']
    labels_list = batch['labels']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx in range(min(num_samples, len(images))):
        img = images[idx].permute(1, 2, 0).numpy()
        img = (img * 255).astype(np.uint8)
        boxes = boxes_list[idx].numpy()
        labels = labels_list[idx].numpy()
        
        # Draw image
        axes[idx].imshow(img)
        axes[idx].set_title(f"Image {idx+1} - {len(boxes)} boxes")
        axes[idx].axis('off')
        
        # Draw bounding boxes
        h, w = img.shape[:2]
        for box, label in zip(boxes, labels):
            x_center, y_center, width, height = box
            x1 = (x_center - width / 2) * w
            y1 = (y_center - height / 2) * h
            x2 = (x_center + width / 2) * w
            y2 = (y_center + height / 2) * h
            
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, color='red', linewidth=2
            )
            axes[idx].add_patch(rect)
            axes[idx].text(x1, y1, f"cls:{label}", color='red', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "base.yaml"
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Please create a config file first.")
        return
    
    config = load_config(config_path)
    
    # Check if data directory exists
    data_root = Path(config['data']['root_dir'])
    if not data_root.exists():
        print(f"Data directory not found: {data_root}")
        print("Please create the data directory structure:")
        print(f"  {data_root}/images/  - containing your images")
        print(f"  {data_root}/labels/  - containing YOLO format .txt files")
        return
    
    # Build transforms
    print("Building augmentation transforms...")
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
    
    # Create dataset
    print("Creating dataset...")
    try:
        train_dataset = UAVDetectionDataset(
            root_dir=data_root,
            images_dir=config['data']['images_dir'],
            labels_dir=config['data']['labels_dir'],
            transforms=train_transform,
        )
        
        print(f"Dataset created with {len(train_dataset)} samples")
        
        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(config['data']['batch_size'], len(train_dataset)),
            shuffle=True,
            num_workers=0,  # Set to 0 for debugging, use config value for training
            collate_fn=collate_fn,
        )
        
        # Test loading
        print("Testing data loading...")
        batch = next(iter(train_loader))
        print(f"Batch shape: {batch['images'].shape}")
        print(f"Number of boxes in first image: {len(batch['boxes'][0])}")
        print(f"Number of labels in first image: {len(batch['labels'][0])}")
        
        # Visualize
        print("Creating visualization...")
        visualize_batch(train_loader, save_path="test_augmentations.png")
        print("✓ Data pipeline test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

