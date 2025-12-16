# UAV Object Detection Framework

A modular PyTorch framework for YOLO-style object detection in UAV/aerial imagery, optimized for small objects and robustness to adverse weather conditions.

## Project Structure

```
YOLO/
├── configs/              # Configuration files (YAML)
├── src/
│   ├── data/            # Dataset and augmentation modules
│   ├── models/          # Model components (to be implemented)
│   ├── loss/            # Loss functions (to be implemented)
│   ├── utils/           # Utilities (config, metrics, etc.)
│   ├── train.py         # Training script (to be implemented)
│   └── eval.py          # Evaluation script (to be implemented)
├── scripts/             # Shell scripts for training/eval
└── requirements.txt     # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Dataset Format

The framework expects YOLO format annotations:
- Images in `data/images/` (or custom path)
- Labels in `data/labels/` (one `.txt` file per image)
- Label format: `class_id x_center y_center width height` (all normalized 0-1)

Example label file (`labels/img001.txt`):
```
0 0.5 0.3 0.2 0.15
1 0.7 0.6 0.1 0.08
```

### Using the Dataset

```python
from src.data import UAVDetectionDataset, build_train_transform, build_val_transform
from src.utils.config import load_config

# Load config
config = load_config('configs/base.yaml')

# Build transforms
train_transform = build_train_transform(
    img_size=config['data']['img_size'],
    use_geometric=config['augmentation']['use_geometric'],
    use_fog=config['augmentation']['use_fog'],
    use_rain=config['augmentation']['use_rain'],
    # ... other flags
    config=config['augmentation'],
)

val_transform = build_val_transform(img_size=config['data']['img_size'])

# Create dataset
train_dataset = UAVDetectionDataset(
    root_dir=config['data']['root_dir'],
    transforms=train_transform,
)

# Use with DataLoader
from torch.utils.data import DataLoader
from src.data.dataset_uav import collate_fn

train_loader = DataLoader(
    train_dataset,
    batch_size=config['data']['batch_size'],
    shuffle=True,
    num_workers=config['data']['num_workers'],
    collate_fn=collate_fn,
)
```

## Configuration

Edit `configs/base.yaml` to configure:
- Data paths and parameters
- Augmentation settings (geometric and weather augmentations)
- Model architecture (to be added in Step 3)
- Training hyperparameters

Use `configs/baseline.yaml` for minimal configuration (no robustness modules).
Use `configs/robust.yaml` for full configuration with all robustness features.

## Augmentations

### Geometric Augmentations
- Horizontal flip
- Random rotation (±10°)
- Random scaling
- Resize and padding to target size

### Weather/Robustness Augmentations
- **Fog/Haze**: Simulates atmospheric fog
- **Rain**: Adds rain streaks
- **Blur**: Motion blur and Gaussian blur
- **Noise**: Gaussian or salt-and-pepper noise
- **Brightness/Contrast**: Simulates varying lighting
- **Shadows**: Random shadow effects

All augmentations can be enabled/disabled via config flags.

## Model Architecture

The framework implements a lightweight YOLO-style detector:

### Backbone (`src/models/backbone.py`)
- **CSP-style** (Cross-Stage Partial) backbone
- Multi-scale feature extraction:
  - P3: 1/8 resolution (for small objects)
  - P4: 1/16 resolution
  - P5: 1/32 resolution
- Configurable width and depth multipliers for model scaling
- Optional depthwise separable convolutions for efficiency

### Neck (`src/models/neck.py`)
- **FPN** (Feature Pyramid Network) or **PAN** (Path Aggregation Network)
- Fuses multi-scale features for better small object detection
- Top-down (FPN) or top-down + bottom-up (PAN) pathways

### Head (`src/models/head.py`)
- **Anchor-free** detection head
- Predicts at each spatial location:
  - Bounding box (4 params: x_center, y_center, width, height)
  - Objectness score (1 param)
  - Class logits (N classes)
- Separate or shared heads per scale

### Usage

```python
from src.models import build_model
from src.utils.config import load_config

# Load config
config = load_config('configs/base.yaml')

# Build model
model = build_model(
    config=config,
    num_classes=config['classes']['num_classes'],
    img_size=config['data']['img_size'],
)

# Forward pass
images = torch.randn(2, 3, 640, 640)  # (B, C, H, W)
predictions = model(images, mode='val')

# Access predictions at each scale
for scale in ['P3', 'P4', 'P5']:
    bbox = predictions[scale]['bbox']  # (B, 4, H, W)
    obj = predictions[scale]['obj']    # (B, 1, H, W)
    cls = predictions[scale]['cls']    # (B, num_classes, H, W)
```

## Robustness Modules

The framework includes optional robustness modules for adverse weather and complex backgrounds:

### Background Suppression Module
- Uses spatial + channel attention to suppress uniform background regions
- Helps focus on foreground objects, especially small ones
- Residual-style: `x = x + alpha * attention(x)`

### Frequency Enhancement Module
- Lightweight frequency-domain enhancement for small objects
- Learnable high-pass filtering to emphasize edges and textures
- Useful under blur, haze, or noise conditions
- Only adds ~100K parameters per scale

### Condition-Aware Modulation
- Predicts weather/lighting condition from input image
- Uses FiLM (Feature-wise Linear Modulation) to adapt features
- Modulates features based on predicted condition
- Can be applied at multiple points in the network

### Usage

Enable robustness modules in config:

```yaml
robust_modules:
  use_background_suppression: true
  use_frequency_enhancement: true
  use_condition_aware: true
```

All modules are fully optional - disabling them gives the baseline model.

## Training & Evaluation

### Training

```bash
# Train baseline model
python src/train.py --config configs/baseline.yaml

# Train robust model (with robustness modules)
python src/train.py --config configs/robust.yaml

# Resume from checkpoint
python src/train.py --config configs/base.yaml --resume checkpoints/experiment/checkpoint_epoch_50.pth

# Custom experiment name
python src/train.py --config configs/base.yaml --experiment my_experiment
```

Or use the provided scripts:
```bash
bash scripts/train_baseline.sh
bash scripts/train_robust.sh
```

### Evaluation

```bash
# Evaluate model
python src/eval.py \
    --config configs/base.yaml \
    --checkpoint checkpoints/experiment/best_model.pth \
    --output results/eval_results.json
```

Or use the script:
```bash
bash scripts/eval_on_testset.sh checkpoints/experiment/best_model.pth
```

The evaluation computes:
- Overall mAP (mean Average Precision)
- mAP broken down by object size (small/medium/large)
- mAP broken down by condition (if metadata available)

## Loss Functions

The framework implements:
- **Box loss**: CIoU, GIoU, or IoU (configurable)
- **Objectness loss**: BCE or Focal loss
- **Classification loss**: BCE or Focal loss

Loss weights and types can be configured in the config file.

