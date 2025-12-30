# UAV YOLO Object Detection Framework - Copilot Instructions

## Project Overview

This is a modular PyTorch framework for YOLO-style object detection optimized for UAV/aerial imagery with robustness to small objects and adverse weather conditions. It uses a **Backbone → Neck → Head** architecture with optional robustness modules.

## Architecture Pattern

### Model Structure (Core Components)

1. **Backbone** (`src/models/backbone.py`): CSP-style CNN extracting multi-scale features (P3, P4, P5)
2. **Neck** (`src/models/neck.py`): FPN/PAN-style feature fusion across scales  
3. **Head** (`src/models/head.py`): Multi-scale YOLO-style predictions (bbox, objectness, class)
4. **YOLOModel** (`src/models/yolo_model.py`): Orchestrates backbone→neck→head pipeline

```python
model = build_model(
    backbone={'type': 'csp', 'depth_mult': 0.5, ...},
    neck={'type': 'fpn', ...},
    head={'num_classes': 3, ...},
    robust_modules_config={...}  # Optional
)
output = model(images, targets=None, mode='inference')
# Returns: {'P3': {bbox, obj, cls}, 'P4': {...}, 'P5': {...}}
```

### Robustness Module Integration

Modular architecture allows selective insertion after backbone stages (P3, P4, P5):

- **BackgroundSuppressionModule**: Channel/spatial attention to suppress uniform backgrounds
- **FrequencyEnhancementModule**: Frequency-domain processing to enhance small objects
- **ConditionAwareModulation**: Weather condition detection → adaptive feature modulation
- **ConditionPredictor**: Infers weather condition from raw image

Each robust module is optional (disabled via config) and uses residual connections with small alphas (~0.1-0.2).

## Configuration System

All configuration via YAML in `configs/`:
- `base.yaml`: Data paths, augmentation, model architecture, optimizer settings
- Specialized configs: `baseline.yaml` (no robustness), `robust.yaml` (full features), `ablation_*.yaml`

Key config structure:
```yaml
data:
  root_dir: "data_light"
  batch_size: 16
  img_size: 640
augmentation:
  use_fog: true
  use_rain: true
  use_geometric: true
training:
  optimizer: {type: adamw, lr: 0.001}
  scheduler: {type: cosine}
model:
  backbone: {type: csp, ...}
  neck: {type: fpn, ...}
  head: {...}
robust_modules:
  use_background_suppression: true
  use_frequency_enhancement: true
  use_condition_aware: true
```

Load with: `config = load_config('configs/base.yaml')` then merge: `config = merge_configs(base, override)`

## Data Pipeline

- **Dataset Format**: YOLO format (normalized box coordinates per image)
- **UAVDetectionDataset** (`src/data/dataset_uav.py`): Handles loading, augmentation, returns `(image, {'boxes': tensor, 'labels': tensor})`
- **Augmentations**: Geometric (rotation, scale, flip) + Weather (fog, rain, blur, noise)
- **Batch Format**: Use custom `collate_fn` which returns `{'images': Tensor, 'boxes': List, 'labels': List}`

```python
from src.data import UAVDetectionDataset, build_train_transform
dataset = UAVDetectionDataset('data_light', transforms=build_train_transform(config))
loader = DataLoader(dataset, collate_fn=collate_fn)  # Important: use custom collate
```

## Training Workflow

**Main entry**: `src/train.py` with config-driven workflow:

1. Load config → build model, optimizer, scheduler, loss
2. Training loop: forward pass → loss computation → backward → optimizer step
3. Periodic validation & checkpointing (best model saved as `best_model.pth`)
4. Results saved to `checkpoints/<experiment_name>/`

Key patterns:
- Loss computation: `DetectionLoss` assigns targets to P3/P4/P5 scales based on object size
- Logging: Metrics tracked during training (loss components, mAP, class-wise metrics)
- Device handling: Auto GPU if available, fallback to CPU

**Running training**:
```bash
python -m src.train --config configs/base.yaml --device cuda --experiment my_exp
```

Or use provided scripts in `scripts/run_*.sh` for standard experiments.

## Evaluation & Inference

**Evaluation** (`src/eval.py`): Computes mAP, precision, recall on validation/test set
- Uses custom `collate_fn` for batch handling
- Post-processes predictions (NMS, confidence thresholding)
- Outputs metrics JSON + optional visualizations

**Script usage**:
```bash
bash scripts/eval_on_testset.sh <model_weights> [config_path] [output_json]
```

## Experiment Framework

Pre-configured experiment scripts in `scripts/`:
- `run_baseline.sh`: Standard YOLO without robustness modules
- `run_robust.sh`: Full robustness pipeline (all modules enabled)
- `run_lightweight.sh`: Lighter backbone with robustness
- `run_ablation_*.sh`: Isolated feature testing (weather aug only, BG suppression only, etc.)
- `run_comparison.sh`: YOLOv8n baseline using ultralytics

All scripts follow pattern: config file → `--experiment` name → results in `checkpoints/`

## Key Implementation Details

### Loss Function Hierarchy
- **Scale Assignment**: Small objects (P3), medium (P4), large (P5) based on box size thresholds
- **Box Loss**: GIoU/CIoU options (asymmetric loss preferred for small objects)
- **Class Loss**: BCE with optional focal loss weights for imbalance

### Multi-Scale Features
Backbone outputs dict with keys `{'P3', 'P4', 'P5'}` where P3 is highest resolution (1/8 stride).

### Robustness Design
- Modules maintain feature dimensions (in_channels == out_channels)
- Alpha parameters control residual contribution (train jointly, don't freeze)
- Condition predictor operates on full image before backbone

## Common Patterns & Conventions

1. **Config-driven**: Always parameterize architectural decisions in YAML before hardcoding
2. **Device-agnostic**: Use `config['device']` or detect GPU availability, support CPU fallback
3. **Modularity**: Component builders (`build_backbone`, `build_neck`, etc.) return nn.Module instances
4. **Custom collate**: Detection framework requires special batch handling; always use provided `collate_fn`
5. **Resumable training**: Save optimizer state in checkpoints, support `--resume` flag for continuation

## Files to Understand First

- [src/train.py](../src/train.py): Training loop orchestration
- [src/models/yolo_model.py](../src/models/yolo_model.py): Model composition & forward pass
- [configs/base.yaml](../configs/base.yaml): Configuration template
- [src/data/dataset_uav.py](../src/data/dataset_uav.py): Data loading pipeline
- [experiments_guide.md](../experiments_guide.md): Detailed experiment workflows

## External Dependencies

- `torch`, `torchvision`: Core deep learning
- `ultralytics`: YOLOv8 baseline for comparisons
- `albumentations`: Geometric/weather augmentations
- `pyyaml`: Config parsing
- `numpy`, `opencv-python`: Image processing, metrics

Note: Weather augmentations (fog, rain) use custom implementations in `src/data/` (not albumentations).

## Tips for AI Agents

- When adding features, **check existing configs first** to understand parameterization style
- **Robustness modules are optional**: Use boolean flags in config, don't assume they're always enabled
- **Scale assignment matters**: P3 for small objects means different loss weighting/normalization per scale
- **Test with provided scripts**: Use `bash scripts/run_baseline.sh` before custom modifications to verify setup
- **Config merging is important**: base config + override pattern prevents duplication
