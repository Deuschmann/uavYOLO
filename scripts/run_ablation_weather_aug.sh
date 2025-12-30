#!/bin/bash
# Ablation Experiment: Weather Augmentations
#
# Trains a model with ONLY the weather augmentations enabled.
# All robust modules are OFF.
# This isolates the performance contribution of the data augmentation strategy.

python src/train.py \
    --config configs/ablation_weather_aug_only.yaml \
    --device cuda \
    --experiment "ablation_weather_aug"
