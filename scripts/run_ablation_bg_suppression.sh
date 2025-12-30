#!/bin/bash
# Ablation Experiment: Background Suppression
#
# Trains a model with ONLY the background suppression module enabled.
# All weather augmentations are OFF.
# This isolates the performance contribution of the background suppression module.

python src/train.py \
    --config configs/ablation_bg_suppression.yaml \
    --device cuda \
    --experiment "ablation_bg_suppression"
