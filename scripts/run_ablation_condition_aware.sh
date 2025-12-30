#!/bin/bash
# Ablation Experiment: Condition-Aware Module
#
# Trains a model with ONLY the condition-aware module enabled.
# All weather augmentations are OFF.
# This isolates the performance contribution of the condition-aware module.

python src/train.py \
    --config configs/ablation_condition_aware.yaml \
    --device cuda \
    --experiment "ablation_condition_aware"
