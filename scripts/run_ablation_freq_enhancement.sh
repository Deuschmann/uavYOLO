#!/bin/bash
# Ablation Experiment: Frequency Enhancement Module
#
# Trains a model with ONLY the frequency enhancement module enabled.
# All weather augmentations are OFF.
# This isolates the performance contribution of the frequency enhancement module.

python src/train.py \
    --config configs/ablation_freq_enhancement.yaml \
    --device cuda \
    --experiment "ablation_freq_enhancement"
