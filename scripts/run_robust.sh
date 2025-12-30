#!/bin/bash
# Experiment: Full Robust Model
#
# Trains the full model with all robust modules and weather augmentations enabled.
# This represents our proposed model.

python src/train.py \
    --config configs/robust.yaml \
    --device cuda \
    --experiment "robust_full"
