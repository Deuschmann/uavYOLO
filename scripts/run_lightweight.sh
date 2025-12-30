#!/bin/bash
# Experiment: Lightweight Robust Model
#
# Trains a lightweight version of the full robust model.
# It uses a smaller backbone architecture but keeps all robust modules
# and weather augmentations enabled.

python src/train.py \
    --config configs/lightweight.yaml \
    --device cuda \
    --experiment "lightweight_robust"
