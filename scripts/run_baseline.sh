#!/bin/bash
# Experiment: Baseline
#
# Trains a standard model with no robust modules and no weather augmentations.
# This serves as the performance baseline to compare against.

python src/train.py \
    --config configs/baseline.yaml \
    --device cuda \
    --experiment "baseline"
