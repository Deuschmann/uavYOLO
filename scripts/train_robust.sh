#!/bin/bash

# Train robust model (with all robustness modules)

python src/train.py \
    --config configs/robust.yaml \
    --device cuda \
    --experiment robust_experiment

