#!/bin/bash

# Train baseline model (no robustness modules)

python src/train.py \
    --config configs/baseline.yaml \
    --device cuda \
    --experiment baseline_experiment

