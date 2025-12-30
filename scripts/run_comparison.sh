#!/bin/bash
# Experiment: Comparison with YOLOv8
#
# Trains a standard YOLOv8n model on the same dataset.
# This serves as a comparison to the custom model.

python src/train_yolov8.py \
    --config configs/comparison.yaml \
    --device cuda
