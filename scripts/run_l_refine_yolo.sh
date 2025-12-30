#!/bin/bash
# Experiment: L_Refine_YOLO
#
# Trains the advanced L_Refine_YOLO model with improved backbone (LDown),
# deep feature blocks (IRDCB), and adaptive scale fusion (ASFB).

python src/train.py \
    --config configs/l_refine_yolo.yaml \
    --device cpu \
    --experiment "l_refine_yolo"
