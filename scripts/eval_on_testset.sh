#!/bin/bash

# Evaluate model on test set

CHECKPOINT=$1  # Path to checkpoint
CONFIG=${2:-configs/base.yaml}  # Config file (default: base.yaml)
OUTPUT=${3:-results/eval_results.json}  # Output file (default: results/eval_results.json)

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: $0 <checkpoint_path> [config_file] [output_file]"
    exit 1
fi

python src/eval.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --device cuda \
    --output "$OUTPUT"

