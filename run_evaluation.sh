#!/bin/bash

# Check if dataset name is provided as an argument
if [ -z "$1" ]; then
    echo "❌ Error: You must provide the dataset name as an argument."
    echo "Usage: $0 <dataset_name>"
    echo "Example: $0 oitaven"
    exit 1
fi

# Dataset name (it can be: ermidas, oitaven, eiras, ...)
DATASET_NAME="$1"

# Uppercase version for network paths
DATASET_NAME_UPPER=$(echo "$DATASET_NAME" | tr '[:lower:]' '[:upper:]')

# List of network paths (relative paths)
EXPERIMENTS=(
    "training-runs/${DATASET_NAME_UPPER}_TRAIN/00002-stylegan3-t-${DATASET_NAME}_train-gpus1-batch32-gamma0.125"
)

# Derived paths based on dataset name
INPUT_DIR="data/${DATASET_NAME}"
DATA_ZIP="data/${DATASET_NAME}_test.zip"
OUTPUT_DIR="data/${DATASET_NAME}/patches"
FILENAME="${DATASET_NAME}"

# Run the evaluation command for each network
for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    python evaluate_pixel_accuracy.py \
        --experiment-dir="$EXPERIMENT" \
        --input-dir="$INPUT_DIR" \
        --filename="$FILENAME" \
        --data-zip="$DATA_ZIP" \
        --output-dir="$OUTPUT_DIR" \
        --write-report \
        --remove \
        --output-csv="experiments_accuracies.csv"
done
