#!/bin/bash

# Dataset name (can be: ermidas, oitaven, eiras, ...)
DATASET_NAME="ermidas"

# Uppercase version for network paths
DATASET_NAME_UPPER=$(echo "$DATASET_NAME" | tr '[:lower:]' '[:upper:]')

# List of network paths (relative paths)
NETWORKS=(
    "training-runs/${DATASET_NAME_UPPER}_TRAIN/00000-stylegan3-t-${DATASET_NAME}_train-gpus2-batch64-gamma0.125/network-snapshot-001266.pkl"
    "training-runs/${DATASET_NAME_UPPER}_TRAIN/00001-stylegan3-t-${DATASET_NAME}_train-gpus2-batch64-gamma0.125/network-snapshot-001085.pkl"
    "training-runs/${DATASET_NAME_UPPER}_TRAIN/00002-stylegan3-t-${DATASET_NAME}_train-gpus2-batch64-gamma0.125/network-snapshot-001175.pkl"
    "training-runs/${DATASET_NAME_UPPER}_TRAIN/00003-stylegan3-t-${DATASET_NAME}_train-gpus2-batch64-gamma0.125/network-snapshot-001698.pkl"
)

# Derived paths based on dataset name
INPUT_DIR="data/${DATASET_NAME}"
DATA_ZIP="data/${DATASET_NAME}_test.zip"
OUTPUT_DIR="data/${DATASET_NAME}/patches"
FILENAME="${DATASET_NAME}"

# Run the evaluation command for each network
for NETWORK in "${NETWORKS[@]}"; do
    python evaluate_pixel_accuracy.py \
        --network="$NETWORK" \
        --input-dir="$INPUT_DIR" \
        --filename="$FILENAME" \
        --data-zip="$DATA_ZIP" \
        --output-dir="$OUTPUT_DIR" \
        --write-report
done