#!/bin/bash

# Check if dataset name is provided as an argument
if [ -z "$1" ]; then
    echo "❌ Error: You must provide the dataset name as an argument."
    echo "Usage: $0 <FILENAME> [SEED]"
    echo "Example: $0 oitaven"
    echo "Example with seed: $0 oitaven 42"
    exit 1
fi

# Dataset name (it can be: ermidas, oitaven, eiras, ...)
FILENAME="$1"

# Optional seed parameter
SEED="$2"

# Build dataset suffix based on whether seed is provided
if [ -n "$SEED" ]; then
    DATASET_SUFFIX="_${SEED}"
else
    DATASET_SUFFIX=""
    SEED=0 # default seed to 0 if not provided
fi

# Uppercase version for network paths
FILENAME_UPPER=$(echo "$FILENAME" | tr '[:lower:]' '[:upper:]')

# Derived paths based on dataset name (with optional seed suffix)
INPUT_PATH="data"
DATA_ZIP="$INPUT_PATH/${FILENAME}/${FILENAME}_test${DATASET_SUFFIX}.zip"

#############################
# USE CASE 1: USE DIRECTORY #
#############################

# List of network paths (relative paths) with optional seed suffix
EXPERIMENTS=(
    "training-runs/${FILENAME_UPPER}_TRAIN${DATASET_SUFFIX}/00000-stylegan3-t-${FILENAME}_train${DATASET_SUFFIX}-gpus1-batch8-gamma0.125"
    "training-runs/${FILENAME_UPPER}_TRAIN${DATASET_SUFFIX}/00001-stylegan3-t-${FILENAME}_train${DATASET_SUFFIX}-gpus1-batch8-gamma0.125"
)

# Run the evaluation command for each experiment
for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    python evaluate_pixel_accuracy.py \
        --experiment-dir="$EXPERIMENT" \
        --filename="$FILENAME" \
        --dataset-seed="$SEED" \
        --data-zip="$DATA_ZIP" \
        --batch-size=256 \
        --selection-method="best_val_aa" \
        --output-csv="results_aa.csv"
done

################################
# USE CASE 2: USE NETWORK PATH #
################################

NETWORKS=(
    "training-runs/${FILENAME_UPPER}_TRAIN${DATASET_SUFFIX}/00000-stylegan3-t-${FILENAME}_train${DATASET_SUFFIX}-gpus2-batch128-gamma0.125/network-snapshot-000070.pkl"
    "training-runs/${FILENAME_UPPER}_TRAIN${DATASET_SUFFIX}/00000-stylegan3-t-${FILENAME}_train${DATASET_SUFFIX}-gpus2-batch128-gamma0.125/network-snapshot-000173.pkl"
)

# # Run the evaluation command for each network
# for NETWORK in "${NETWORKS[@]}"; do
#     python evaluate_pixel_accuracy.py \
#         --network-pkl="$NETWORK" \
#         --filename="$FILENAME" \
#         --dataset-seed="$SEED" \
#         --data-zip="$DATA_ZIP"
# done