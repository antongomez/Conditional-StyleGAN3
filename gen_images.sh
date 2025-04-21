#!/bin/bash

# Check if the script received an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <network_file>"
    exit 1
fi

# Common parameters
OUTDIR="out"
TRUNC=1
BASE_SEED=42
IMAGES_PER_CLASS=10  # Number of images per class
NUM_CLASSES=10  # Number of classes (0-9)

network_path="$1"
if [ ! -f "$network_path" ]; then
    echo "Error: Network file $network_path not found!"
    exit 1
fi

OUTDIR="$OUTDIR/$(basename $network_path .pkl)"
mkdir -p "$OUTDIR"
if [ ! -d "$OUTDIR" ]; then
    echo "Error: Failed to create output directory $OUTDIR!"
    exit 1
fi

# Generate seeds for this class
SEEDS=""
for ((i=0; i<IMAGES_PER_CLASS; i++)); do
    SEEDS+=$((BASE_SEED + i))","
done
SEEDS=${SEEDS%,}  # Remove the last comma

# Generate 
CLASSES=""
for ((i=0; i<NUM_CLASSES; i++)); do
    CLASSES+=$i","
done
CLASSES=${CLASSES%,}  # Remove the last comma

# Execute the command to generate images
python gen_images.py --outdir="$OUTDIR" --trunc="$TRUNC" --seeds="$SEEDS" --classes="$CLASSES" --network="$network_path"