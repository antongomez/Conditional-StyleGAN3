#!/bin/bash

is_power_of_two() {
  local n=$1
  (( n > 0 )) && (( (n & (n-1)) == 0 ))
}

# Default values
EPOCHS=0
DATASET="oitaven_train"
DATSET_VAL="oitaven_val"
GPUS=1 
BATCH=64

# Parse arguments
for arg in "$@"; do
  case $arg in
    --epochs=*)
      EPOCHS="${arg#*=}"
      shift
      ;;
    --dataset=*)
      DATASET="${arg#*=}"
      shift
      ;;
    --dataset-val=*)
      DATSET_VAL="${arg#*=}"
      shift
      ;;
    --gpus=*)
      GPUS="${arg#*=}"
      shift
      ;;
    --batch=*)
      BATCH="${arg#*=}"
      shift
      ;;
    *)
      OTHER_ARGS+="$arg "
      ;;
  esac
done

if ! is_power_of_two "$BATCH"; then
  echo "* Error: --batch must be a power of two."
  exit 1
fi

# Paths
DATASET_ZIP="data/${DATASET}.zip"
DATASET_VAL_ZIP="data/${DATSET_VAL}.zip"
DATASET_TMP_DIR="data/tmp"
DATASET_JSON="${DATASET_TMP_DIR}/dataset.json"
OUTDIR="./training-runs/$(echo ${DATASET} | tr '[:lower:]' '[:upper:]')"
BATCH_GPU=$(( BATCH / GPUS ))

# Calculate kimg if epochs are specified
if [ "$EPOCHS" -gt 0 ]; then

  # Create tmp directory if it doesn't exist
  if [ ! -d "$DATASET_TMP_DIR" ]; then
    mkdir -p "$DATASET_TMP_DIR"
  fi

  # Extract dataset.json if it doesn't exist
  if [ ! -f "$DATASET_JSON" ]; then
    echo "* Extracting dataset.json from ${DATASET_ZIP}..."
    unzip -oq "$DATASET_ZIP" dataset.json -d "$DATASET_TMP_DIR"
  fi

  # Count the number of images in the dataset
  # Count the number of images in the dataset
  if [ -f "$DATASET_JSON" ]; then
    NUM_IMAGES=$(python -c "import json; print(len(json.load(open('${DATASET_JSON}'))['labels']))")
    echo "* Number of images in the dataset: $NUM_IMAGES"

    # Delete dataset.json after reading it
    rm -f "$DATASET_JSON"
    rmdir "$DATASET_TMP_DIR"
  else
    echo "Error: dataset.json not found in ${DATASET_ZIP}"
    exit 1
  fi

  NUM_KIMG=$(( NUM_IMAGES  * EPOCHS / 1000 ))
  echo "* Calculated kimg: $NUM_KIMG (for $EPOCHS epochs)"

else
  NUM_KIMG=1500  # Default value
  echo "* Using default kimg: $NUM_KIMG"
fi

# Run training
python train.py --outdir="$OUTDIR" --cfg=stylegan3-t --data="$DATASET_ZIP" --cond=True \
  --gpus=$GPUS --batch=$BATCH --gamma=0.125 --batch-gpu=$BATCH_GPU \
  --kimg=$NUM_KIMG --tick=10 --snap=5 --metrics=none \
  --mirror=False --aug=noaug --data-val=$DATASET_VAL_ZIP $OTHER_ARGS