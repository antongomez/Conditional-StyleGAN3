#!/bin/bash

# Force use of dot as decimal separator
export LC_NUMERIC=C

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
CFG="stylegan3-t"
TICK=10
SNAP=1
DRY_RUN=False
GAMMA=0.125

# Parse arguments
for arg in "$@"; do
  case $arg in
    --cfg=*)
      CFG="${arg#*=}"
      shift
      ;;
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
    --tick=*)
      TICK="${arg#*=}"
      shift
      ;;
    --snap=*)
      SNAP="${arg#*=}"
      shift
      ;;
    --dry-run=*)
      DRY_RUN="${arg#*=}"
      shift
      ;;
    *)
      OTHER_ARGS+="$arg "
      ;;
  esac
done

# Validate that --cfg has an allowed value
case "${CFG}" in
  stylegan3-t|stylegan3-r|stylegan2)
    # valid value
    ;;
  *)
    echo "ERROR: invalid value for --cfg: '${CFG}'"
    echo "       Allowed values: stylegan3-t, stylegan3-r, stylegan2"
    exit 1
    ;;
esac

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
  echo ">>> Using default kimg: $NUM_KIMG"
fi

# Print configuration
echo "Configuration:"
echo "  cfg          = ${CFG}"
echo "  epochs       = ${EPOCHS}"
echo "  kimg         = ${NUM_KIMG}"
echo "  dataset      = ${DATASET}"
echo "  dataset-val  = ${DATSET_VAL}"
echo "  gpus         = ${GPUS}"
echo "  batch        = ${BATCH}"
echo "  batch-gpu    = ${BATCH_GPU}"
echo "  tick         = ${TICK}"
echo "  snap         = ${SNAP}"
echo "  dry-run      = ${DRY_RUN}"
if [ ${#OTHER_ARGS[@]} -gt 0 ]; then
  echo "  other args   = ${OTHER_ARGS[*]}"
else
  echo "  other args   = (none)"
fi

# Get all subdirectories inside $OUTDIR that start with numbers
prev_run_dirs=($(find "$OUTDIR" -maxdepth 1 -type d -printf "%f\n" | grep -E '^[0-9]+' || true))

echo "Found ${#prev_run_dirs[@]} previous run directories."

# If there are previous run directories, extract numeric prefixes; otherwise use -1
if [[ ${#prev_run_dirs[@]} -gt 0 ]]; then
    prev_run_ids=()
    for d in "${prev_run_dirs[@]}"; do
        id=$(echo "$d" | grep -oE '^[0-9]+')
        prev_run_ids+=("$id")
    done

    if [[ ${#prev_run_ids[@]} -gt 0 ]]; then
        # Remove leading zeros to avoid octal interpretation
        max_id=$(printf "%s\n" "${prev_run_ids[@]}" | sed 's/^0*//' | sort -n | tail -n 1)
        # If empty (e.g. "00000"), force to 0
        [[ -z "$max_id" ]] && max_id=0
    fi
else
    max_id=-1
fi

# Compute new run ID (+1 from max)
cur_run_id=$((max_id + 1))

# Format run directory with zero-padded ID (5 digits) and other params
run_dir=$(LC_NUMERIC=C printf "%s/%05d-%s-%s-gpus%d-batch%d-gamma%.3f" \
    "$OUTDIR" "$cur_run_id" "${CFG}" "${DATASET}" "$GPUS" "$BATCH" "$GAMMA")

echo ">>> Run directory will be: $run_dir"

# Make sure it does not already exist
if [[ -e "$run_dir" ]]; then
    echo "ERROR: $run_dir already exists!"
    exit 1
fi

# Check if dry run is enabled
if [[ "${DRY_RUN,,}" == "true" ]]; then
  echo "Dry run enabled. Exiting without training."
  exit 0
fi

# Run training
python train.py --outdir="$OUTDIR" --cfg="$CFG" --data="$DATASET_ZIP" --cond=True \
  --gpus=$GPUS --batch=$BATCH --gamma=0.125 --batch-gpu=$BATCH_GPU \
  --kimg=$NUM_KIMG --tick=$TICK --snap=$SNAP --metrics=none \
  --mirror=False --data-val=$DATASET_VAL_ZIP --use-label-map=True $OTHER_ARGS

echo ">>> Training completed. Models and logs are saved in: $run_dir"

FILENAME="${DATASET%_train}"
EXPERIMENT_DIR="$run_dir"
INPUT_DIR="data/${FILENAME}"
DATA_ZIP="data/${FILENAME}_test.zip"
OUTPUT_DIR="data/${FILENAME}/patches"

# Run the evaluation command for each network
python evaluate_pixel_accuracy.py \
    --experiment-dir="$EXPERIMENT_DIR" \
    --input-dir="$INPUT_DIR" \
    --filename="$FILENAME" \
    --data-zip="$DATA_ZIP" \
    --output-dir="$OUTPUT_DIR" \
    --write-report \
    --remove \
    --output-csv="experiments_accuracies.csv"