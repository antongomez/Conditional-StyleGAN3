#!/bin/bash

# Force use of dot as decimal separator
export LC_NUMERIC=C

is_power_of_two() {
  local n=$1
  (( n > 0 )) && (( (n & (n-1)) == 0 ))
}

# Default values
EPOCHS=0
AUTOEN_EPOCHS=0
AUTOEN_PATIENCE="0"
AUTOEN_MIN_DELTA="0.0"
if [ -n "$STORE" ]; then
  INPUT_PATH="$STORE/data" # we are in cesga, use $STORE/data as input path
else
  INPUT_PATH="./data" # default input path (for local runs)
fi
FILENAME="oitaven"
SEED=""
GPUS=2
BATCH=128
CFG="stylegan3-t"
CLS_WEIGHT=0.1
CLS_START=0
TICK=10
SNAP=1
SAVE_ALL_SNAPS=False
SAVE_ALL_FAKES=False
DRY_RUN=False
GAMMA=0.125
EVAL_MANIFOLD=True
EVAL_CLASSIFICATION=True
JOIN_RESULTS=True

JUDGE_MODEL_PATH="manifold_metrics/oitaven_judge_32.pth"  # Path to the judge model for manifold metrics
MANIFOLD_INTERVAL=1  # Interval (in ticks) to evaluate manifold metrics
MANIFOLD_NUM_IMAGES=200  # Number of images per class to generate for manifold metrics
MANIFOLD_EXPERIMENTS=1  # Number of manifold metric experiments to average

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
    --autoen-epochs=*)
      AUTOEN_EPOCHS="${arg#*=}"
      shift
      ;;
    --autoen-patience=*)
      AUTOEN_PATIENCE="${arg#*=}"
      shift
      ;;
    --autoen-min-delta=*)
      AUTOEN_MIN_DELTA="${arg#*=}"
      shift
      ;;
    --input-path=*)
      INPUT_PATH="${arg#*=}"
      shift
      ;;
    --filename=*)
      FILENAME="${arg#*=}"
      shift
      ;;
    --seed=*)
      SEED="${arg#*=}"
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
    --batch-gpu=*)
      BATCH_GPU="${arg#*=}"
      shift
      ;;
    --cls-weight=*)
      CLS_WEIGHT="${arg#*=}"
      shift
      ;;
    --cls-start=*)
      CLS_START="${arg#*=}"
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
    --save-all-snaps)
      SAVE_ALL_SNAPS=True
      shift
      ;;
    --save-all-fakes)
      SAVE_ALL_FAKES=True
      shift
      ;;
    --dry-run)
      DRY_RUN=True
      shift
      ;;
    --eval-manifold)
      EVAL_MANIFOLD=True
      shift
      ;;
    --eval-classification)
      EVAL_CLASSIFICATION=True
      shift
      ;;
    --join-results)
      JOIN_RESULTS=True
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
if [ -n "$SEED" ]; then
  DATASET="${FILENAME}_train_${SEED}"
  DATASET_VAL="${FILENAME}_val_${SEED}"
  DATASET_ZIP="$INPUT_PATH/${FILENAME}/${DATASET}.zip"
  DATASET_VAL_ZIP="$INPUT_PATH/${FILENAME}/${DATASET_VAL}.zip"
else
  DATASET="${FILENAME}_train"
  DATASET_VAL="${FILENAME}_val"
  DATASET_ZIP="$INPUT_PATH/${FILENAME}/${DATASET}.zip"
  DATASET_VAL_ZIP="$INPUT_PATH/${FILENAME}/${DATASET_VAL}.zip"
fi

DATASET_TMP_DIR="$INPUT_PATH/${FILENAME}/tmp"
DATASET_JSON="${DATASET_TMP_DIR}/dataset.json"
OUTDIR="./training-runs/$(echo ${DATASET} | tr '[:lower:]' '[:upper:]')"

# Set BATCH_GPU if not specified
if [ -z "$BATCH_GPU" ]; then
  BATCH_GPU=$((BATCH / GPUS))
fi

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

  if [ "$AUTOEN_EPOCHS" -gt 0 ]; then
    AUTOEN_KIMG=$(( NUM_IMAGES  * AUTOEN_EPOCHS / 1000 ))
    echo ">>> Auto kimg for adversarial training: $AUTOEN_KIMG (for $AUTOEN_EPOCHS epochs)"
    if [ "$AUTOEN_KIMG" -gt "$NUM_KIMG" ]; then
      echo ">>> Warning: Auto kimg ($AUTOEN_KIMG) is greater than total kimg ($NUM_KIMG). Adjusting auto kimg to total kimg."
      AUTOEN_KIMG=$NUM_KIMG
    fi
  else 
    AUTOEN_KIMG=0
  fi

else
  NUM_KIMG=1500  # Default value
  echo ">>> Using default kimg: $NUM_KIMG"
fi

# Print configuration
echo "Configuration:"
echo "  cfg                 = ${CFG}"
echo "  epochs              = ${EPOCHS}"
echo "  kimg                = ${NUM_KIMG}"
echo "  auto-epochs         = ${AUTOEN_EPOCHS}"
echo "  autoen-kimg         = ${AUTOEN_KIMG}"
echo "  autoen-patience     = ${AUTOEN_PATIENCE}"
echo "  autoen-min-delta    = ${AUTOEN_MIN_DELTA}"
echo "  dataset             = ${DATASET}"
echo "  dataset-val         = ${DATASET_VAL}"
echo "  gpus                = ${GPUS}"
echo "  batch               = ${BATCH}"
echo "  batch-gpu           = ${BATCH_GPU}"
echo "  cls-weight          = ${CLS_WEIGHT}"
echo "  cls-start           = ${CLS_START}"
echo "  tick                = ${TICK}"
echo "  snap                = ${SNAP}"
echo "  save-all-snaps      = ${SAVE_ALL_SNAPS}"
echo "  save-all-fakes      = ${SAVE_ALL_FAKES}"
echo "  judge-model-path   = ${JUDGE_MODEL_PATH}"
echo "  manifold-interval   = ${MANIFOLD_INTERVAL}"
echo "  manifold-num-images = ${MANIFOLD_NUM_IMAGES}"
echo "  manifold-experiments= ${MANIFOLD_EXPERIMENTS}"
echo "  dry-run             = ${DRY_RUN}"
echo "  gamma               = ${GAMMA}"
echo "  eval-manifold       = ${EVAL_MANIFOLD}"
echo "  eval-classification = ${EVAL_CLASSIFICATION}"
echo "  join-results        = ${JOIN_RESULTS}"
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
python train.py --outdir="$OUTDIR" --cfg="$CFG" --cond=True --gamma=0.125  \
  --data="$DATASET_ZIP" --data-val=$DATASET_VAL_ZIP  \
  --gpus=$GPUS --batch=$BATCH --batch-gpu=$BATCH_GPU \
  --kimg=$NUM_KIMG --autoen-kimg=$AUTOEN_KIMG --tick=$TICK --snap=$SNAP \
  --autoen-patience=$AUTOEN_PATIENCE --autoen-min-delta=$AUTOEN_MIN_DELTA \
  --cls-weight=$CLS_WEIGHT --cls-start=$CLS_START \
  --save-all-snaps=$SAVE_ALL_SNAPS --save-all-fakes=$SAVE_ALL_FAKES --use-label-map=True \
  --mirror=False --metrics=none \
  --judge-model-path=$JUDGE_MODEL_PATH \
  --manifold-interval=$MANIFOLD_INTERVAL \
  --manifold-num-images=$MANIFOLD_NUM_IMAGES \
  --manifold-experiments=$MANIFOLD_EXPERIMENTS \
  $OTHER_ARGS

echo ">>> Training completed. Models and logs are saved in: $run_dir"

# Run evaluation if not disabled
if [[ "${EVAL_MANIFOLD,,}" == "true" ]]; then
  EXPERIMENT_DIR="$run_dir"
  if [ -n "$SEED" ]; then
    DATA_TEST_ZIP="data/${FILENAME}/${FILENAME}_test_${SEED}.zip"
  else
    DATA_TEST_ZIP="data/${FILENAME}/${FILENAME}_test.zip"
    SEED=0 # Set SEED to 0 if not provided (evaluate_pixel_accuracy.py requires an integer, put a 0 has the same behaviour as no seed)
  fi

  SELECTION_METHOD="best_val_aa"
  MANIFOLD_RESULTS_FILE="0_aa_lambda_manifold_results.csv"
  CLASSIFICATION_RESULTS_FILE="0_aa_lambda_results.csv"

  # Run the manifold metrics evaluation
  python calculate_manifold_metrics.py \
      --experiment-dir="$EXPERIMENT_DIR" \
      --input-path="$INPUT_PATH" \
      --filename="$FILENAME" \
      --dataset-seed="$SEED" \
      --selection-method="$SELECTION_METHOD" \
      --fid-model-path="$JUDGE_MODEL_PATH" \
      --output-csv="$MANIFOLD_RESULTS_FILE"
else
  echo "Manifold evaluation skipped as per --no-eval-manifold flag."
fi

if [[ "${EVAL_CLASSIFICATION,,}" == "true" ]]; then
  # Run the evaluation command for each network
  python evaluate_pixel_accuracy.py \
      --experiment-dir="$EXPERIMENT_DIR" \
      --input-path="$INPUT_PATH" \
      --filename="$FILENAME" \
      --dataset-seed="$SEED" \
      --data-zip="$DATA_TEST_ZIP" \
      --selection-method="$SELECTION_METHOD" \
      --output-csv="$CLASSIFICATION_RESULTS_FILE"
else
  echo "Classification evaluation skipped as per --no-eval-classification flag."
fi

[[ "${JOIN_RESULTS,,}" == "true" ]]; then
  JOINED_RESULTS_FILE="joined_results.csv"
  python scripts/join.py \
      --csv1="$CLASSIFICATION_RESULTS_FILE" \
      --csv2="$MANIFOLD_RESULTS_FILE" \
      --output="$JOINED_RESULTS_FILE"
else
    echo "Skipping joining of results."
fi

exit 0
##############